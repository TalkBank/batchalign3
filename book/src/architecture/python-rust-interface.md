# Python/Rust Interface Reference

**Status:** Current
**Last updated:** 2026-03-17

This document describes the `batchalign_core` PyO3 module — the Rust extension
that Python code uses for all CHAT manipulation.

## Overview

`batchalign_core` is a PyO3 extension (`pyo3/`) exposing two kinds of entry points:

1. **`ParsedChat`** — a `#[pyclass]` wrapping a Rust `ChatFile` AST. Parse once,
   mutate in place via methods, serialize once.
2. **Standalone `#[pyfunction]`s** — text-in/text-out convenience wrappers, plus
   utilities (DP alignment, WER, Cantonese normalization, Rev.AI HTTP client).

Python never touches raw CHAT text. All parsing, mutation, validation, and
serialization happens in Rust through these entry points.

## Two Runtime Paths

The PyO3 interface serves two distinct runtime paths:

```
1. Server path (production):
   Rust server parses CHAT → extracts payloads → sends to a task-targeted Python worker (IPC) →
   worker runs ML model → returns tagged raw results → Rust normalizes and injects into AST

2. Python API path (pipeline_api.py):
   Python declares a list of document operations and exposes one raw provider callback →
   Rust parses CHAT, sequences operations, extracts payloads, and mutates the AST →
   Python runs ML model provider batches and returns raw dicts →
   Rust deserializes responses and injects into AST
```

The **server path** uses worker IPC (stdio JSON-lines) and never touches the
callback methods. The **Python API path** now also keeps its orchestration on
the Rust side through `run_provider_pipeline()`. It still reuses the existing
`ParsedChat.add_*` callback methods internally, but Rust synthesizes those
callbacks from a generic Python batch-provider function instead of letting
Python own the pipeline loop.

## Locked de-Pythonization target

This repository's current finish line is explicit:

- keep subprocess workers;
- keep Python only for direct ML/library calls and the thinnest glue needed to
  host those calls;
- keep provider-independent workflow logic and all CHAT-aware semantics in Rust;
- keep already-landed BA2 compatibility shims out of scope for this wave.

### Must stay Python

| Surface | Why it stays |
|---|---|
| `batchalign/worker/_main.py`, `_execute_v2.py`, `_text_v2.py`, `_protocol*.py`, `_model_loading/`, `_stanza_loading.py` | Thin worker host for Python-native runtimes, task bootstrap, and model-host execution. Rust now owns stdio op validation/dispatch, prepared-artifact lookup/read plus prepared-audio descriptor validation, text-task batch-result normalization, the ASR and forced-alignment executor control planes, and the simple prepared-audio speaker/openSMILE/AVQI executor control plane through the PyO3 bridge. |
| `batchalign/inference/{morphosyntax,utseg,translate,coref,fa,asr,speaker,opensmile,avqi}.py` | Direct model or SDK invocation. |
| `batchalign/inference/hk/` | Python-only HK/Cantonese SDK and model boundaries. |
| `batchalign/inference/audio.py`, `batchalign/inference/_tokenizer_realign.py`, `batchalign/device.py` | Worker-local helpers that only exist to feed or interpret Python-native model runtimes. |
| `batchalign/models/` | Training and model-engineering code that directly depends on Python ML libraries. |

### Move now

| Surface | Rust-owned target |
|---|---|
| `batchalign/config.py` | Production config discovery and provider-credential resolution should happen at Rust CLI/server startup; workers should receive resolved settings instead of reopening `~/.batchalign.ini`. Rev.AI and HK cloud-provider credentials now follow that boundary. |
| `batchalign/inference/benchmark.py` | WER already lives in Rust (`batchalign_core.wer_compute()`); keep this only as legacy convenience while callers migrate, and do not widen it. |
| Generic compatibility request bags (`process`, `batch_infer`) | Keep shrinking them; new work should target Rust-owned orchestration plus typed `execute_v2`, not wider Python request shapes. |

### Split / partially move

| Surface | What stays Python | What moves or stays Rust |
|---|---|---|
| `batchalign/pipeline_api.py` | Thin provider adapters and callback glue | Parsed-document loop, operation sequencing, payload semantics, batch-infer request/response adapter logic, injection, validation |
| `batchalign/runtime.py` | Worker-local environment probing such as free-threaded detection or live memory visibility | Shared command classification, memory policy, release/runtime constants, and other control-plane decisions |
| `batchalign/providers/` and worker schema mirrors | Narrow wire-type re-exports for Python callers | Schema ownership, command semantics, and orchestration policy |

### Out of scope for this wave

`batchalign.compat` is intentionally excluded from this de-Pythonization wave.
It remains a migration surface for BA2-style callers, not the target
architecture. Do not keep or add new non-ML Python logic just because the shim
still exists.

## ParsedChat API

### Construction

| Method | Description |
|--------|-------------|
| `ParsedChat.parse(chat_text)` | Strict parse (rejects on error) |
| `ParsedChat.parse_lenient(chat_text)` | Error-recovery parse (marks broken tiers) |
| `ParsedChat.build(transcript_json)` | Build from JSON transcript (ASR output) |

### Serialization and Metadata

| Method | Returns | Description |
|--------|---------|-------------|
| `serialize()` | `str` | CHAT text from AST |
| `extract_languages()` | `list[str]` | ISO 639-3 codes from `@Languages` header |
| `extract_metadata()` | `str` (JSON) | Languages, media name, media type |
| `extract_nlp_words(domain)` | `str` (JSON) | Words for NLP processing (`"mor"`, `"wor"`, `"pho"`, `"sin"`) |
| `is_no_align()` | `bool` | Whether `@Options: NoAlign` is set |
| `is_ca()` | `bool` | Whether `@Options: CA` is set |

### Validation

| Method | Returns | Description |
|--------|---------|-------------|
| `validate()` | `list[str]` | Tier alignment errors (human-readable) |
| `validate_structured()` | `str` (JSON) | Tier alignment errors (structured) |
| `validate_chat_structured()` | `str` (JSON) | Full semantic validation (headers, tiers, temporal) |
| `parse_warnings()` | `str` (JSON) | Warnings from lenient parse |

### Direct Mutations

| Method | Description |
|--------|-------------|
| `clear_morphosyntax()` | Remove all %mor/%gra tiers |
| `strip_timing()` | Remove all timing bullets and %wor tiers |
| `add_comment(text)` | Insert `@Comment:` header line |
| `add_dependent_tiers(json)` | Add custom dependent tiers to utterances |
| `add_utterance_timing(json)` | Inject ASR word timings (UTR) |
| `add_retrace_markers(lang)` | Add `[/]` retrace markers |
| `add_disfluency_markers(pauses, replacements)` | Add `&-` filled pause markers |
| `reassign_speakers(segments, lang)` | Reassign speaker codes from diarization |
| `replace_inner(other)` | Replace AST with another ParsedChat's AST |

### Morphosyntax Cache Methods

These support the server-side cache workflow (extract payloads for cache key
computation, inject cached results, extract results for cache storage):

| Method | Signature | Description |
|--------|-----------|-------------|
| `extract_morphosyntax_payloads` | `(lang, *, skipmultilang) -> str` | JSON array of per-utterance payloads |
| `inject_morphosyntax_from_cache` | `(injections_json) -> None` | Inject cached %mor/%gra strings |
| `extract_morphosyntax_strings` | `(line_indices_json) -> str` | Extract %mor/%gra for cache storage |

### Callback Methods (Rust-owned Python API path)

These methods accept a Python callable. Rust extracts data from the AST, calls
the callable with a `dict`/`list` payload, and injects the response back into
the AST. They are still used internally by the Python API path, but the
pipeline loop now lives in Rust rather than in Python helper modules.

| Method | Callback signature | Description |
|--------|-------------------|-------------|
| `add_morphosyntax(lang, fn, ...)` | `(payload_dict, lang) -> response_dict` | Per-utterance morphosyntax |
| `add_morphosyntax_batched(lang, fn, ...)` | `([payload_dict, ...], lang) -> [response_dict, ...]` | Batched morphosyntax |
| `add_forced_alignment(fn, ...)` | `(payload_dict) -> response_dict` | Per-group forced alignment |
| `add_translation(fn, ...)` | `(payload_dict) -> response_dict` | Per-utterance translation |
| `add_utterance_segmentation(fn, ...)` | `(payload_dict) -> response_dict` | Per-utterance segmentation |
| `add_utterance_segmentation_batched(fn, ...)` | `([payload_dict, ...]) -> [response_dict, ...]` | Batched segmentation |

Payload and response schemas are defined as `TypedDict`s in
`stubs/batchalign_core/__init__.pyi`.

These callback-driven mutators, plus morphosyntax cache injection, now stage
changes on a cloned `ChatFile` and only commit the mutated copy on success.
That keeps the long-lived `ParsedChat` handle transactional at the PyO3
boundary: callback failures, progress-hook failures, cache-injection failures,
and response-validation failures do not leave a partially mutated AST behind.

## Standalone Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `build_chat(json)` | `str -> str` | Build CHAT from JSON transcript |
| `parse_and_serialize(text)` | `str -> str` | Round-trip parse+serialize |
| `extract_nlp_words(text, domain)` | `(str, str) -> str` | Extract words (standalone) |
| `extract_timed_tiers(text, by_word)` | `(str, bool) -> str` | Extract timed tiers for TextGrid |
| `dp_align(payload, reference)` | `(list, list) -> str` | Hirschberg DP alignment |
| `align_tokens(original, stanza)` | `(list, list) -> list` | Map Stanza tokens to original |
| `wer_conform(words)` | `list -> list` | Normalize words for WER |
| `wer_compute(hyp, ref)` | `(list, list) -> str` | Compute WER diff metrics (returns JSON) |
| `wer_metrics(hyp, ref)` | `(list, list) -> str` | Compute structured WER metrics for the thin Python benchmark wrapper |
| `chat_terminators()` | `-> list[str]` | Valid CHAT terminators |
| `chat_mor_punct()` | `-> list[str]` | %mor punctuation tokens |
| `normalize_cantonese(text)` | `str -> str` | Simplified → HK traditional + domain fixes |
| `cantonese_char_tokens(text)` | `str -> list[str]` | Cantonese character tokenization |
| `clean_funaudio_segment_text(text)` | `str -> str` | FunASR segment cleanup before token projection |
| `funaudio_segments_to_asr(segments, lang)` | `(object, str) -> str` | Project raw FunASR output into monologues + timed words |
| `tencent_result_detail_to_asr(result_detail, lang)` | `(object, str) -> str` | Project Tencent result-detail objects into monologues + timed words |
| `aliyun_sentences_to_asr(sentences, lang)` | `(object, str) -> str` | Project Aliyun sentence results into monologues + timed words |
| `strip_timing(text)` | `str -> str` | Remove timing (standalone) |

Plus standalone versions of all `add_*` methods that take `chat_text: str`
and return `str` (parse → mutate → serialize internally).

## Rev.AI Native HTTP Client

The shared Rust crate `crates/batchalign-revai/` provides the blocking Rev.AI
HTTP client used by the Rust server control plane and, when needed, by the
PyO3 direct-Python bridge. In server mode, Rev.AI-backed `transcribe`,
`transcribe_s`, `benchmark`, and Rev-backed UTR now use the Rust client
directly instead of routing those code paths through Python workers. The PyO3
layer still exposes the client through these `#[pyfunction]` wrappers for
direct Python workflows:

| Function | Description |
|----------|-------------|
| `rev_transcribe(audio_path, api_key, language, ...)` | Upload, poll, download transcript (blocking) |
| `rev_get_timed_words(audio_path, api_key, language)` | Upload, poll, download timed words (blocking) |
| `rev_submit(audio_path, api_key, language, ...)` | Submit job only (for pre-submission) |
| `rev_poll(job_id, api_key)` | Poll for transcript result |
| `rev_poll_timed_words(job_id, api_key)` | Poll for timed words result |

All Rev.AI functions release the GIL via `py.detach()` for the entire HTTP
lifecycle. In direct Python workflows, that still permits Python-level
parallelism. In server mode, file-level concurrency is owned by Rust and does
not require Python worker fan-out.

### Rev.AI Pre-Submission

For Rev.AI-backed commands, the Rust server can pre-submit audio files before
the normal per-file dispatch loop via `rev_submit`. Later per-file work now
also stays Rust-owned: the server polls those job IDs and projects the
transcript either into the shared ASR response domain (`transcribe`,
`benchmark`) or into UTR timed tokens (`align`) without widening the Python
worker contract.

## ASR Worker Contract

ASR is now the clearest example of the intended Python/Rust split.

- Python-owned:
  - load Whisper or HK provider SDKs
  - call the provider or local model
  - return a tagged raw payload close to the provider boundary
- Rust-owned:
  - normalize that payload into shared timing/token records
  - apply ASR post-processing
  - build CHAT
  - run optional utseg and morphosyntax
  - cache or reuse the normalized result

The worker-side ASR result is now one of two tagged raw shapes:

- `{"kind": "monologues", ...}` for Rev-style and HK provider monologues
- `{"kind": "whisper_chunks", ...}` for HuggingFace Whisper chunk output

That keeps provider-local adaptation inside Python while making the shared
normalization rule a Rust concern. Incremental processing still stays entirely
on the Rust side; it does not require a wider Python boundary.

For ASR, the migration has now gone further than that. Local Whisper runs
through live `execute_v2` prepared-audio requests, and the HK engines run
through live `execute_v2` provider-media requests. Rust prepares full-file PCM
audio for local models up front, and Python feeds that prepared waveform
directly into the HuggingFace runtime. HK providers still receive a media path
because their SDKs are Python-only, but they no longer depend on the legacy
`batch_infer(task="asr")` request bag either.

For HK engines, Python now keeps only the unavoidable SDK/model boundary. The
shared projection from raw FunASR, Tencent, and Aliyun output into typed
speaker-monologue payloads lives in dedicated Rust helpers exposed through
`batchalign_core`, and Rust owns the final normalization into the shared
internal `AsrResponse` domain. Rust also owns legacy credential discovery for
Tencent and Aliyun when it launches workers, injecting those provider settings
into the worker environment so the Python adapters no longer reopen the legacy
config file during ASR bootstrap.

Speaker diarization is now on the same typed worker boundary. The live stdio
worker exposes `execute_v2(task="speaker")`, Rust builds typed speaker
requests, and Python returns only raw diarization segments. Rust now prepares
speaker audio as the same typed PCM artifact family used by FA and local
Whisper. When `transcribe_s` needs diarization and the ASR backend did not
already supply usable speaker labels, the Rust server now composes that speaker
request itself and applies the returned segments through
`batchalign-chat-ops::speaker::reassign_speakers`. This stays a low-level
worker capability that preserves the batchalign2 diarization feature surface
(`transcribe_s`) without inventing a new standalone CLI `speaker` command. The
PyO3 `ParsedChat.reassign_speakers()` entry point remains for direct Python
workflows, but the production server path no longer depends on a PyO3-only
speaker rewrite.

The same cutover now applies to `opensmile` and `avqi`. Rust builds typed V2
requests, owns the full-file audio preparation step, and Python receives only
prepared PCM attachments plus task-local parameters before returning raw
feature/metric payloads.

Forced alignment is now on the same staged path. The V2 redesign already has a
Rust-side FA request builder that takes the existing `FaInferItem`, writes its
word arrays into a prepared JSON artifact, extracts the audio window into a
prepared PCM artifact, and emits a typed `ForcedAlignmentRequestV2`. That path
is no longer merely staged: the live stdio worker now carries an `execute_v2`
op, and both full-file FA and incremental FA dispatch through that typed V2
boundary in production. The Python side therefore no longer reconstructs
legacy FA payloads or reopens source media paths for live FA execution; it
reads Rust-owned prepared artifacts and returns raw Whisper token timings or
indexed word timings through typed V2 result wrappers. `batchalign-app` then
maps those V2 FA responses back into the established `batchalign-chat-ops`
alignment domain, so the worker boundary does not become the place where
timing interpretation lives.

## GIL Strategy

All pure-Rust methods use `py.detach()` (PyO3 0.28) to release the GIL during
computation. The callback methods hold the GIL only during Python callback
invocation — Rust payload construction and response injection run GIL-free.

## Module Layout (`pyo3/src/`)

```
lib.rs                    — module registration, serde structs
parsed_chat/              — #[pymethods] on ParsedChat, split by domain:
  mod.rs                  — constructors, serialization, validation, metadata
  morphosyntax.rs         — %mor/%gra methods
  fa.rs                   — forced alignment methods
  text.rs                 — translation and utseg methods
  speakers.rs             — speaker reassignment shim and utterance timing
  cleanup.rs              — disfluency and retrace markers
pyfunctions.rs            — standalone #[pyfunction]s
morphosyntax_ops.rs       — morphosyntax inner functions
fa_ops.rs                 — FA orchestration inner functions
text_ops.rs               — translation/utseg inner functions
  speaker_ops.rs            — speaker shim delegating to chat-ops + utterance timing
cleanup_ops.rs            — disfluency inner functions
tier_ops.rs               — dependent tier management
extract.rs                — NLP word extraction from AST
inject.rs                 — morphosyntax injection into AST
forced_alignment.rs       — FA grouping, timing injection, %wor generation
dp_align.rs               — Hirschberg DP alignment
utterance_segmentation.rs — utterance splitting
retokenize.rs             — Stanza retokenization mapping
build.rs                  — CHAT building from JSON
parse.rs                  — parse helpers
metadata.rs               — metadata extraction
py_json_bridge.rs         — JSON serialization bridge
provider_pipeline.rs      — Rust-owned Python pipeline executor
revai/                    — Rev.AI HTTP client
cli_entry.rs              — console_scripts entry point
```

## Incremental Processing Note

Shared speaker reassignment now lives in
`crates/batchalign-chat-ops/src/speaker.rs`. The PyO3 speaker modules keep only
the direct-Python surface and the utterance-timing helper.

Incremental morphosyntax and FA processing are server/runtime concerns, not
PyO3 API concerns. The cache-aware diffing logic lives in the Rust app layer,
so it does not require a wider Python surface.
