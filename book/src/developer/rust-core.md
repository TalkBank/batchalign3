# Rust Core (batchalign_core)

**Status:** Current
**Last updated:** 2026-03-15

For new contributors, start with:

- [Rust Contributor Onboarding](rust-contributor-onboarding.md)
- [Rust Workspace Map](rust-workspace-map.md)

## Repository Structure

The PyO3 bridge lives in `pyo3/` within the batchalign3 repository. This is a
single-crate project (`batchalign-pyo3`) that builds the `batchalign_core`
Python extension module. It depends on `batchalign-chat-ops` (same workspace)
and on `talkbank-*` crates via path dependencies pointing to the sibling
`talkbank-tools` repo — they are **not** vendored copies.

| Crate | Location | Purpose |
|-------|----------|---------|
| `batchalign-pyo3` | `pyo3/src/` | PyO3 bridge — what Python imports as `batchalign_core` |
| `batchalign-chat-ops` | `crates/batchalign-chat-ops/` | CHAT lifecycle: parsing, injection, extraction, alignment, post-processing |
| `talkbank-model` | `../talkbank-tools/crates/talkbank-model/` | Shared data model (path dep) |
| `talkbank-parser` | `../talkbank-tools/crates/talkbank-parser/` | Tree-sitter parser (path dep) |
| `talkbank-direct-parser` | `../talkbank-tools/crates/talkbank-direct-parser/` | Chumsky parser (path dep) |

## Module Organization

The PyO3 crate is organized by domain. `pyo3/src/lib.rs` declares:

**Core CHAT operations (delegated to `batchalign-chat-ops`):**

| Module | Purpose |
|--------|---------|
| `dp_align` | Hirschberg DP alignment (WER, retokenization) |
| `extract` | NLP word extraction from AST (domains: mor, wor, pho, sin) |
| `forced_alignment` | FA grouping, timing injection, %wor tier generation |
| `inject` | Morphosyntax/retokenize injection from callback response |
| `retokenize` | Maps Stanza re-tokenized output back to CHAT words |
| `utterance_segmentation` | Utterance splitting based on segmentation callback |
| `nlp` | NLP mapping/validation (re-exports from `batchalign-chat-ops`) |

**`ParsedChat` method implementations (split by domain):**

| Module | Methods |
|--------|---------|
| `parsed_chat/mod.rs` | Constructors (`parse`, `parse_lenient`, `build`), serialization, validation, metadata, simple mutations |
| `parsed_chat/morphosyntax.rs` | `add_morphosyntax`, `add_morphosyntax_batched`, `extract_morphosyntax_payloads`, `inject_morphosyntax_from_cache`, `extract_morphosyntax_strings` |
| `parsed_chat/fa.rs` | `add_forced_alignment` |
| `parsed_chat/text.rs` | `add_translation`, `add_utterance_segmentation`, `add_utterance_segmentation_batched` |
| `parsed_chat/speakers.rs` | `reassign_speakers` shim, `add_utterance_timing` |
| `parsed_chat/cleanup.rs` | `add_disfluency_markers`, `add_retrace_markers` |

**Inner function modules (business logic called by `ParsedChat` methods):**

| Module | Purpose |
|--------|---------|
| `morphosyntax_ops` | Per-utterance and batched morphosyntax orchestration |
| `fa_ops` | Forced alignment orchestration |
| `text_ops` | Translation and utterance segmentation inner functions |
| `speaker_ops` | Speaker shim (delegates to `batchalign-chat-ops::speaker`) and utterance timing |
| `cleanup_ops` | Disfluency and retrace markers |
| `tier_ops` | Dependent tier management |

**Other modules:**

| Module | Purpose |
|--------|---------|
| `build` | Build CHAT files from JSON transcript descriptions |
| `parse` | Pure-Rust parse helpers |
| `metadata` | Metadata extraction from CHAT headers |
| `pyfunctions` | Standalone `#[pyfunction]`s (see below) |
| `cli_entry` | Console_scripts entry point for `batchalign3` command |
| `py_json_bridge` | JSON serialization/deserialization for Python ↔ Rust |
| `revai` | Rev.AI native HTTP client |

Shared speaker reassignment now lives in
`crates/batchalign-chat-ops/src/speaker.rs`, so the PyO3 speaker modules keep
only the Python-facing shim plus the utterance-timing helper.

## Key PyO3 Entry Points

### `ParsedChat` methods

**Parsing and serialization:**
- `ParsedChat.parse(text)` — strict parse; rejects files with errors
- `ParsedChat.parse_lenient(text)` — error-recovery parse; used by alignment
- `ParsedChat.build(...)` — construct CHAT from structured data
- `handle.serialize()` — serialize the AST back to valid CHAT text

**Validation:**
- `handle.validate()` — list of validation error strings
- `handle.validate_structured()` — structured validation results
- `handle.validate_chat_structured()` — CHAT-specific structured validation
- `handle.parse_warnings()` — warnings from the parse phase

**Morphosyntax:**
- `add_morphosyntax(callback)` — inject %mor/%gra (per-utterance callback)
- `add_morphosyntax_batched(callback)` — batched variant (one callback call)
- `extract_morphosyntax_payloads()` — get utterance texts and cache keys
- `inject_morphosyntax_from_cache(...)` — inject cached %mor/%gra strings
- `extract_morphosyntax_strings()` — extract final %mor/%gra for cache storage
- `clear_morphosyntax()` — remove existing %mor/%gra tiers

**Forced alignment and timing:**
- `add_forced_alignment(callback)` — inject word-level timing
- `add_utterance_timing(asr_words)` — inject timed ASR words

**Text analysis:**
- `add_translation(callback)` — inject %xtra translation tiers
- `add_utterance_segmentation(callback)` — split/merge utterance boundaries
- `add_utterance_segmentation_batched(callback)` — batched variant

**Speaker and cleanup:**
- `reassign_speakers(...)` — compatibility shim over `batchalign-chat-ops::speaker`
- `add_disfluency_markers(...)` — inject disfluency markers
- `add_retrace_markers(...)` — inject retrace markers

**Metadata and utilities:**
- `extract_nlp_words(...)` — extract words from the AST for NLP processing
- `extract_metadata()` — extract header metadata
- `extract_languages()` — extract declared languages
- `is_ca()` — check if file uses conversation-analysis conventions
- `is_no_align()` — check if file has `@Options: noalign`
- `strip_timing()` — remove all timing bullets
- `add_dependent_tiers(...)` — add/replace dependent tiers
- `add_comment(...)` — add a comment header

### Standalone `#[pyfunction]`s

These are module-level functions, not methods on `ParsedChat`:

| Function | Purpose |
|----------|---------|
| `parse_and_serialize(text)` | Round-trip parse and serialize |
| `extract_nlp_words(...)` | Extract words (also available as method) |
| `build_chat(...)` | Build CHAT and return as string (vs. `ParsedChat.build` which returns a handle) |
| `add_dependent_tiers(...)` | Add tiers to CHAT text directly |
| `extract_timed_tiers(...)` | Extract timed tier data |
| `py_dp_align(...)` | DP alignment exposed to Python |
| `chat_terminators()` | List valid CHAT terminators |
| `chat_mor_punct()` | List MOR punctuation items |
| `align_tokens(...)` | Token alignment |
| `wer_conform(...)` | Word normalization for WER evaluation |
| `wer_compute(...)` | Full WER computation |
| `normalize_cantonese(...)` | Cantonese text normalization (simplified → HK traditional) |
| `cantonese_char_tokens(...)` | Cantonese character tokenization |

### Callback Pattern

Most mutation methods accept a Python callback. The Rust side collects data
from the AST, calls the Python callback with that data (typically as JSON),
and injects the results back into the AST. For batched variants, Rust
collects all utterance payloads in a single pass, calls the callback once
with a JSON array, then injects all results.

## Tree-Sitter Grammar

The CHAT grammar definition lives in the sibling `talkbank-tools` repo at
`talkbank-tools/grammar/`. After editing it, regenerate the C parser:

```bash
cd ../talkbank-tools/grammar && tree-sitter generate
```

This regenerates `parser.c`, which the tree-sitter parser depends on.
**Forgetting this step causes the parser to use a stale grammar.**

After grammar changes, always test against real corpus data in addition to
the curated test suite.

## Building for Development

When you change the PyO3 bridge or the shared `batchalign-chat-ops` logic that
Python consumes, rebuild the extension with the repo-native command:

```bash
make build-python
```

This runs `uv run maturin develop -m pyo3/Cargo.toml` under the hood.

If you also plan to validate the standalone Rust CLI directly after a shared
crate change, rebuild that binary too:

```bash
cargo build -p batchalign-cli
```

## Running Rust Tests

```bash
cargo nextest run --manifest-path pyo3/Cargo.toml
```

To run the full parser integration test suite (both parsers), use the
`talkbank-tools` workspace:

```bash
cargo nextest run --manifest-path ../talkbank-tools/Cargo.toml -p talkbank-parser-tests
```

## GIL Release Strategy

All pure-Rust `batchalign_core` methods release the Python GIL via
`py.detach()` (pyo3 0.28). This allows Python threads to run concurrently
while Rust does heavy computation.

Callback-based methods hold the GIL only during the Python callback
invocation. The pattern is:

1. Release GIL, walk the Rust AST, collect data.
2. Acquire GIL, call the Python callback.
3. Release GIL, inject results back into the AST.

CPU-bound Rust work does not block other Python threads, while the callback
(which runs Python model inference) holds the GIL as expected.

## Workflow: Adding a New CHAT Transformation

1. Add the Rust function in `batchalign-pyo3` — as `#[pymethods]` on
   `ParsedChat` (in the appropriate domain file under `parsed_chat/`) or as
   a standalone `#[pyfunction]` in `pyfunctions.rs`.
2. Add the inner logic in the corresponding `*_ops.rs` module.
3. Write Rust tests.
4. Regenerate the grammar if you changed `grammar.js`.
5. Rebuild `batchalign_core` (`make build-python`, which runs
   `uv run maturin develop -m pyo3/Cargo.toml`).
6. Call the new function from Python — either through the worker dispatch
   (`batchalign/worker/_infer.py`) or through `pipeline_api.py` operations.
7. Test against real corpus data, not just unit tests.
