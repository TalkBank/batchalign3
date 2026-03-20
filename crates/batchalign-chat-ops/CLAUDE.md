# batchalign-chat-ops — CHAT Manipulation for Server-Side Orchestration

## Overview

Shared library implementing the extract→modify→inject round-trip for all NLP tasks.
Used by both the PyO3 bridge (`batchalign-core`) and the standalone server
(`batchalign-app`). This crate owns **all** pre-processing, post-processing, and domain
logic — Python workers are pure model servers that return raw ML output. All text normalization,
DP alignment, WER computation, ASR post-processing, compound merging, number expansion,
retokenization, and result injection live here.

## Module Map

| Module | Purpose |
|--------|---------|
| `parse.rs` | `parse_lenient()`, `parse_strict()`, `is_dummy()` |
| `serialize.rs` | `to_chat_string()` |
| `extract.rs` | `extract_words()` — domain-aware word extraction (Mor/Wor/Pho/Sin) |
| `inject.rs` | `inject_morphosyntax()`, `replace_or_add_tier()` |
| `retokenize/` | Stanza-induced word splits/merges: deterministic mapping with length-aware fallback |
| `morphosyntax.rs` | Payload collection, cache key, clear/inject for %mor/%gra |
| `utseg.rs` | Utterance segmentation payloads, cache key, apply |
| `translate.rs` | Translation payloads, cache key, inject %xtra |
| `coref.rs` | Coreference payloads (document-level), inject %xcoref (sparse) |
| `fa/` | Forced alignment: grouping, extraction, injection, postprocess, DP alignment, UTR timing recovery |
| `dp_align.rs` | Hirschberg divide-and-conquer sequence alignment (linear space) |
| `text_types.rs` | Provenance newtypes: `ChatRawText`, `ChatCleanedText`, `SpeakerCode` |
| `nlp/` | UD types (`UdWord`, `UniversalPos`), UD→CHAT mapping, validation, language-specific rules |
| `asr_postprocess/` | ASR post-processing: compound merging, number expansion, Cantonese normalization, retokenization |
| `wer_conform.rs` | WER word normalization: compound splitting, name replacement, filler/contraction expansion |

## Key Commands

```bash
cargo nextest run -p batchalign-chat-ops
cargo clippy -p batchalign-chat-ops -- -D warnings
```

## NLP Task Modules

Each task module exports: **batch item type**, **response type**, **collect_payloads()**,
**cache_key()**, and **apply/inject results**.

| Task | Granularity | Cache Key |
|------|-------------|-----------|
| Morphosyntax | Per-utterance | `SHA256("{words}\|{lang}\|mwt")` |
| Utseg | Per-utterance | `SHA256("{words}\|{lang}")` |
| Translate | Per-utterance | `SHA256("{text}\|{src_lang}\|{tgt_lang}")` |
| Coref | Per-document | No caching (full-document context) |
| FA | Per-group (time-windowed) | `SHA256("{audio_identity}\|{start}\|{end}\|{text}\|{pauses}\|{engine}")` |

### UTR (Utterance Timing Recovery)

`fa/utr.rs` — Pre-pass that injects utterance-level timing from ASR tokens
into untimed CHAT utterances. Uses a single global Hirschberg DP alignment
(`dp_align::align(..., CaseInsensitive)`) of ALL document words against ALL
ASR tokens. Timed utterances participate to anchor the alignment but their
bullets are left unchanged. The global approach avoids token starvation that
per-utterance windowed alignment suffered from, but it is still a monotonic
aligner, so dense overlap / text-audio reordering remains a known limitation.

Key types: `AsrTimingToken` (text + start_ms + end_ms), `UtrResult`
(injected/skipped/unmatched counts).

Entry point: `inject_utr_timing(&mut ChatFile, &[AsrTimingToken]) -> UtrResult`.

Detection helper: `count_utterance_timing(&ChatFile) -> (timed, untimed)` in
`fa/grouping.rs`.

Cache key helpers: `utr_asr_cache_key()` (full-file), `utr_asr_segment_cache_key()`
(partial-window). Both produce BLAKE3-keyed `CacheKey` values.

Window finder: `find_untimed_windows(&ChatFile, total_audio_ms, padding_ms) -> Vec<(u64, u64)>`
identifies audio windows covering contiguous untimed utterances for partial-window ASR.

## Dependencies

Path deps to `talkbank-tools` crates (talkbank-model, talkbank-direct-parser,
talkbank-parser).

Re-exports `ChatFile` and `LanguageCode` for downstream convenience.

## Design Principles

- **No string hacking** — all CHAT operations through AST manipulation
- **Provenance types** — `ChatRawText` vs `ChatCleanedText` (CHAT direction) and `AsrRawText` → `AsrNormalizedText` → `ChatWordText` (ASR direction) prevent mixing text at different pipeline stages
- **Domain-aware extraction** — `TierDomain` selects which word properties to extract
- **Alignment validation** — %mor word count must match main tier word count before injection
- **Content walker** — `walk_words()` / `walk_words_mut()` from `talkbank-model` centralizes
  UtteranceContent/BracketedItem traversal. Callers provide only leaf-handling closures.
  Used by `extract.rs`, `fa/extraction.rs`, `fa/injection.rs`, `fa/postprocess.rs`.

## ASR Post-Processing (`asr_postprocess/`)

Ported from Python `batchalign/pipelines/asr/utils.py`. Transforms raw ASR tokens
into utterances ready for CHAT assembly. Sub-modules:

| File | Purpose |
|------|---------|
| `asr_types.rs` | Provenance newtypes: `AsrRawText`, `AsrNormalizedText`, `ChatWordText`, `AsrTimestampSecs`, `SpeakerIndex` |
| `mod.rs` | Pipeline (`process_raw_asr`), types (`AsrWord`, `Utterance`, `AsrOutput`, `AsrElementKind`), retokenization |
| `compounds.rs` | 3,584 compound word pairs, `merge_compounds()` with O(1) HashSet lookup |
| `num2text.rs` | Number-to-words expansion via NUM2LANG tables (12 languages) |
| `num2chinese.rs` | Chinese/Japanese number converter (simplified + traditional, up to 10^48) |
| `cantonese.rs` | Cantonese text normalization: `zhconv` zh-HK + 31-entry Aho-Corasick replacement table |

### Pipeline Stages

```
1. Compound merging
2. Timed word extraction (seconds → ms)
3. Multi-word splitting (space-separated tokens with timestamp interpolation)
4. Number expansion (digits → word form)
4b. Cantonese normalization (lang=yue only: simplified→HK traditional + domain replacements)
5. Long turn splitting (>300 words)
6. Retokenization (split into utterances by punctuation)
```

### Cantonese Normalization

Migrated from Python `batchalign/inference/hk/_common.py` to Rust. Uses:
- **`zhconv`** crate (pure Rust, 100-200 MB/s) — Aho-Corasick automata compiled from OpenCC + MediaWiki rulesets for `Variant::ZhHK` conversion
- **Domain replacement table** — 31 entries (13 multi-char + 18 single-char) for Cantonese-specific character corrections, applied via a second Aho-Corasick pass with leftmost-longest matching

Exposed to Python via `batchalign_core.normalize_cantonese()` and `batchalign_core.cantonese_char_tokens()`. Python `_common.py` delegates to these Rust functions — **no OpenCC Python dependency needed**.

Data files in `data/`: `compounds.json` (3,660 pairs, 76 duplicates), `num2lang.json` (12 languages),
`names.json` (~6,700 proper names), `abbrev.json` (~400 abbreviations).

---
Last Updated: 2026-03-18
