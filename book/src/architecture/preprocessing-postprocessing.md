# Preprocessing and Postprocessing for Model Inference

**Status:** Current
**Last updated:** 2026-03-18

All domain logic — text normalization, alignment, result injection, and error recovery — lives in Rust. Python workers are stateless ML inference endpoints. This chapter documents the preprocessing that prepares data for inference and the postprocessing that incorporates results back into the CHAT AST.

## The Boundary Principle

Python receives structured payloads (lists of words, audio paths, language codes) and returns structured results (POS tags, timestamps, parse trees). It never sees CHAT text, never parses tiers, and never makes alignment decisions.

```
Rust: CHAT AST → extract words → clean text → build payload
                                                    │
Python: load model → run inference → return structured output
                                                    │
Rust: validate response → align with AST → inject results → serialize CHAT
```

## Preprocessing by Task

### Morphosyntax

**Extract** (`morphosyntax.rs: collect_payloads`):
1. Walk content with `walk_words(domain=Mor)`
2. Collect `cleaned_text()` for each alignable word
3. Replace special forms (`@c` → `"xbxxx"`, `@s` → language marker) — Stanza can't handle CHAT-specific markers
4. Build payload: `Vec<String>` of words per utterance

**Payload → Python:**
```json
{"words": ["I", "want", "cookie"], "lang": "eng"}
```

**Python returns:** Raw Stanza `to_dict()` output — POS tags, lemmas, dependency parse, features.

**Postprocess** (`morphosyntax.rs: inject`, `retokenize/`):
1. **Retokenization:** Stanza may split (`don't` → `do`, `n't`) or merge words differently than CHAT. The retokenizer uses character-level DP alignment to deterministically map Stanza tokens back to CHAT words.
2. **UD → CHAT mapping:** Convert Universal Dependencies POS/features to TalkBank %mor format (category mappings, stem extraction, feature translation).
3. **MWT handling:** Multi-word tokens (clitic languages) generate multiple %mor items for one surface word.
4. **%gra construction:** Build dependency graph with chunk-based indexing (GRA indices are %mor word positions, not surface word positions).
5. **Validation:** Check word count alignment, GRA cycle detection, chunk count consistency.
6. **Injection:** Replace or add %mor and %gra dependent tiers on the utterance.

### Forced Alignment

FA preprocessing has two stages — UTR (Utterance Timing Recovery) and FA proper.
See [Forced Alignment](../reference/forced-alignment.md) for the complete pipeline.

**UTR:** Injects utterance-level timing from ASR tokens before FA runs. Supports
global single-pass and two-pass overlap-aware strategies. See
[Overlapping Speech](../reference/overlap-markers.md) for the two-pass algorithm
and CA marker-aware windowing.

**FA:** Groups utterances into time-windowed clusters, sends each group's words +
audio window to Python for word-level timestamp inference, then injects timing
back into the AST.

**Python receives:** Audio window (start_ms, end_ms) + word list.
**Python returns:** Per-word timestamps.
**Rust postprocessing:** Word end-time chaining, monotonicity enforcement, pause
assignment, %wor tier generation.

### ASR (Automatic Speech Recognition)

ASR preprocessing is the most complex because raw ASR output needs extensive normalization before it becomes CHAT:

**ASR postprocessing pipeline** (`asr_postprocess/`):

| Stage | Module | What it does |
|-------|--------|-------------|
| 1. Compound merging | `compounds.rs` | Join split compounds: `ice` + `cream` → `ice+cream` (3,584 pairs, O(1) HashSet) |
| 2. Timed word extraction | `mod.rs` | Convert seconds → milliseconds, extract ASR tokens |
| 3. Multi-word splitting | `mod.rs` | Split space-separated tokens with timestamp interpolation |
| 4. Number expansion | `num2text.rs` | `42` → `forty two` (12 languages via NUM2LANG tables) |
| 4b. Cantonese normalization | `cantonese.rs` | Simplified → HK traditional + domain replacements (31 entries, Aho-Corasick) |
| 5. Long turn splitting | `mod.rs` | Break turns > 300 words into separate utterances |
| 6. Retokenization | `mod.rs` | Split into utterances by punctuation boundaries |

**Retokenization** (step 6) is particularly important: ASR produces one long stream of text, but CHAT needs it segmented into utterances. The retokenizer uses punctuation (`.`, `?`, `!`) as utterance boundaries and assigns timing from the ASR tokens.

### Translation

**Extract:** Full utterance text (all words concatenated).
**Python:** Google Translate or SeamlessM4T → translated text.
**Inject:** Add `%xtra` dependent tier with the translated text.

### Utterance Segmentation

**Extract:** Words per utterance (same as morphosyntax).
**Python:** Stanza constituency parser → parse tree with boundary predictions.
**Postprocess:** Assign boundary codes (utterance break, clause break, continuation) based on constituency structure. Apply boundaries to merge/split utterances.

### Coreference

**Extract:** All sentences in the document (document-level, not per-utterance).
**Python:** Stanza coref → coreference chains.
**Inject:** Sparse `%xcoref` tiers on utterances that contain coreferent mentions.

## Retokenization: The Character-Level Bridge

When Stanza tokenizes differently than CHAT, the retokenizer (`retokenize/`) bridges the gap:

```
CHAT words:     ["don't",    "wanna"]
Stanza tokens:  ["do", "n't", "wan", "na"]
```

The retokenizer:
1. Concatenates both word lists into character strings
2. Runs character-level DP alignment
3. Builds a deterministic mapping from Stanza token indices back to CHAT word indices
4. Uses this mapping to assign Stanza annotations (POS, lemma, depparse) to the correct CHAT words

This handles splits (`don't` → `do` + `n't`), merges, and even reorderings across languages. The mapping uses a length-aware fallback for ambiguous cases.

## Cache Keys

Each task computes a cache key from its input payload, so identical inputs skip inference:

| Task | Cache key formula |
|------|------------------|
| Morphosyntax | `SHA256("{words}|{lang}|mwt")` |
| Utseg | `SHA256("{words}|{lang}")` |
| Translation | `SHA256("{text}|{src_lang}|{tgt_lang}")` |
| FA | `SHA256("{audio_identity}|{start}|{end}|{text}|{pauses}|{engine}")` |
| UTR ASR | `BLAKE3("utr_asr|{audio_identity}|{lang}")` |
| Coref | No caching (document-level context) |

Cache is tiered: moka in-memory (hot) → SQLite on-disk (cold).
