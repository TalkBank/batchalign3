# Overlap Encoding: `&*` and `+<` Internals

**Status:** Current
**Last updated:** 2026-03-17 (grouping-aware fallback added)

## AST Representation

### `&*` — `OtherSpokenEvent`

**Model (talkbank-tools):** `crates/talkbank-model/src/model/content/other_spoken.rs`

```rust
pub struct OtherSpokenEvent {
    pub speaker: SpeakerCode,     // e.g., "INV"
    pub text: smol_str::SmolStr,  // e.g., "oh_okay_yeah"
    pub span: Span,               // source location (skipped in serde)
}
```

Appears in two enum locations:
- `UtteranceContent::OtherSpokenEvent(OtherSpokenEvent)` — top-level content
- `BracketedItem::OtherSpokenEvent(OtherSpokenEvent)` — inside groups

**Parser (talkbank-tools):**
`crates/talkbank-direct-parser/src/main_tier/words.rs`

The parser accepts `&*` + speaker chars + `:` + non-whitespace chars. It is
registered before `&=` events to ensure correct precedence.

**Serialization:** `&*SPK:text` — roundtrips cleanly via `WriteChat`.

### `+<` — `Linker::LazyOverlapPrecedes`

**Model (talkbank-tools):** `crates/talkbank-model/src/model/content/linker.rs`

```rust
pub enum Linker {
    LazyOverlapPrecedes,  // +<
    OtherCompletion,      // ++
    QuickUptakeOverlap,   // +^
    // ...
}
```

Stored on `TierContent.linkers: TierLinkers` (a `Vec<Linker>` newtype).
Linkers appear at the start of an utterance's content, before words.

## Content Walker Behavior

The content walker (`for_each_leaf` / `for_each_leaf_mut`) **skips**
`OtherSpokenEvent` entirely. It is listed in the no-op match arm alongside
events, pauses, overlap points, and other non-alignable content:

```rust
UtteranceContent::OtherSpokenEvent(_) => {}  // skipped
```

This means `&*` content:
- Is **not counted** in word alignment (Wor, Mor, Pho, Sin domains)
- Does **not appear** in `%wor` tier generation
- Is **not extracted** by `collect_fa_words()` for forced alignment
- Is **not included** in the UTR reference word sequence

## Two-Pass UTR Strategy

**Code (batchalign3):** `crates/batchalign-chat-ops/src/fa/utr.rs` and
`crates/batchalign-chat-ops/src/fa/utr/two_pass.rs`

### Trait architecture

```rust
pub trait UtrStrategy: Send + Sync {
    fn inject(&self, chat_file: &mut ChatFile, asr_tokens: &[AsrTimingToken]) -> UtrResult;
}

pub struct GlobalUtr;           // Original single-pass algorithm
pub struct TwoPassOverlapUtr {  // +<-aware two-pass with grouping-aware fallback
    pub grouping_context: Option<GroupingContext>,
}

pub struct GroupingContext {
    pub total_audio_ms: u64,
    pub max_group_ms: u64,
}
```

`select_strategy(chat_file, grouping_context)` inspects the file for
`Linker::LazyOverlapPrecedes` and returns the appropriate strategy. The
`--utr-strategy` CLI flag overrides this with `Global`, `TwoPass`, or
`Auto` (default). When called from the dispatch layer, `GroupingContext`
is populated from `FaParams` and `UtrPassContext`.

### Grouping-aware best-of-both fallback

`TwoPassOverlapUtr` runs both approaches and uses FA group counts as
the primary comparison signal:

1. **Two-pass candidate:** Run pass 1 (global DP excluding `+<`) then pass 2
   (adaptive windowed recovery for each `+<` utterance).
2. **Global candidate:** Run the standard global DP including all utterances.
3. **Compare FA grouping** (when `GroupingContext` is available):
   - Call `group_utterances()` on both outputs (cheap — microseconds).
   - If two-pass creates **fewer groups**, fall back to global. Fewer groups
     means wider FA windows, which causes worse word-level alignment.
   - If groups are equal, use timed utterance count as tiebreaker.
   - When equal, prefer two-pass (better backchannel placement).
4. **Without grouping context:** Fall back to timed utterance count only.

This makes two-pass **never worse** than global for the observed failure
mode (FA group merging). On English where pass 2 succeeds, backchannels
get better timing placement. On German where two-pass creates fewer FA
groups (152 vs 162), the fallback detects this and uses global results.

**Motivation:** Experiments showed two-pass lost 27 percentage points of
overall coverage on German files. Stage decomposition revealed the cause:
two-pass changes UTR bullet distribution, which changes
`estimate_untimed_boundaries` anchor points, which changes FA group
boundaries. On German, two-pass created 10 fewer FA groups — wider
windows led to worse word-level alignment. The grouping-level comparison
catches this specific failure mode without re-running FA.

**Known limitation:** The Welsh regression (1502→1308) has a different
root cause — two-pass creates *more* groups (313 vs 284) but with
different boundaries that still lead to worse FA. The group-count
heuristic cannot detect this case. A full-pipeline comparison (run FA
with both strategies, keep the better output) would be the complete fix
but is expensive (2x alignment per file).

### Pass 1: Global alignment excluding `+<`

`run_global_utr(chat_file, asr_tokens, skip_lazy_overlap=true)` — same as the
original algorithm but the flatten loop skips utterances where
`has_lazy_overlap == true`. The DP reference contains only main-speaker words
in their correct temporal order. `+<` utterances are counted as unmatched.

### Pass 2: Adaptive windowed recovery for `+<` utterances

For each `+<` utterance at index `i`:
1. Find the nearest preceding utterance with a bullet (set in pass 1).
2. Try increasingly wide windows around the predecessor:
   - Narrow: ±500ms (sufficient for English)
   - Medium: ±predecessor duration (min 2s)
   - Wide: ±2x predecessor duration (min 5s)
3. At each width, filter ASR tokens to the window, run a small Hirschberg DP.
4. Accept the first match (prefers tight placement).

**When no `+<` utterances exist:** The strategy detects this via
`select_strategy()` and uses `GlobalUtr` directly (no cloning overhead).

## `&*` → `+<` Conversion

A private workspace experiment tool outside this repo
(`../analysis/per-speaker-utr-experiment-2026-03-16/` from the workspace root)
includes a `convert` subcommand that transforms `&*` to separate `+<`
utterances using the typed AST:

1. Walk each utterance's content (including inside groups).
2. Extract `OtherSpokenEvent` nodes, recording speaker + text.
3. Remove them from the host utterance.
4. For each extracted event, create a new `Utterance` with `+<` linker and
   words split from the underscore-joined text.
5. Insert after the host utterance.

### Edge cases handled

- Multiple `&*` in one utterance (each becomes its own `+<` utterance)
- Multi-word `&*` with underscores (`oh_okay_yeah` → `oh okay yeah`)
- `&*` inside groups (`<... &*INV:mhm ...> [//]`)
- Reverse direction (`&*PAR:yeah` on INV's line)
- Host utterances with and without timing bullets
- Host dependent tiers preserved (they were already `&*`-invisible)

## Corpus Statistics

### `&*` (OtherSpokenEvent)

| Corpus | Files | Total markers | Single-word % |
|--------|-------|---------------|---------------|
| ca-data | 256 | 12,016 | 96% |
| aphasia-data | 644 | 10,161 | 88% |
| rhd-data | 190 | 5,160 | 83% |
| psychosis-data | 236 | 2,799 | 98% |
| tbi-data | 135 | 2,105 | 90% |
| dementia-data | 390 | 1,680 | 89% |
| slabank-data | 191 | 774 | — |
| childes-data | 146 | 411 | — |
| **Total** | | **~35,000** | **91%** |

Top words: mhm (~12,500), yeah (~5,500), okay (~3,300), mm (~1,400).

### `+<` (LazyOverlapPrecedes)

| Corpus | Files | `+<` utterances |
|--------|-------|----------------|
| childes-data | 10,596 | 194,720 |
| phon-data | 614 | 50,892 |
| biling-data | 248 | 37,727 |
| aphasia-data | 1,241 | 15,720 |
| tbi-data | 251 | 7,469 |
| ca-data | 242 | 6,606 |
| dementia-data | 1,536 | 4,745 |
| **Total** | | **~327,000** |

Of these, ~131,000 (40%) already have timing bullets.

## File Locations

| File | Purpose |
|------|---------|
| `crates/batchalign-chat-ops/src/fa/utr.rs` | UtrStrategy trait, GlobalUtr, select_strategy, run_global_utr |
| `crates/batchalign-chat-ops/src/fa/utr/two_pass.rs` | TwoPassOverlapUtr, recover_overlap_timing |
| `crates/batchalign-chat-ops/src/fa/tests.rs` | Integration tests for both strategies |
| `crates/batchalign-app/src/runner/dispatch/utr.rs` | resolve_strategy, UtrPassContext.overlap_strategy |
| `crates/batchalign-app/src/types/options.rs` | UtrOverlapStrategy enum |
| `crates/batchalign-cli/src/args/commands.rs` | `--utr-strategy` CLI flag |
