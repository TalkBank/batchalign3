# Main Tier Content Audit for Alignment

**Status:** Current
**Last updated:** 2026-03-18

## Purpose

Audit every content type on CHAT main tiers to determine what information
is available for alignment, what we currently use, what we drop, and what
we could potentially use to improve alignment quality.

## Corpus-Wide Usage Frequencies

Counts across ca-data + childes-data + aphasia-data:

| Construct | Count | Notes |
|-----------|-------|-------|
| **Fillers** (`&-um`, `&-uh`) | 670,873+ | Acoustically produced, INCLUDED in alignment |
| **Untranscribed** (`xxx`) | 1,441,559 | Not in audio as words, correctly EXCLUDED |
| **Events** (`&=laughs`) | 98,383+ | Acoustically present but not words |
| **Nonwords** (`&~`) | 45,694+ | Acoustically present, currently EXCLUDED |
| **Fragments** (`&+`) | 16,590+ | Acoustically present, currently EXCLUDED |
| **`&*SPK:word`** (other spoken) | 22,545 | Background speech, correctly EXCLUDED from main speaker |
| **`www`** (untranscribed) | 182,590 | Correctly EXCLUDED |
| **Quotations** (`"..."`) | 5,743 files | Spoken content, INCLUDED |
| **SinGroups** (`〔...〕`) | 1 file | Gesture, no acoustic content |
| **PhoGroups** (`‹...›`) | 0 files | Not used in data |

## Content Decision Table

### Currently INCLUDED and correct

| Content | Count | Acoustic? | Rationale |
|---------|-------|-----------|-----------|
| Regular words | (majority) | Yes | Core alignable content |
| Fillers (`&-um`, `&-uh`) | 670K+ | Yes | Speaker produces "um" — ASR detects it |
| Retrace groups (`<word> [/]`) | common | Yes | Speaker produced the words (Wor includes, Mor excludes) |
| Quotations (`"..."`) | 5.7K files | Yes | Quoted speech was spoken |
| Replaced words (`word [: fix]`) | common | Yes | Original was spoken; replacement used for alignment |

### Currently EXCLUDED and correct

| Content | Count | Acoustic? | Rationale |
|---------|-------|-----------|-----------|
| Untranscribed (`xxx`, `yyy`, `www`) | 1.65M | No (as words) | Nothing to match — ASR can't produce these |
| Omissions (`0word`) | rare | No | Word was NOT produced |
| Events (`&=laughs`) | 98K+ | Yes (non-speech) | Not words — ASR doesn't produce "laughs" as a token |
| Pauses (`(.)`, `(1.5)`) | very common | Yes (silence) | Silence, not a word — ASR doesn't tokenize silence |
| `&*SPK:word` (other spoken) | 22K+ | Yes | Other speaker's speech — not this speaker's alignment |
| Markers (overlap, bullets, etc.) | common | N/A | Metadata, not content |

### Currently EXCLUDED — should we reconsider?

| Content | Count | Acoustic? | Current status | Potential value | Risk |
|---------|-------|-----------|---------------|----------------|------|
| **Nonwords** (`&~gaga`) | 45K+ | **Yes** | Excluded | Could provide timing anchors — speaker produces these sounds, ASR might detect fragments of them | ASR won't recognize "gaga" as a word → DP mismatch, could shift alignment |
| **Fragments** (`&+fr`) | 16K+ | **Yes** | Excluded | Partial word attempts are acoustically present — ASR may detect the beginning of the word | Same risk: ASR produces full words, fragments are partial → mismatch |

### Information we have but DON'T use for alignment

| Information | Where | How it could help |
|-------------|-------|-------------------|
| **Pause durations** `(1.5)` | Main tier | Timed pauses give inter-word silence boundaries — could constrain where words fall in the audio |
| **Event positions** `&=laughs` | Main tier | Events mark non-speech audio segments — could help skip over non-speech in ASR |
| **CA terminators** `⇘ → ↗` | Main tier | Intonation patterns carry acoustic information — could inform utterance boundary detection |
| **Speaker identity** | `@Participants` | Multi-speaker alignment could use diarization output to constrain which ASR segments belong to which speaker |
| **Overlap markers** `⌈⌉⌊⌋` | Main tier | Already using for onset windowing (this session's work) |
| **Timing bullets** on some utterances | Main tier | Already using for UTR anchoring |
| **`@Languages`** | Header | Should be passed to Whisper for language detection |
| **Word lengthening** `:` | WordContent | Indicates sustained vowel — could adjust expected word duration |
| **Stress markers** `ˈ ˌ` | WordContent | Prosodic information not currently used |

## Key Findings

### 1. Pauses are the biggest untapped resource

Timed pauses like `(1.5)` tell us there's 1.5 seconds of silence between
two words. This information could:
- Provide inter-word timing constraints for FA
- Help UTR place utterances more accurately
- Reduce false DP matches across silence gaps

**Impact:** High.
**Effort:** Moderate — need to thread pause durations through the alignment pipeline.

### 2. Events could mark non-speech segments

`&=laughs` at position N tells us there's laughter in the audio between
words N-1 and N+1. ASR doesn't produce "laughs" as a token, so this
segment in the audio is unaccounted for. If we know where events are, we
could:
- Skip event-duration audio in FA group construction
- Prevent ASR word tokens from being placed during laughter

**Impact:** Moderate. Depends on how often events cause timing misalignment.
**Effort:** High — need to estimate event durations and integrate with FA grouping.

### 3. Nonwords and fragments are a precision/coverage tradeoff

Including `&~gaga` and `&+fr` would add timing anchors (they're acoustically
present) but could cause DP alignment errors (ASR won't produce matching
tokens). Net effect is unclear — needs experimentation.

**Impact:** Unknown.
**Effort:** Low — just change `is_wor_excluded_word()` rules.

### 4. `cleaned_text()` drops useful information

The `cleaned_text()` function strips:
- Lengthening markers (`:`) → "hel:lo" becomes "hello"
- Stress markers (`ˈ ˌ`) → stripped
- CA elements (prosody) → stripped
- Shortenings (restored) → "(lo)" in "hel(lo)" becomes "hello"

This is correct for ASR matching (ASR doesn't produce these markers). But
the stripped information could theoretically inform expected word durations
(lengthened words take longer).

### 5. ~~Language code is not being passed to ASR~~ — VERIFIED OK

The `@Languages` header IS passed to Whisper's `language` parameter via
`gen_kwargs()` in `types.py`. Verified identical to batchalign2-master.
The poor non-English ASR (Welsh 9.6s, German 23s median error) is a
model quality issue, not a missing parameter.

## Recommendations

**Priority 1 (investigate):**
- ~~Verify `@Languages` is passed to Whisper~~ — DONE, confirmed working
- Measure pause-aware alignment improvement

**Priority 2 (experiment):**
- Test including fragments (`&+`) in alignment on aphasia data
- Test pause-constrained FA grouping

**Priority 3 (longer term):**
- Event-aware FA grouping
- Speaker diarization integration for multi-speaker files
