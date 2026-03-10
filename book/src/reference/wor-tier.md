# %wor Tier Specification

How main tier words map to the %wor (word-level timing) dependent tier.

## Overview

The %wor tier is a **flat** list of words, each optionally paired with a
timing bullet. It mirrors the main tier's "phonated" words in the same
order, providing word-level audio timestamps. Unlike the main tier, %wor
never contains groups, annotations, replacements, events, pauses, or any
nested structure.

```
*CHI:    I want cookies .
%wor:    I 1000_1200 want 1200_1400 cookies 1400_1800 .
```

## 1-1 Correspondence

**Each word in %wor corresponds 1-1 to a phonated word in the main tier,
in document order.**

Both the forced alignment word extraction (`collect_fa_words`) and the %wor
generation (`generate_wor_tier`) walk the main tier AST identically,
applying the same alignability rules. This guarantees positional alignment:
%wor word at index N corresponds to the Nth phonated main tier word.

## What Text Appears in %wor

The %wor tier uses each word's **`cleaned_text`** — the linguistic content
with CHAT-specific prosodic markup removed:

| Main tier | cleaned_text (in %wor) | Notes |
|-----------|----------------------|-------|
| `a::n` | `an` | Lengthening `:` removed |
| `hel^lo` | `hello` | Syllable pause `^` removed |
| `som(e)thing` | `something` | Shortening expanded |
| `°softer°` | `softer` | CA delimiters removed |
| `⌈word⌉` | `word` | Overlap points removed |
| `&-uh` | `uh` | Category prefix `&-` stripped |
| `ice+cream` | `icecream` | Compound marker `+` removed |

## Inclusion Rules

### Words INCLUDED in %wor

The %wor tier includes words that were phonologically produced as standard
lexical items, filled pauses, or retraced speech:

| Form | Example | In %wor? | cleaned_text |
|------|---------|----------|-------------|
| Regular words | `want`, `cookie` | Yes | `want`, `cookie` |
| Fillers | `&-uh`, `&-um` | Yes | `uh`, `um` |
| Words with error marks | `goed [*]` | Yes | `goed` |
| Words inside retrace groups | `<I want> [/] I need` | Yes (all 4 words) | `I`, `want`, `I`, `need` |
| Words inside reformulation groups | `<I want> [//] I need` | Yes (all 4 words) | `I`, `want`, `I`, `need` |
| Words inside quotations | `+"/.` ... `+".` | Yes | word text |
| Words inside phonological groups | `[pho]` | Yes | word text |
| Words inside special form groups | `[sin]` | Yes | word text |
| MOR punctuation | `,`, `‡`, `„` | Yes | `,`, `‡`, `„` |
| Terminator punctuation | `.`, `?`, `!`, `+...` | Yes | terminator text |

### Words EXCLUDED from %wor

| Form | Example | Why excluded |
|------|---------|-------------|
| **Nonwords / babbling** | `&~gaga` | Not a recognizable lexical form; not phonated |
| **Phonological fragments** | `&+fr`, `&+w` | Incomplete word; not phonated |
| **Untranscribed material** | `xxx`, `yyy`, `www` | No identifiable word to align |
| **Omitted words** | `0is`, `0det` | Never spoken (`WordCategory::Omission`) |
| **CA-style omissions** | `(word)` in CA mode | Never spoken (`WordCategory::CAOmission`) |
| **Timing tokens** | `100_200` | %wor metadata artifacts, not lexical content |
| **Empty words** | (parser artifacts) | `cleaned_text` is empty string |

### Non-word items that never appear in %wor

These main tier elements are not words and are simply skipped during tree
traversal:

- **Pauses**: `(.)`, `(..)`, `(...)`, `(2.5)`
- **Events / actions**: `&=laughs`, `0 [=! vocalizes]`
- **Internal bullets**: timing markers between words
- **Linkers**: `++`, `+<`, `+^`, etc.
- **Postcodes**: `[+ text]`, `[+bch]`
- **Utterance-level annotations**: language codes `[- spa]`, etc.

## Replacement Words (`[: ...]`)

For words with replacement annotations (`original [: replacement]`):

**The REPLACEMENT word appears in %wor**, not the original. The replacement
is treated as a regular word and receives timing normally.

```
*CHI:    want [: wanted] cookie .
%wor:    wanted 1000_1200 cookie 1200_1600 .
```

This means replacement words with `[: ...]` are "substituted" for %wor
purposes — the original form is discarded and the replacement takes its
place as a normal phonated word.

### Fragment / nonword / untranscribed with replacement

When a form that would normally be excluded (`&+fr`, `&~gaga`, `xxx`, etc.)
has a `[: replacement]`, the replacement word takes its place:

```
*CHI:    &+fr [: friend] is here .
%wor:    friend 1000_1200 is 1200_1400 here 1400_1800 .
```

```
*CHI:    xxx [: something] is here .
%wor:    something 1000_1200 is 1200_1400 here 1400_1800 .
```

The excluded form is replaced by the replacement text, which becomes a
normal phonated word in %wor.

### Omission with replacement

If an omission (`0word`) has a replacement, the omission is still excluded
(the replacement does not rescue it):

```
*CHI:    0gonna [: going+to] eat .
         (omission — not in %wor regardless of replacement)
```

## Retrace and Reformulation Groups

Retraced and reformulated content (`<...> [/]`, `<...> [//]`, `<...> [///]`,
`<...> [/?]`) **IS included** in %wor.

This differs from %mor, where retraced content is excluded. The rationale
is:

- **%mor** = linguistic/morphological analysis → retraced words are
  corrected speech, not linguistically intended
- **%wor** = word-level audio timing → retraced words were phonologically
  produced and occupy audio time

```
*CHI:    <I want> [/] I need cookie .
%wor:    I 100_200 want 200_400 I 500_600 need 600_800 cookie 800_1200 .
```

The `generate_wor_tier` code calls `for_each_leaf` with `domain=None`,
which unconditionally descends into all groups including `AnnotatedGroup`
(retrace/reformulation groups are only skipped for the Mor domain).

## Timing Bullet Format

Each word may optionally have a timing bullet:

```
word \u0015start_ms_end_ms\u0015
```

Where:
- `\u0015` is the Unicode control character U+0015 (NAK), used as the CHAT
  bullet delimiter
- `start_ms` and `end_ms` are unsigned integers representing milliseconds
- Words without timing simply appear without a following bullet

Example raw encoding:
```
%wor:    hello \u00150_500\u0015 world \u0015500_1000\u0015 .
```

Words CAN lack timing bullets — this means timing is unknown, NOT an error.

## Tier-Level Structure

A complete %wor tier has:

```
%wor:\t[- lang_code] word1 [bullet1] word2 [bullet2] ... terminator [utterance_bullet]
```

| Component | Required | Notes |
|-----------|----------|-------|
| Language code | No | Inherited from main tier's `[- code]` |
| Words | Yes | Flat list of cleaned_text values |
| Timing bullets | No | Per-word, optional |
| Terminator | Yes | Same as main tier (`.`, `?`, `!`, `+...`, etc.) |
| Utterance bullet | No | Span of entire utterance (first word start to last word end) |

## Generation Pipeline

1. **Forced alignment engines** extract phonated words from the main tier
   AST via `collect_fa_words()`
2. The FA model processes the audio and returns per-word `[start_ms,
   end_ms]` pairs (or `null` for unaligned words)
3. Timings are injected back into the AST via
   `inject_timings_for_utterance()`, stored on each word's
   `timing_alignment` field
4. Post-processing (`postprocess_utterance_timings`) optionally chains end
   times and bounds timings within the utterance bullet range
5. `MainTier::generate_wor_tier()` walks the AST one final time, collecting
   each phonated word's `cleaned_text` and `timing_alignment` into a flat
   `WorTier`
6. The `WorTier` is serialized via `WriteChat` into the `%wor:\t...` line

Steps 1 and 5 both use `for_each_leaf()` with `domain=None`, guaranteeing
identical traversal order and 1-1 correspondence.

## Comparison with %mor Domain

| Aspect | %wor | %mor |
|--------|------|------|
| Fillers (`&-uh`) | Included | Excluded |
| Nonwords (`&~gaga`) | Excluded | Excluded |
| Fragments (`&+fr`) | Excluded | Excluded |
| Untranscribed (`xxx`, `yyy`, `www`) | Excluded | Excluded |
| Retraced groups (`<...> [/]`) | Included | Excluded |
| Replacement (`word [: repl]`) | Replacement text | Replacement text |
| Regular words | Included | Included |
| Omissions (`0word`) | Excluded | Excluded |
| Tag separators (`,`, `„`, `‡`) | Included | Included (as cm\|cm, etc.) |

## Source Code References

- **Content walker**: `talkbank-model/src/alignment/helpers/walk/` —
  `for_each_leaf()`, `for_each_leaf_mut()`, `ContentLeaf`, `ContentLeafMut`.
  Centralizes recursive traversal of `UtteranceContent` and `BracketedItem`;
  used by %wor generation, FA extraction, FA injection, and FA postprocessing.
- **Alignability rules**: `talkbank-model/src/alignment/helpers/rules.rs` —
  `word_is_alignable()`, `should_skip_group()`,
  `should_align_replaced_word_in_pho_sin()`
- **%wor tier model**: `talkbank-model/src/model/dependent_tier/wor.rs` —
  `WorWord`, `WorTier`, serialization
- **%wor generation from AST**:
  `talkbank-model/src/model/content/main_tier.rs` —
  `generate_wor_tier()`, `collect_wor_items_content()` (uses `for_each_leaf`)
- **FA word extraction**: `batchalign-chat-ops/src/fa/extraction.rs` —
  `collect_fa_words()` (uses `for_each_leaf`)
- **Timing injection**: `batchalign-chat-ops/src/fa/injection.rs` —
  `inject_timings_for_utterance()` (uses `for_each_leaf_mut`)
- **Timing postprocessing**: `batchalign-chat-ops/src/fa/postprocess.rs` —
  `postprocess_utterance_timings()` (uses both `for_each_leaf` and `for_each_leaf_mut`)
- **Word categories**:
  `talkbank-model/src/model/content/word/category.rs` —
  `WordCategory` enum
- **Untranscribed status**:
  `talkbank-model/src/model/content/word/untranscribed.rs` —
  `UntranscribedStatus` enum
- **Alignment domains**:
  `talkbank-model/src/alignment/helpers/domain.rs` —
  `AlignmentDomain` enum
