# Fuzzy Matching for DP Alignment: Proposal

**Status:** Draft
**Last updated:** 2026-03-18

## Problem

The Hirschberg DP alignment in `dp_align.rs` uses exact string comparison
(`MatchMode::CaseInsensitive`). When ASR output differs from the CHAT
transcript — substitutions, normalizations, dialectal forms — the aligner
treats them as mismatches. This can cause:

1. **UTR gaps**: ASR "gonna" vs transcript "going to" → no match → untimed
2. **Backchannel misses**: ASR "uh huh" vs CHAT "mhm" → no match
3. **Non-English drift**: ASR normalizations differ from CHAT conventions
4. **Compound splits**: ASR "ice cream" vs CHAT "ice+cream"

## Proposed Experiment

Add `MatchMode::Fuzzy` to `dp_align.rs` and evaluate empirically on the
existing test files (SBCSAE, Jefferson NB, TaiwanHakka).

### What to Measure

For each file, compare exact vs fuzzy matching:
- UTR coverage (% utterances timed)
- Timing precision (median start error vs ground truth)
- False positive rate (fuzzy matches that are actually wrong)
- Runtime cost

### Approach: Normalized Edit Distance

A match is accepted when:
```
edit_distance(a, b) / max(len(a), len(b)) <= threshold
```

Typical threshold: 0.3–0.4 (allows ~1 character difference per 3 characters).

### Implementation Options

**Option A: Use the `strsim` crate** (recommended)

[strsim](https://crates.io/crates/strsim) provides normalized Levenshtein,
Jaro-Winkler, and other metrics. Well-maintained, zero dependencies, pure Rust.

```rust
use strsim::normalized_levenshtein;

pub enum MatchMode {
    Exact,
    CaseInsensitive,
    Fuzzy { threshold: f64 },
}

fn words_match(a: &str, b: &str, mode: &MatchMode) -> bool {
    match mode {
        MatchMode::Exact => a == b,
        MatchMode::CaseInsensitive => a.eq_ignore_ascii_case(b),
        MatchMode::Fuzzy { threshold } => {
            // Fast path: exact match
            if a.eq_ignore_ascii_case(b) {
                return true;
            }
            // Fuzzy: normalized Levenshtein similarity (1.0 = identical)
            normalized_levenshtein(
                &a.to_lowercase(),
                &b.to_lowercase(),
            ) >= *threshold
        }
    }
}
```

Pros: well-tested, multiple metrics to compare, negligible binary size.
Cons: adds a dependency (though it's tiny and pure Rust).

**Option B: Roll our own normalized Levenshtein**

~30 lines of code. The standard DP algorithm with O(n*m) time/space.

```rust
fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let n = a.len();
    let m = b.len();
    if n == 0 && m == 0 { return 1.0; }

    let mut prev = (0..=m).collect::<Vec<_>>();
    let mut curr = vec![0; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i-1] == b[j-1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j-1] + 1)
                .min(prev[j-1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    1.0 - (prev[m] as f64 / n.max(m) as f64)
}
```

Pros: no dependency, simple.
Cons: less tested, only one metric, Unicode char-level (not grapheme).

**Recommendation:** Use `strsim`. It's 200 lines of pure Rust with zero
dependencies, well-tested, and gives us Jaro-Winkler as a bonus (better for
short words like backchannels where edit distance is too coarse).

## Where Fuzzy Matching Would Apply

| Component | Current | Fuzzy would help? |
|-----------|---------|-------------------|
| UTR global DP | Case-insensitive exact | Yes — ASR↔transcript substitutions |
| UTR pass-2 windowed DP | Case-insensitive exact | Yes — backchannel variants |
| Retokenizer char-level DP | Character-level exact | No — already character-level |
| WER computation | `wer_conform.rs` normalizes first | Maybe — but normalization handles most cases |

The highest-value target is UTR (both global and windowed) where ASR↔transcript
mismatches directly cause untimed utterances.

## Risks

1. **False positives**: "he" fuzzy-matching "she" (edit distance 1). Short words
   are particularly vulnerable. Jaro-Winkler handles this better than Levenshtein.

2. **Performance**: Fuzzy comparison is O(n*m) per word pair vs O(1) for exact.
   In the Hirschberg DP, this multiplies the constant factor. For typical
   utterances (10-50 words) this is negligible. For the full-file UTR DP
   (100-1000 words × 100-1000 ASR tokens), it could be noticeable.

3. **Threshold tuning**: The right threshold depends on language and ASR quality.
   English ASR is better → lower threshold needed. Non-English → higher threshold
   but more false positives. May need per-language tuning.

## Experiment Plan

1. Add `MatchMode::Fuzzy { threshold: f64 }` to `dp_align.rs`
2. Add `--utr-match-mode` CLI flag (exact/fuzzy)
3. Run alignment on the 20 existing test files with threshold 0.7, 0.8, 0.9
4. Compare coverage and precision against exact matching
5. Check for false positives by manual inspection of fuzzy matches

Estimated effort: ~2 hours for implementation + ~1 hour for experiment.

## Related Work

- `wer_conform.rs` already does text normalization (compound splitting, name
  replacement, filler expansion) before WER scoring. Some of these
  normalizations could be applied to UTR words instead of fuzzy matching.
- The ASR postprocessing pipeline (`asr_postprocess/`) normalizes ASR output
  before UTR sees it. Improving normalization is an alternative to fuzzy matching.
