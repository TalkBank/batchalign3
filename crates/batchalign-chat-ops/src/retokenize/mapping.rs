//! Word-to-token mapping: maps original CHAT word indices to Stanza token indices.

use crate::extract::ExtractedWord;
use smallvec::SmallVec;

/// Maps each CHAT word (by index) to one or more Stanza token indices.
///
/// Invariant: `word_count() == original_word_count` passed at construction.
/// Each entry is non-empty for successfully mapped words (may be empty for
/// unmapped words, which triggers fallback behavior in rebuild).
///
/// # Arena-style optimizations
///
/// Uses a dense `Vec<SmallVec<[usize; 4]>>` indexed by word index instead
/// of `HashMap<usize, Vec<usize>>`, eliminating hashing overhead and
/// providing O(1) lookup. Most words map to 1-2 tokens, so `SmallVec<[usize; 4]>`
/// keeps them inline without heap allocation.
///
/// See `book/src/developer/arena-allocators.md` Patterns 4 and 5.
#[derive(Debug, Clone)]
pub struct WordTokenMapping {
    inner: Vec<SmallVec<[usize; 4]>>,
}

#[allow(dead_code)]
impl WordTokenMapping {
    /// Number of original words this mapping covers.
    pub fn word_count(&self) -> usize {
        self.inner.len()
    }

    /// Token indices for the given original word index.
    ///
    /// Returns an empty slice if the word has no mapping.
    pub fn tokens_for_word(&self, word_idx: usize) -> &[usize] {
        self.inner.get(word_idx).map_or(&[], |s| s.as_slice())
    }

    /// Get a non-empty mapping for the given word, or `None`.
    pub fn get_nonempty(&self, word_idx: usize) -> Option<&[usize]> {
        self.inner
            .get(word_idx)
            .filter(|v| !v.is_empty())
            .map(|s| s.as_slice())
    }
}

/// Build a mapping: original_word_index -> Vec<stanza_token_index>
///
/// First tries deterministic span-join mapping when normalized concatenated text
/// is identical on both sides. When text diverges, uses a conservative
/// length-aware monotonic fallback (no DP).
pub fn build_word_token_mapping(
    original_words: &[ExtractedWord],
    stanza_tokens: &[String],
) -> WordTokenMapping {
    if let Some(mapping) = try_deterministic_word_token_mapping(original_words, stanza_tokens) {
        return WordTokenMapping { inner: mapping };
    }

    tracing::warn!(
        original_word_count = original_words.len(),
        stanza_token_count = stanza_tokens.len(),
        "retokenize text diverged; using length-aware monotonic fallback without DP"
    );

    WordTokenMapping {
        inner: build_length_fallback_mapping(original_words.len(), stanza_tokens.len()),
    }
}

/// Normalize a text unit for alignment comparison (lowercase).
pub fn normalize_alignment_unit(text: &str) -> String {
    text.chars().flat_map(|ch| ch.to_lowercase()).collect()
}

/// Try to build a deterministic span-join mapping.
///
/// Returns `None` if normalized text doesn't match (fallback should be used).
pub fn try_deterministic_word_token_mapping(
    original_words: &[ExtractedWord],
    stanza_tokens: &[String],
) -> Option<Vec<SmallVec<[usize; 4]>>> {
    if original_words.is_empty() || stanza_tokens.is_empty() {
        return Some(vec![SmallVec::new(); original_words.len()]);
    }

    let mut original_ranges: Vec<(usize, usize)> = Vec::with_capacity(original_words.len());
    let mut token_ranges: Vec<(usize, usize)> = Vec::with_capacity(stanza_tokens.len());
    let mut original_concat = String::new();
    let mut token_concat = String::new();

    let mut cursor = 0usize;
    for word in original_words {
        let normalized = normalize_alignment_unit(word.text.as_str());
        if normalized.is_empty() {
            return None;
        }
        let len = normalized.chars().count();
        original_ranges.push((cursor, cursor + len));
        cursor += len;
        original_concat.push_str(&normalized);
    }

    cursor = 0;
    for token in stanza_tokens {
        let normalized = normalize_alignment_unit(token);
        if normalized.is_empty() {
            return None;
        }
        let len = normalized.chars().count();
        token_ranges.push((cursor, cursor + len));
        cursor += len;
        token_concat.push_str(&normalized);
    }

    if original_concat != token_concat {
        return None;
    }

    let mut mapping: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); original_words.len()];
    let mut token_idx = 0usize;

    for (word_idx, &(word_start, word_end)) in original_ranges.iter().enumerate() {
        while token_idx < token_ranges.len() && token_ranges[token_idx].1 <= word_start {
            token_idx += 1;
        }

        let mut cursor_idx = token_idx;
        while cursor_idx < token_ranges.len() {
            let (token_start, token_end) = token_ranges[cursor_idx];
            if token_start >= word_end {
                break;
            }
            if token_end > word_start {
                mapping[word_idx].push(cursor_idx);
            }
            cursor_idx += 1;
        }

        if mapping[word_idx].is_empty() {
            return None;
        }
    }

    Some(mapping)
}

pub(super) fn build_length_fallback_mapping(
    original_word_count: usize,
    stanza_token_count: usize,
) -> Vec<SmallVec<[usize; 4]>> {
    let mut mapping: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); original_word_count];

    if original_word_count == 0 || stanza_token_count == 0 {
        return mapping;
    }

    if original_word_count == stanza_token_count {
        for (idx, slot) in mapping.iter_mut().enumerate() {
            slot.push(idx);
        }
        return mapping;
    }

    for (word_idx, slot) in mapping.iter_mut().enumerate() {
        let start = word_idx * stanza_token_count / original_word_count;
        let mut end = (word_idx + 1) * stanza_token_count / original_word_count;
        if end <= start {
            end = (start + 1).min(stanza_token_count);
        }
        for token_idx in start..end {
            slot.push(token_idx);
        }
    }

    mapping
}
