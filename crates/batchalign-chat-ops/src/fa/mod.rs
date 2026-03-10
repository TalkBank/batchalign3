//! Forced alignment orchestration for CHAT files.
//!
//! Extracts pure-Rust FA logic from the PyO3 bridge (`batchalign-core`) so that
//! both the PyO3 layer and the root Rust workspace can share it.
//!
//! Pipeline: parse -> group utterances -> dispatch inference -> parse responses
//! -> inject timings -> postprocess -> generate %wor -> enforce monotonicity/E704.

mod alignment;
mod extraction;
mod grouping;
mod injection;
mod orchestrate;
mod postprocess;
pub mod utr;

#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use talkbank_model::alignment::helpers::{
    AlignmentDomain, ContentLeaf, for_each_leaf, word_is_alignable,
};
use talkbank_model::model::{Bullet, ChatFile, DependentTier, Line, Utterance, Word};

use crate::indices::{UtteranceIdx, WordIdx};

// Re-export public API so that `crate::fa::Foo` paths remain unchanged.
pub use self::alignment::parse_fa_response;
pub use self::extraction::collect_fa_words;
pub use self::grouping::{count_utterance_timing, estimate_untimed_boundaries, group_utterances};
pub use self::injection::inject_timings_for_utterance;
pub use self::orchestrate::{
    apply_fa_results, enforce_monotonicity, has_reusable_wor_timing_for_utterance,
    refresh_existing_alignment, refresh_existing_alignment_for_utterance,
    refresh_reusable_utterances, strip_e704_same_speaker_overlaps, strip_timing_from_content,
};
pub use self::postprocess::postprocess_utterance_timings;
pub use self::utr::{
    GlobalUtr, GroupingContext, TwoPassOverlapUtr, UtrStrategy, find_untimed_windows,
    select_strategy, utr_asr_cache_key, utr_asr_segment_cache_key,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A time interval in milliseconds, guaranteed start <= end at construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSpan {
    /// Start time in milliseconds.
    pub start_ms: u64,
    /// End time in milliseconds.
    pub end_ms: u64,
}

impl TimeSpan {
    /// Create a new time span. Caller is responsible for ensuring start <= end.
    pub fn new(start_ms: u64, end_ms: u64) -> Self {
        Self { start_ms, end_ms }
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// A timing result for a single word (alias for [`TimeSpan`]).
pub type WordTiming = TimeSpan;

/// A word extracted for forced alignment, with its position in the AST.
#[derive(Debug, Clone)]
pub struct FaWord {
    /// Index of the utterance in the file (among utterances only).
    pub utterance_index: UtteranceIdx,
    /// Index among alignable words within the utterance.
    pub utterance_word_index: WordIdx,
    /// Cleaned text for the FA model.
    pub text: String,
}

impl FaWord {
    /// Stable word identifier for callback protocols.
    pub fn stable_id(&self) -> String {
        format!("u{}:w{}", self.utterance_index, self.utterance_word_index)
    }
}

/// A group of utterances clustered for a single FA call.
#[derive(Debug)]
pub struct FaGroup {
    /// Audio window for this group.
    #[allow(dead_code)]
    pub audio_span: TimeSpan,
    /// Words in this group with positional indices.
    pub words: Vec<FaWord>,
    /// Utterance indices included in this group.
    pub utterance_indices: Vec<UtteranceIdx>,
}

impl FaGroup {
    /// Start of the audio window (ms).
    pub fn audio_start_ms(&self) -> u64 {
        self.audio_span.start_ms
    }

    /// End of the audio window (ms).
    pub fn audio_end_ms(&self) -> u64 {
        self.audio_span.end_ms
    }
}

/// Controls how word end times are set during FA post-processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaTimingMode {
    /// End of each word = start of next word (no silence between words).
    /// Used when the FA engine returns onset-only times (Wave2Vec).
    Continuous,
    /// End of each word = engine-provided end time (preserves pauses).
    /// Used when the FA engine returns word-level start+end (Whisper).
    WithPauses,
}

/// Wire type for the FA infer protocol -- one group sent to a Python worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaInferItem {
    /// Words to align (cleaned text).
    pub words: Vec<String>,
    /// Stable word IDs aligned 1:1 with `words`.
    pub word_ids: Vec<String>,
    /// Utterance indices aligned 1:1 with `words`.
    pub word_utterance_indices: Vec<usize>,
    /// Word indices inside each utterance aligned 1:1 with `words`.
    pub word_utterance_word_indices: Vec<usize>,
    /// Path to the audio file.
    pub audio_path: String,
    /// Start of the audio window (ms).
    pub audio_start_ms: u64,
    /// End of the audio window (ms).
    pub audio_end_ms: u64,
    /// How to handle word end times during post-processing.
    pub timing_mode: FaTimingMode,
}

impl FaInferItem {
    /// Audio window as a [`TimeSpan`].
    pub fn audio_span(&self) -> TimeSpan {
        TimeSpan::new(self.audio_start_ms, self.audio_end_ms)
    }
}

/// The forced alignment engine that produced word timings.
///
/// Determines how FA responses are interpreted:
/// - WhisperFa returns token-level onset times → requires DP alignment
/// - Wave2Vec returns word-level start+end pairs → index-aligned
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaEngineType {
    /// Whisper token-level FA. Onset times only; Hirschberg DP alignment
    /// maps tokens to words.
    WhisperFa,
    /// Wav2Vec word-level FA. Start+end times per word, 1:1 index-aligned
    /// with input words.
    Wave2Vec,
}

impl FaEngineType {
    /// Wire-format string for cache keys and serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::WhisperFa => "whisper_fa",
            Self::Wave2Vec => "wave2vec",
        }
    }

    /// Parse from a wire-format string.
    ///
    /// Matches both `"wav2vec"` and `"wave2vec"` spellings, since the CLI
    /// generates `"wav2vec_fa"` while some older code used `"wave2vec"`.
    pub fn from_str_lossy(s: &str) -> Self {
        if s.contains("wav2vec") || s.contains("wave2vec") {
            Self::Wave2Vec
        } else {
            Self::WhisperFa
        }
    }
}

// ---------------------------------------------------------------------------
// %wor tier management
// ---------------------------------------------------------------------------

/// Remove existing %wor tier from an utterance (if any).
pub fn remove_wor_tier(utterance: &mut Utterance) {
    utterance
        .dependent_tiers
        .retain(|t| !matches!(t, DependentTier::Wor(_)));
}

/// Add a %wor tier generated from the inline bullets on words.
pub fn add_wor_tier(utterance: &mut Utterance) {
    remove_wor_tier(utterance);
    let wor_tier = utterance.main.generate_wor_tier();
    utterance.dependent_tiers.push(DependentTier::Wor(wor_tier));
}

/// Return `true` when every alignable FA word in the file already has reusable
/// `%wor` timing.
///
/// This intentionally does **not** look at `main` tier `inline_bullet` alone.
/// After a parse roundtrip, main-tier word timing may be represented as
/// `InternalBullet` tokens, while `%wor` carries the durable first-class timing
/// bullets. For the cheap rerun path we therefore verify that `%wor` fully and
/// cleanly aligns back to the main tier.
pub fn has_reusable_wor_timing(chat_file: &ChatFile) -> bool {
    let mut saw_alignable_word = false;

    for line in &chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };

        let main_word_count = count_alignable_main_words(utterance);
        if main_word_count == 0 {
            continue;
        }
        saw_alignable_word = true;

        if !has_reusable_wor_timing_for_utterance(utterance) {
            return false;
        }
    }

    saw_alignable_word
}

/// Find utterance indices that have reusable `%wor` timing.
///
/// Returns a set of utterance ordinal indices where
/// [`has_reusable_wor_timing_for_utterance()`] succeeds. Used by the plain
/// rerun path to selectively skip FA for utterances whose `%wor` is still
/// clean after manual edits to other utterances.
pub fn find_reusable_utterance_indices(chat_file: &ChatFile) -> std::collections::HashSet<usize> {
    let mut reusable = std::collections::HashSet::new();
    let mut utt_idx = 0usize;
    for line in &chat_file.lines {
        let Line::Utterance(utterance) = line else {
            continue;
        };
        if count_alignable_main_words(utterance) > 0
            && has_reusable_wor_timing_for_utterance(utterance)
        {
            reusable.insert(utt_idx);
        }
        utt_idx += 1;
    }
    reusable
}

/// Count Wor-alignable words in the main tier.
pub(crate) fn count_alignable_main_words(utterance: &Utterance) -> usize {
    let mut count = 0usize;
    for_each_leaf(
        &utterance.main.content.content,
        None,
        &mut |leaf| match leaf {
            ContentLeaf::Word(word, _annotations) => {
                if word_is_alignable(word, AlignmentDomain::Wor) {
                    count += 1;
                }
            }
            ContentLeaf::ReplacedWord(replaced) => {
                if !replaced.replacement.words.is_empty() {
                    for word in &replaced.replacement.words {
                        if word_is_alignable(word, AlignmentDomain::Wor) {
                            count += 1;
                        }
                    }
                } else if word_is_alignable(&replaced.word, AlignmentDomain::Wor) {
                    count += 1;
                }
            }
            ContentLeaf::Separator(_) => {}
        },
    );
    count
}

/// Update the utterance-level bullet from word timings.
///
/// When the utterance already has a bullet (e.g., from hand-linked input),
/// the result is the **union** of the original bullet and the word timing
/// span — the bullet can expand but never shrink. This preserves coverage
/// of fillers, pauses, gestures, and false starts that FA cannot align.
///
/// When there is no pre-existing bullet, sets it from the word timing span.
pub fn update_utterance_bullet(utterance: &mut Utterance) {
    let mut first_start: Option<u64> = None;
    let mut last_end: Option<u64> = None;

    let mut timings: Vec<Option<TimeSpan>> = Vec::new();
    postprocess::collect_word_timings(&utterance.main.content.content, &mut timings);

    for span in timings.iter().flatten() {
        if first_start.is_none() || span.start_ms < first_start.unwrap() {
            first_start = Some(span.start_ms);
        }
        if last_end.is_none() || span.end_ms > last_end.unwrap() {
            last_end = Some(span.end_ms);
        }
    }

    if let (Some(word_start), Some(word_end)) = (first_start, last_end) {
        // Union with original bullet: never shrink, only expand or create.
        let (final_start, final_end) = if let Some(ref existing) = utterance.main.content.bullet {
            (
                word_start.min(existing.timing.start_ms),
                word_end.max(existing.timing.end_ms),
            )
        } else {
            (word_start, word_end)
        };
        utterance.main.content.bullet = Some(Bullet::new(final_start, final_end));
    }
}

/// Collect current main-tier word timings in the exact order FA uses for
/// extraction and injection.
///
/// This is the stable timing surface for selective reuse: when an utterance has
/// already been refreshed from `%wor`, the returned vector can be stitched
/// directly into a preserved FA group without a worker roundtrip.
pub fn collect_existing_fa_word_timings(utterance: &Utterance) -> Vec<Option<WordTiming>> {
    let mut timings = Vec::new();
    for_each_leaf(
        &utterance.main.content.content,
        None,
        &mut |leaf| match leaf {
            ContentLeaf::Word(word, _annotations) => {
                if word_is_alignable(word, AlignmentDomain::Wor) {
                    timings.push(get_word_timing(word));
                }
            }
            ContentLeaf::ReplacedWord(replaced) => {
                if !replaced.replacement.words.is_empty() {
                    for word in &replaced.replacement.words {
                        if word_is_alignable(word, AlignmentDomain::Wor) {
                            timings.push(get_word_timing(word));
                        }
                    }
                } else if word_is_alignable(&replaced.word, AlignmentDomain::Wor) {
                    timings.push(get_word_timing(&replaced.word));
                }
            }
            ContentLeaf::Separator(_) => {}
        },
    );
    timings
}

// ---------------------------------------------------------------------------
// Helpers shared across submodules
// ---------------------------------------------------------------------------

/// Get a mutable reference to the nth utterance in the file.
pub(super) fn get_utterance_mut(
    chat_file: &mut talkbank_model::model::ChatFile,
    utt_idx: UtteranceIdx,
) -> Option<&mut Utterance> {
    use talkbank_model::model::Line;
    let mut current = 0;
    for line in &mut chat_file.lines {
        if let Line::Utterance(utt) = line {
            if current == utt_idx.raw() {
                return Some(utt);
            }
            current += 1;
        }
    }
    None
}

/// Get the inline timing from a word, if present.
pub(super) fn get_word_timing(word: &Word) -> Option<TimeSpan> {
    word.inline_bullet
        .as_ref()
        .map(|b| TimeSpan::new(b.timing.start_ms, b.timing.end_ms))
}

// ---------------------------------------------------------------------------
// AudioIdentity
// ---------------------------------------------------------------------------

/// Content identity for an audio file used in FA cache keys.
///
/// # Invariant
///
/// Format: `"{resolved_path}|{mtime_secs}|{file_size}"`. Fast identity
/// based on filesystem metadata (no file content hashing). Created by
/// [`AudioIdentity::from_metadata`] in the server runner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AudioIdentity(String);

impl AudioIdentity {
    /// Build an identity from resolved path + filesystem metadata.
    pub fn from_metadata(path: &str, mtime_secs: u64, size: u64) -> Self {
        Self(format!("{path}|{mtime_secs}|{size}"))
    }

    /// Access the raw identity string (for display/logging).
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AudioIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Cache key computation
// ---------------------------------------------------------------------------

/// Compute cache key for an FA result.
///
/// Key = `BLAKE3("{audio_identity}|{start}|{end}|{text}|{timing_flag}|{engine}")`.
pub fn cache_key(
    words: &[String],
    audio_identity: &AudioIdentity,
    start_ms: u64,
    end_ms: u64,
    timing_mode: FaTimingMode,
    engine: FaEngineType,
) -> crate::CacheKey {
    let text = words.join(" ");
    let timing_flag = match timing_mode {
        FaTimingMode::Continuous => "no_pauses",
        FaTimingMode::WithPauses => "pauses",
    };
    let engine_str = engine.as_str();
    let input = format!(
        "{}|{start_ms}|{end_ms}|{text}|{timing_flag}|{engine_str}",
        audio_identity.0
    );
    crate::CacheKey::from_content(&input)
}
