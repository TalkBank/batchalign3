//! Provenance newtypes for text flowing through the ASR post-processing pipeline.
//!
//! These types encode WHERE text is in the ASR→CHAT pipeline:
//!
//! - [`AsrRawText`] — raw tokens from an ASR provider (before any normalization)
//! - [`AsrNormalizedText`] — tokens after the full 8-stage pipeline
//! - [`ChatWordText`] — text ready for CHAT assembly via `DirectParser`
//!
//! The progression is: `AsrRawText` (on [`AsrElement`]) → `AsrNormalizedText`
//! (on [`AsrWord`]) → `ChatWordText` (on [`WordDesc`]).
//!
//! Each newtype follows the same pattern as [`ChatRawText`] / [`ChatCleanedText`]
//! in `text_types.rs`: `#[serde(transparent)]`, `new()`, `as_str()`, `Display`,
//! `AsRef<str>`.
//!
//! [`AsrElement`]: super::AsrElement
//! [`AsrWord`]: super::AsrWord
//! [`WordDesc`]: crate::build_chat::WordDesc
//! [`ChatRawText`]: crate::text_types::ChatRawText
//! [`ChatCleanedText`]: crate::text_types::ChatCleanedText

use serde::{Deserialize, Serialize};
use std::fmt;

/// Raw text from an ASR provider, before any normalization.
///
/// **Source**: Provider-specific bridge code (Rev.AI, Whisper, HK engines).
///
/// **Contains**: Digits, spaces, provider markers (`<pause>`), untouched
/// provider output. May include punctuation tokens, multi-word strings,
/// or language-specific characters that haven't been normalized yet.
///
/// The ASR post-processing pipeline reads this and produces
/// [`AsrNormalizedText`] after all 8 stages.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct AsrRawText(String);

impl AsrRawText {
    /// Wraps a string as raw ASR text.
    ///
    /// No validation is performed — the caller supplies text exactly as
    /// received from the ASR provider.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrows the raw ASR text for read-only inspection.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AsrRawText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for AsrRawText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<&str> for AsrRawText {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

/// ASR text after the full normalization pipeline (8 stages).
///
/// **Source**: `process_raw_asr()` in `asr_postprocess/mod.rs`.
///
/// **Contains**: Compound-merged, number-expanded, disfluency-marked text.
/// Filled pauses are in `&-um` form, orthographic replacements applied
/// (`'cause` → `(be)cause`), Cantonese normalization done (for `yue`).
///
/// **NOT yet CHAT syntax** — still needs `DirectParser` to become AST nodes.
/// The next step is conversion to [`ChatWordText`] at the boundary between
/// ASR post-processing and CHAT assembly.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct AsrNormalizedText(String);

impl AsrNormalizedText {
    /// Wraps a string as normalized ASR text.
    ///
    /// Call this after the normalization pipeline has processed the text
    /// through compound merging, number expansion, disfluency replacement,
    /// and retrace detection.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrows the normalized text for read-only inspection.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Applies a transformation to the inner text, returning a new wrapper.
    ///
    /// Useful for pipeline stages that mutate text in place:
    /// ```ignore
    /// w.text = w.text.map(|t| expand_number(t, lang));
    /// ```
    pub fn map(self, f: impl FnOnce(&str) -> String) -> Self {
        Self(f(&self.0))
    }

    /// Appends a string to the inner text.
    ///
    /// Used by hyphen-joining in `split_multiword_tokens`.
    pub fn push_str(&mut self, s: &str) {
        self.0.push_str(s);
    }

    /// Returns a lowercase copy of the inner text.
    pub fn to_lowercase(&self) -> String {
        self.0.to_lowercase()
    }

    /// Returns `true` if the text starts with the given pattern.
    pub fn starts_with(&self, pat: char) -> bool {
        self.0.starts_with(pat)
    }
}

impl fmt::Display for AsrNormalizedText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for AsrNormalizedText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<&str> for AsrNormalizedText {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

/// Text ready for CHAT assembly via `DirectParser`.
///
/// **Source**: `transcript_from_asr_utterances()` in `build_chat.rs`.
///
/// **Semantically identical** to [`AsrNormalizedText`] but marks the boundary
/// crossing from the ASR domain into the CHAT domain. After this point,
/// the text is parsed by `talkbank_direct_parser` into CHAT AST nodes.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ChatWordText(String);

impl ChatWordText {
    /// Wraps a string as CHAT-ready word text.
    ///
    /// The caller supplies text that has been fully normalized by the ASR
    /// pipeline and is ready for parsing into the CHAT AST.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrows the word text for read-only use.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ChatWordText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ChatWordText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl PartialEq<&str> for ChatWordText {
    fn eq(&self, other: &&str) -> bool {
        self.0 == *other
    }
}

// ---------------------------------------------------------------------------
// Timing and speaker newtypes
// ---------------------------------------------------------------------------

/// Timestamp in seconds from an ASR provider (raw timing).
///
/// ASR providers report element boundaries in fractional seconds.
/// This newtype distinguishes provider timestamps from the millisecond
/// timings used internally by `AsrWord` (plain `i64`).
#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct AsrTimestampSecs(pub f64);

impl AsrTimestampSecs {
    /// Returns the inner `f64` value.
    pub fn as_f64(self) -> f64 {
        self.0
    }
}

impl PartialEq<f64> for AsrTimestampSecs {
    fn eq(&self, other: &f64) -> bool {
        self.0 == *other
    }
}

impl fmt::Display for AsrTimestampSecs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}s", self.0)
    }
}

/// Zero-based speaker index within a recording.
///
/// Maps to participant codes (`PAR`, `INV`, `SP0`, etc.) during CHAT
/// assembly in `transcript_from_asr_utterances()`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct SpeakerIndex(pub usize);

impl SpeakerIndex {
    /// Returns the inner `usize` value.
    pub fn as_usize(self) -> usize {
        self.0
    }
}

impl PartialEq<usize> for SpeakerIndex {
    fn eq(&self, other: &usize) -> bool {
        self.0 == *other
    }
}

impl fmt::Display for SpeakerIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_transparency() {
        let raw = AsrRawText::new("hello world");
        let json = serde_json::to_string(&raw).unwrap();
        assert_eq!(json, "\"hello world\"");
        let decoded: AsrRawText = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, raw);
    }

    #[test]
    fn normalized_map() {
        let text = AsrNormalizedText::new("5");
        let mapped = text.map(|t| t.replace('5', "five"));
        assert_eq!(mapped.as_str(), "five");
    }

    #[test]
    fn normalized_push_str() {
        let mut text = AsrNormalizedText::new("hello");
        text.push_str("-world");
        assert_eq!(text.as_str(), "hello-world");
    }

    #[test]
    fn chat_word_text_display() {
        let text = ChatWordText::new("(be)cause");
        assert_eq!(format!("{text}"), "(be)cause");
    }

    #[test]
    fn timestamp_serde_roundtrip() {
        let ts = AsrTimestampSecs(1.234);
        let json = serde_json::to_string(&ts).unwrap();
        assert_eq!(json, "1.234");
        let decoded: AsrTimestampSecs = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, ts);
    }

    #[test]
    fn speaker_index_serde_roundtrip() {
        let idx = SpeakerIndex(3);
        let json = serde_json::to_string(&idx).unwrap();
        assert_eq!(json, "3");
        let decoded: SpeakerIndex = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, idx);
    }
}
