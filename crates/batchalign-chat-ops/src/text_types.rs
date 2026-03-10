//! Provenance-encoding newtype wrappers for all text in the system.
//!
//! Every string is wrapped in a newtype that encodes WHERE it came from.
//! This prevents mixing text from different sources and makes data flow explicit.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Raw text as it appears on a CHAT main tier, before any cleaning.
///
/// **Source**: `Word::raw_text()` from the CHAT AST.
///
/// **Contains**: The full surface form including CHAT markers (`@c`, `@s`),
/// timing bullets, annotations, and special form notation. This is the
/// text that would appear in a `.cha` file, not what you would send to
/// an NLP model.
///
/// Use [`ChatCleanedText`] instead when you need text suitable for
/// linguistic processing (morphosyntax, alignment, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ChatRawText(String);

impl ChatRawText {
    /// Wraps an arbitrary string as raw CHAT text.
    ///
    /// No validation or normalization is performed -- the caller is
    /// responsible for providing text that faithfully represents the
    /// CHAT surface form (markers, bullets, and all).
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrows the inner string for read-only inspection.
    ///
    /// The returned slice includes all CHAT markers and annotations.
    /// To get cleaned text suitable for NLP, extract a [`ChatCleanedText`]
    /// from the AST instead.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ChatRawText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ChatRawText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// Lexical content extracted from CHAT, with all markup stripped.
///
/// **Source**: `Word::cleaned_text()` from the CHAT AST.
///
/// **Contains**: Pure lexical content -- CHAT markers (`@c`, `@s`),
/// timing bullets, annotations, and special-form brackets have all been
/// removed by the parser. This is the text that should be sent to NLP
/// models (Stanza, alignment, translation, etc.).
///
/// The cleaning is performed at parse time by the CHAT parser, not by
/// this type. Constructing a `ChatCleanedText` via [`new()`](Self::new)
/// does not perform any stripping -- callers must supply already-cleaned
/// content.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct ChatCleanedText(String);

impl ChatCleanedText {
    /// Wraps an already-cleaned string as cleaned CHAT text.
    ///
    /// No stripping or normalization is performed here -- the caller
    /// must supply text that has already had CHAT markers removed
    /// (typically obtained from the parser's `Word::cleaned_text()`).
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrows the cleaned string for read-only use.
    ///
    /// The returned slice contains only lexical content and is safe to
    /// pass directly to NLP models, hash for cache keys, or use in
    /// sequence alignment.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns an iterator over the chars.
    pub fn chars(&self) -> std::str::Chars<'_> {
        self.0.chars()
    }

    /// Returns the lowercase equivalent.
    pub fn to_lowercase(&self) -> String {
        self.0.to_lowercase()
    }
}

impl fmt::Display for ChatCleanedText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for ChatCleanedText {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A participant identifier from the CHAT `@Participants` header.
///
/// **Source**: `Utterance.speaker` field from the CHAT AST.
///
/// **Format**: Conventionally a 3-letter uppercase code (`CHI`, `MOT`,
/// `INV`, etc.), though the CHAT spec permits codes of other lengths.
/// The code uniquely identifies a speaker within a single transcript
/// and is used to key per-speaker analysis (MLU, frequency counts, etc.).
///
/// See <https://talkbank.org/0info/manuals/CHAT.html#Participants> for
/// the full specification of participant codes and roles.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
#[repr(transparent)]
pub struct SpeakerCode(String);

impl SpeakerCode {
    /// Wraps a string as a speaker code.
    ///
    /// No validation is performed -- the caller is responsible for
    /// providing a code that exists in the transcript's `@Participants`
    /// header. Passing an undeclared code will cause validation errors
    /// downstream.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrows the speaker code string for comparison or display.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for SpeakerCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for SpeakerCode {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_cleaned_vs_raw() {
        let raw = ChatRawText::new("hello@c");
        let cleaned = ChatCleanedText::new("hello");

        assert_eq!(raw.as_str(), "hello@c");
        assert_eq!(cleaned.as_str(), "hello");
    }

    #[test]
    fn test_serde_transparency() {
        let text = ChatCleanedText::new("test");
        let json = serde_json::to_string(&text).unwrap();
        assert_eq!(json, "\"test\"");

        let decoded: ChatCleanedText = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_speaker_code() {
        let speaker = SpeakerCode::new("CHI");
        assert_eq!(speaker.as_str(), "CHI");
    }
}
