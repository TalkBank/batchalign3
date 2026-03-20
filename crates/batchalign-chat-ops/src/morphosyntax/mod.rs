//! Morphosyntax cache key computation, payload extraction, and injection.
//!
//! This module contains the pure-Rust logic for morphosyntax cache operations:
//! - Cache key computation (SHA-256 of words|lang|mwt)
//! - Payload extraction (walk utterances, collect words, compute keys)
//! - Cache injection (inject cached MorTier/GraTier into utterances)
//! - String extraction (extract final MorTier/GraTier JSON from utterances)
//! - Result injection (inject UD NLP results back into utterances)

mod cache;
mod inject;
mod payloads;
pub mod preprocess;
pub mod stanza_raw;
#[cfg(test)]
mod tests;

pub use cache::*;
pub use inject::*;
pub use payloads::*;

use crate::extract;

/// Controls whether the morphosyntax pipeline retokenizes using Stanza's
/// neural tokenizer or preserves original CHAT word boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TokenizationMode {
    /// Preserve original CHAT tokenization. Stanza annotates existing
    /// word boundaries without splitting or merging.
    Preserve,
    /// Allow Stanza retokenization: compounds may be split (don't -> do + n't),
    /// or fragments merged. Main-tier words are updated to match.
    StanzaRetokenize,
}

impl From<bool> for TokenizationMode {
    fn from(retokenize: bool) -> Self {
        if retokenize {
            Self::StanzaRetokenize
        } else {
            Self::Preserve
        }
    }
}

/// Controls whether utterances marked with a non-primary language are processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MultilingualPolicy {
    /// Process all utterances regardless of @s language marking.
    ProcessAll,
    /// Skip utterances whose @s language marker differs from the
    /// primary file language. For monolingual-optimized pipelines.
    SkipNonPrimary,
}

impl MultilingualPolicy {
    /// Convert from the legacy boolean flag used at CLI and PyO3 boundaries.
    ///
    /// - `true`  -> `SkipNonPrimary`
    /// - `false` -> `ProcessAll`
    pub fn from_skip_flag(skip: bool) -> Self {
        if skip {
            Self::SkipNonPrimary
        } else {
            Self::ProcessAll
        }
    }

    /// Whether non-primary-language utterances should be skipped.
    pub fn should_skip_non_primary(self) -> bool {
        matches!(self, Self::SkipNonPrimary)
    }
}

// ---------------------------------------------------------------------------
// Serde types for morphosyntax operations
// ---------------------------------------------------------------------------

/// Batch item for NLP processing.
///
/// `special_forms` carries typed enums internally (for Rust-side injection)
/// but serializes to `(Option<String>, Option<String>)` for the Python V2
/// wire protocol, where Pydantic expects `list[list[str | None]]`.
#[derive(Clone, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct MorphosyntaxBatchItem {
    /// Word texts for NLP processing.
    pub words: Vec<String>,
    /// Utterance terminator string.
    pub terminator: String,
    /// Special form and language per word: (form_type, resolved_language).
    ///
    /// Serializes form_type via `FormType::write_chat()` and language via
    /// `LanguageResolution::as_language_code()` so the JSON wire format is
    /// `[string|null, string|null]` — matching the Python Pydantic model.
    #[serde(serialize_with = "serialize_special_forms")]
    #[schemars(with = "Vec<(Option<String>, Option<String>)>")]
    pub special_forms: Vec<(
        Option<talkbank_model::model::FormType>,
        Option<talkbank_model::validation::LanguageResolution>,
    )>,
    /// Language code for this utterance (ISO 639-3).
    #[schemars(with = "String")]
    pub lang: talkbank_model::model::LanguageCode,
}

/// Serialize `special_forms` as `Vec<(Option<String>, Option<String>)>` for
/// the Python V2 wire protocol. Rust-side code uses the typed enums directly
/// via field access; this flattening only affects JSON serialization.
fn serialize_special_forms<S: serde::Serializer>(
    forms: &[(
        Option<talkbank_model::model::FormType>,
        Option<talkbank_model::validation::LanguageResolution>,
    )],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;
    use talkbank_model::WriteChat;

    let mut seq = serializer.serialize_seq(Some(forms.len()))?;
    for (form_type, lang_res) in forms {
        let ft_str: Option<String> = form_type.as_ref().map(|ft| {
            let mut buf = String::new();
            ft.write_chat(&mut buf);
            buf
        });
        let lang_str: Option<String> = lang_res.as_ref().and_then(|lr| {
            lr.languages().first().map(|lc| lc.to_string())
        });
        seq.serialize_element(&(ft_str, lang_str))?;
    }
    seq.end()
}

/// A collected batch item with its position in the ChatFile, for injection.
pub type BatchItemWithPosition = (
    usize,                       // line_idx in ChatFile.lines
    usize,                       // utt_ordinal (0-based)
    MorphosyntaxBatchItem,       // payload for NLP
    Vec<extract::ExtractedWord>, // raw extracted words (for retokenize)
);

/// JSON payload for `extract_morphosyntax_payloads` return value.
#[derive(serde::Serialize)]
pub struct MorphosyntaxPayloadJson {
    /// Index into `ChatFile.lines`.
    pub line_idx: usize,
    /// NLP-ready words extracted from the utterance.
    pub words: Vec<String>,
    /// ISO 639-3 language code.
    pub lang: String,
    /// SHA-256 cache key.
    pub key: String,
}

/// JSON payload for `inject_morphosyntax_from_cache` input.
#[derive(serde::Deserialize)]
pub struct CachedMorphosyntaxEntry {
    /// Index into `ChatFile.lines`.
    pub line_idx: usize,
    /// Cached `%mor` tier content string.
    pub mor: String,
    /// Cached `%gra` tier content string.
    pub gra: String,
}

/// JSON payload for `extract_morphosyntax_strings` return value.
#[derive(serde::Serialize)]
pub struct MorphosyntaxStringsEntry {
    /// Index into `ChatFile.lines`.
    pub line_idx: usize,
    /// Serialized `%mor` tier content string.
    pub mor: String,
    /// Serialized `%gra` tier content string.
    pub gra: String,
}

/// Validation warning for a single utterance.
#[derive(Debug)]
pub struct AlignmentWarning {
    /// Zero-based line index in the ChatFile.
    pub line_idx: usize,
    /// Main tier word count (alignable words in the Mor domain).
    pub main_count: usize,
    /// %mor item count.
    pub mor_count: usize,
}

impl std::fmt::Display for AlignmentWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "line {}: main tier has {} alignable words but %mor has {} items",
            self.line_idx, self.main_count, self.mor_count,
        )
    }
}
