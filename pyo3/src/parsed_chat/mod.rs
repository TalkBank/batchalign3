//! Core `#[pymethods]` on `ParsedChat`: constructors, serialization, validation,
//! metadata extraction, and simple mutations.

mod cleanup;
mod fa;
mod morphosyntax;
mod speakers;
mod structure;
mod text;

use pyo3::prelude::*;
use talkbank_model::WriteChat;
use talkbank_model::model::Line;

use crate::ParsedChat;
use crate::build::build_chat_inner;
use crate::metadata::{extract_metadata_from_chat_file_pure, serialize_extracted_words};
use crate::parse::{
    errors_to_json, parse_alignment_domain, parse_lenient_with_warnings, parse_strict_pure,
    strip_timing_on_chat_file,
};

impl ParsedChat {
    /// Apply one fallible mutation transactionally.
    ///
    /// The PyO3 callback-facing mutation methods are performance-insensitive
    /// compared to the server/runtime paths, so it is acceptable to clone the
    /// AST here and only commit the mutated copy on success. That prevents a
    /// callback failure from leaving the long-lived `ParsedChat` handle in a
    /// partially mutated state.
    pub(crate) fn apply_transactional_mutation(
        &mut self,
        mutate: impl FnOnce(&mut talkbank_model::model::ChatFile) -> PyResult<()>,
    ) -> PyResult<()> {
        let mut staged = self.inner.clone();
        mutate(&mut staged)?;
        self.inner = staged;
        Ok(())
    }
}

#[pymethods]
impl ParsedChat {
    // --- Constructors ---

    /// Parse CHAT text strictly (tree-sitter, no error recovery).
    #[staticmethod]
    #[pyo3(name = "parse")]
    fn py_parse(py: Python<'_>, chat_text: talkbank_model::PythonChatText) -> PyResult<Self> {
        let text = chat_text.data;
        py.detach(|| parse_strict_pure(&text))
            .map(|cf| ParsedChat {
                inner: cf,
                warnings: vec![],
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Parse CHAT text leniently (tree-sitter with error recovery, tolerates dep tier errors).
    #[staticmethod]
    #[pyo3(name = "parse_lenient")]
    fn py_parse_lenient(
        py: Python<'_>,
        chat_text: talkbank_model::PythonChatText,
    ) -> PyResult<Self> {
        let text = chat_text.data;
        py.detach(|| parse_lenient_with_warnings(&text))
            .map(|(cf, warnings)| ParsedChat {
                inner: cf,
                warnings,
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Build a ParsedChat from a JSON transcript description (same as build_chat).
    #[staticmethod]
    #[pyo3(name = "build")]
    fn py_build(
        py: Python<'_>,
        transcript_json: talkbank_model::PythonTranscriptJson,
    ) -> PyResult<Self> {
        let result = py.detach(|| build_chat_inner(transcript_json));
        result
            .map(|cf| ParsedChat {
                inner: cf,
                warnings: vec![],
            })
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    // --- Serialization ---

    /// Serialize the AST back to CHAT text.
    #[pyo3(name = "serialize")]
    fn py_serialize(&self, py: Python<'_>) -> String {
        let inner = &self.inner;
        py.detach(|| inner.to_chat_string())
    }

    /// Replace the inner AST with another ParsedChat's AST.
    ///
    /// Used by the Python fallback path: serialize -> process_chat_text -> re-parse -> replace.
    #[pyo3(name = "replace_inner")]
    fn py_replace_inner(&mut self, other: &ParsedChat) {
        self.inner = other.inner.clone();
    }

    // --- Validation ---

    /// Run tier alignment checks (main<->mor, mor<->gra, main<->wor, main<->pho, main<->sin).
    ///
    /// Returns a list of human-readable error strings. Empty list means no issues.
    /// Respects ParseHealth flags -- tainted tiers from lenient parsing are skipped.
    #[pyo3(name = "validate")]
    fn py_validate(&self, py: Python<'_>) -> Vec<String> {
        let inner = &self.inner;
        py.detach(|| {
            inner
                .validate_alignments()
                .into_iter()
                .map(|e| format!("{}", e))
                .collect()
        })
    }

    /// Return parse warnings from lenient parsing as a JSON array string.
    ///
    /// Each element has: code, severity, line, column, message, suggestion.
    /// Returns "[]" if there are no warnings (or if parsed strictly).
    #[pyo3(name = "parse_warnings")]
    pub(crate) fn py_parse_warnings(&self) -> String {
        errors_to_json(&self.warnings)
    }

    /// Run tier alignment checks and return structured errors as a JSON array string.
    ///
    /// Same checks as `validate()`, but returns structured JSON instead of
    /// plain text strings. Each element has: code, severity, line, column,
    /// message, suggestion.
    #[pyo3(name = "validate_structured")]
    fn py_validate_structured(&self, py: Python<'_>) -> String {
        let inner = &self.inner;
        let errors = py.detach(|| inner.validate_alignments());
        errors_to_json(&errors)
    }

    /// Run full semantic validation -- equivalent to `chatter validate`.
    ///
    /// Performs the complete validation suite: header structure, per-utterance
    /// checks, cross-utterance patterns, E362 bullet monotonicity, E701/E704
    /// temporal constraints. E531 (media filename match) is skipped because
    /// we don't know the filename at this point.
    ///
    /// Returns structured JSON (same format as validate_structured). Empty
    /// array means the file is valid. Should be called before serializing.
    #[pyo3(name = "validate_chat_structured")]
    fn py_validate_chat_structured(&self, py: Python<'_>) -> String {
        use talkbank_model::ErrorCollector;

        let inner = &self.inner;
        let errors = ErrorCollector::new();
        py.detach(|| {
            inner.validate(&errors, None);
        });
        errors_to_json(&errors.into_vec())
    }

    // --- Mutations ---

    /// Insert a `@Comment:` header line before the first utterance.
    ///
    /// CHAT comments are metadata lines of the form `@Comment:\t<text>` that
    /// appear in the header section (between `@Begin` and the first `*SPK:` line).
    /// They carry free-text annotations about the transcript (e.g., recording
    /// conditions, transcriber notes, pipeline warnings).
    ///
    /// Placement: the comment is inserted immediately before the first
    /// `Line::Utterance` in the AST. If the file has no utterances, it is
    /// appended at the end (before `@End`).
    ///
    /// The `comment` argument is the raw text content -- do NOT include the
    /// `@Comment:` prefix or leading tab; those are added by serialization.
    #[pyo3(name = "add_comment")]
    fn py_add_comment(&mut self, py: Python<'_>, comment: &str) {
        use talkbank_model::model::{BulletContent, Header};

        let comment_line = py.detach(|| {
            Line::header(Header::Comment {
                content: BulletContent::from_text(comment),
            })
        });

        // Find the first utterance and insert before it
        let pos = self
            .inner
            .lines
            .iter()
            .position(|l| matches!(l, Line::Utterance(_)))
            .unwrap_or(self.inner.lines.len());
        self.inner.lines.0.insert(pos, comment_line);
    }

    /// Strip all timing bullets, word-level timestamps, and %wor tiers.
    #[pyo3(name = "strip_timing")]
    fn py_strip_timing(&mut self, py: Python<'_>) {
        let inner = &mut self.inner;
        py.detach(|| strip_timing_on_chat_file(inner));
    }

    /// Remove all %mor and %gra dependent tiers from every utterance.
    ///
    /// Called before morphosyntax processing so that
    /// `collect_morphosyntax_payloads` sees all utterances as needing
    /// reprocessing (it skips utterances that already have %mor).
    #[pyo3(name = "clear_morphosyntax")]
    fn py_clear_morphosyntax(&mut self, py: Python<'_>) {
        let inner = &mut self.inner;
        py.detach(|| {
            for line in inner.lines.iter_mut() {
                if let Line::Utterance(utt) = line {
                    utt.dependent_tiers.retain(|t| {
                        !matches!(
                            t,
                            talkbank_model::model::DependentTier::Mor(_)
                                | talkbank_model::model::DependentTier::Gra(_)
                        )
                    });
                }
            }
        });
    }

    /// Extract NLP-ready words from the parsed CHAT file.
    ///
    /// Walks every utterance and collects the words relevant to the given
    /// alignment `domain`. The domain controls which words are extracted and
    /// how compound/group structures are flattened:
    ///
    /// - `"mor"` -- words aligned with the %mor tier (most common for NLP).
    ///   Includes tag-marker separators (comma `,`, tag `\u{201E}`, vocative `\u{2021}`)
    ///   because those have corresponding %mor entries.
    /// - `"wor"` -- words aligned with the %wor tier (forced alignment).
    /// - `"pho"` -- words aligned with the %pho tier (phonological).
    /// - `"sin"` -- words aligned with the %sin tier (sincerity/prosody).
    ///
    /// Raises `ValueError` if the domain string is invalid.
    #[pyo3(name = "extract_nlp_words")]
    fn py_extract_nlp_words(
        &self,
        py: Python<'_>,
        domain: talkbank_model::PythonAlignmentDomain,
    ) -> PyResult<String> {
        let domain_name = domain.data;
        let domain = parse_alignment_domain(&domain_name)?;
        let inner = &self.inner;
        Ok(py.detach(|| {
            let extracted = crate::extract::extract_words(inner, domain);
            serialize_extracted_words(&extracted)
        }))
    }

    /// Extract file-level metadata from CHAT headers.
    ///
    /// Scans the `@Languages` and `@Media` headers and returns a JSON object
    /// with the following keys:
    ///
    /// ```json
    /// {
    ///   "langs": ["eng", "spa"],
    ///   "media_name": "interview01",
    ///   "media_type": "audio"
    /// }
    /// ```
    ///
    /// - `langs` -- array of ISO 639-3 language code strings from `@Languages`.
    ///   Empty array if no `@Languages` header is present.
    /// - `media_name` -- filename (without extension) from `@Media`, or `null`
    ///   if no `@Media` header exists.
    /// - `media_type` -- `"audio"` or `"video"` from `@Media`, or `null` if
    ///   no `@Media` header exists.
    #[pyo3(name = "extract_metadata")]
    fn py_extract_metadata(&self, py: Python<'_>) -> PyResult<String> {
        let inner = &self.inner;
        let result = py.detach(|| extract_metadata_from_chat_file_pure(inner));
        result.map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Check whether the file has `@Options: CA` or `@Options: CA-Unicode`.
    ///
    /// Uses `headers()` iterator -- `@Options` is always before the first
    /// utterance so we never need to scan the full line list.
    #[pyo3(name = "is_ca")]
    fn py_is_ca(&self) -> bool {
        self.inner.options.iter().any(|f| f.enables_ca_mode())
    }

    /// Check whether the file has `@Options: NoAlign`.
    #[pyo3(name = "is_no_align")]
    fn py_is_no_align(&self) -> bool {
        self.inner.options.iter().any(|f| f.skips_alignment())
    }

    /// Return the language codes from the `@Languages` header.
    ///
    /// Falls back to `["eng"]` when no `@Languages` header is present,
    /// matching the Python `extract_langs()` helper's default. All codes
    /// are returned as ISO 639-3 strings (e.g. `"eng"`, `"spa"`).
    #[pyo3(name = "extract_languages")]
    fn py_extract_languages(&self, py: Python<'_>) -> Vec<String> {
        let inner = &self.inner;
        py.detach(|| {
            if inner.languages.is_empty() {
                vec!["eng".to_string()]
            } else {
                inner
                    .languages
                    .iter()
                    .map(|c| c.as_str().to_string())
                    .collect()
            }
        })
    }

    /// Add user-defined dependent tiers to specific utterances.
    #[pyo3(name = "add_dependent_tiers")]
    fn py_add_dependent_tiers(
        &mut self,
        py: Python<'_>,
        tiers_json: talkbank_model::model::Provenance<talkbank_model::model::AsrWordsJson, String>,
    ) -> PyResult<()> {
        let t_json = tiers_json.data;
        let inner = &mut self.inner;
        py.detach(|| {
            crate::tier_ops::add_dependent_tiers_inner(inner, &t_json).map_err(|e| e.to_string())
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
