//! Parse and validate raw Stanza `doc.to_dict()` output into typed UD structures.
//!
//! After the CHAT divorce, Python workers return Stanza's native `to_dict()` output
//! (a `Vec<Vec<serde_json::Value>>` — one sentence of word dicts per utterance).
//! This module parses that raw output into the typed [`UdWord`]/[`UdSentence`]/[`UdResponse`]
//! structures, applying the same validation that previously lived in Python:
//!
//! - Tuple-to-list `id` conversion (Stanza emits `(1,)` tuples for word ids)
//! - Default lemma to surface form when lemma is empty
//! - Bogus lemma detection (pure-punctuation lemma for letter-containing word)
//! - Pad deprel sanitization (`<pad>` → `dep`)

use crate::nlp::{UdId, UdResponse, UdSentence, UdWord};

/// Parse raw Stanza `doc.to_dict()` output into a [`UdResponse`].
///
/// Stanza's `to_dict()` returns a list of sentences, each a list of word dicts.
/// This function deserializes each word dict into a [`UdWord`], applying validation.
///
/// # Errors
///
/// Returns `Err` if a word dict cannot be deserialized into [`UdWord`].
pub fn parse_raw_stanza_output(
    raw_sentences: &[serde_json::Value],
) -> Result<UdResponse, StanzaParseError> {
    let mut sentences = Vec::with_capacity(raw_sentences.len());

    for (sent_idx, sent_value) in raw_sentences.iter().enumerate() {
        let word_dicts = sent_value.as_array().ok_or(StanzaParseError::NotAnArray {
            sentence_idx: sent_idx,
        })?;

        let mut words = Vec::with_capacity(word_dicts.len());
        for (word_idx, raw_word) in word_dicts.iter().enumerate() {
            let mut word: UdWord = serde_json::from_value(normalize_word_dict(raw_word.clone()))
                .map_err(|e| StanzaParseError::WordParse {
                    sentence_idx: sent_idx,
                    word_idx,
                    source: e,
                })?;

            validate_and_clean(&mut word);
            words.push(word);
        }

        sentences.push(UdSentence { words });
    }

    Ok(UdResponse { sentences })
}

/// Normalize a raw Stanza word dict before deserialization.
///
/// Handles Stanza quirks:
/// - `id` as tuple `[1]` or `[1, 2]` (Stanza emits tuples for word ids)
/// - Missing fields that should default
fn normalize_word_dict(mut value: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = value.as_object_mut() {
        // Stanza emits id as a tuple — e.g. (1,) becomes [1], (1,2) becomes [1,2]
        // Our UdId deserialization handles [start, end] as Range, but single-element
        // arrays [n] need to be unwrapped to just n for Single variant
        if let Some(id_val) = obj.get("id")
            && let Some(arr) = id_val.as_array()
            && arr.len() == 1
            && let Some(n) = arr[0].as_u64()
        {
            // Single-element tuple: unwrap to scalar.
            // Multi-element tuples [start, end] already match UdId::Range.
            obj.insert("id".to_string(), serde_json::json!(n));
        }

        // Default lemma to text if empty, missing, or null.
        // Stanza can emit any of: no key, "", or null when the lemmatizer
        // fails silently for a token.
        let lemma_empty = obj.get("lemma").is_none_or(|v| v.is_null() || v.as_str().is_some_and(|s| s.is_empty()));
        let is_range = obj
            .get("id")
            .is_some_and(|v| v.as_array().is_some_and(|a| a.len() > 1));

        if lemma_empty
            && !is_range
            && let Some(text) = obj.get("text").and_then(|v| v.as_str())
        {
            obj.insert("lemma".to_string(), serde_json::json!(text));
        }

        // Normalize null string fields to sensible defaults.
        // Stanza can emit null for any field when a processor fails silently.
        for (field, default) in [("upos", "X"), ("deprel", "dep"), ("feats", "")] {
            if obj.get(field).is_some_and(|v| v.is_null()) {
                obj.insert(field.to_string(), serde_json::json!(default));
            }
        }
    }
    value
}

/// Apply post-parse validation and cleaning to a [`UdWord`].
///
/// - Sanitize `<pad>`-style deprels to `dep`
/// - Fix bogus lemmas (pure-punctuation lemma for letter-containing word)
pub fn validate_and_clean(word: &mut UdWord) {
    // Sanitize pad deprels
    if word.deprel.starts_with('<') && word.deprel.ends_with('>') {
        tracing::warn!(
            deprel = %word.deprel,
            text = %word.text,
            "Stanza emitted pad deprel — replacing with 'dep'"
        );
        word.deprel = "dep".to_string();
    }

    // Fix bogus lemmas
    if !matches!(word.id, UdId::Range(_, _)) && is_bogus_lemma(&word.text, &word.lemma) {
        tracing::warn!(
            lemma = %word.lemma,
            text = %word.text,
            "Stanza returned bogus lemma — falling back to surface form"
        );
        word.lemma = word.text.clone();
    }
}

/// Detect when Stanza returns a pure-punctuation/symbol lemma for a word with letters.
///
/// Returns `true` if:
/// - `text` contains at least one Unicode letter
/// - `lemma` contains only punctuation/symbol characters (no letters, no digits)
/// - `text != lemma` and `lemma` is non-empty
///
/// Mirrors the Python logic:
/// ```python
/// text_has_letters = any(unicodedata.category(c).startswith("L") for c in text)
/// lemma_all_punct = all(unicodedata.category(c).startswith(("P", "S")) for c in lemma)
/// ```
pub fn is_bogus_lemma(text: &str, lemma: &str) -> bool {
    if text == lemma || lemma.is_empty() {
        return false;
    }

    let text_has_letters = text.chars().any(|c| c.is_alphabetic());
    // A char is "punctuation or symbol" if it's not a letter, digit, whitespace, or control
    let lemma_all_punct = lemma
        .chars()
        .all(|c| !c.is_alphanumeric() && !c.is_whitespace() && !c.is_control());

    text_has_letters && lemma_all_punct
}

/// Diagnostic for a single problematic word in Stanza output.
#[derive(Debug, Clone)]
pub struct StanzaWordDiagnostic {
    /// Sentence index in the raw output.
    pub sentence_idx: usize,
    /// Word index within the sentence.
    pub word_idx: usize,
    /// The field that has a problem.
    pub field: String,
    /// Human-readable description of the issue.
    pub issue: String,
}

impl std::fmt::Display for StanzaWordDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sentence {} word {}: field '{}' — {}",
            self.sentence_idx, self.word_idx, self.field, self.issue
        )
    }
}

/// Diagnose problems in raw Stanza output without attempting full deserialization.
///
/// Scans each word dict for known issues: missing required fields, null values,
/// `<pad>` sentinels, type mismatches. Returns a list of diagnostics — one per
/// problem found. An empty list means the output looks structurally valid.
///
/// This is called from the error path in `morphosyntax/worker.rs` to provide
/// actionable diagnostics instead of opaque serde errors.
pub fn diagnose_parse_failure(raw_sentences: &[serde_json::Value]) -> Vec<StanzaWordDiagnostic> {
    let required_string_fields = ["text", "lemma", "upos", "deprel"];
    let mut diagnostics = Vec::new();

    for (si, sent_value) in raw_sentences.iter().enumerate() {
        let Some(words) = sent_value.as_array() else {
            diagnostics.push(StanzaWordDiagnostic {
                sentence_idx: si,
                word_idx: 0,
                field: "sentence".into(),
                issue: "not a JSON array".into(),
            });
            continue;
        };

        for (wi, word) in words.iter().enumerate() {
            let Some(obj) = word.as_object() else {
                diagnostics.push(StanzaWordDiagnostic {
                    sentence_idx: si,
                    word_idx: wi,
                    field: "word".into(),
                    issue: "not a JSON object".into(),
                });
                continue;
            };

            let id_val = obj.get("id");
            let is_range = id_val
                .and_then(|v| v.as_array())
                .is_some_and(|a| a.len() > 1);

            // Check required string fields
            for field in &required_string_fields {
                // Range (MWT) tokens are expected to have empty lemma
                if *field == "lemma" && is_range {
                    continue;
                }

                match obj.get(*field) {
                    None => {
                        let keys: Vec<&String> = obj.keys().collect();
                        diagnostics.push(StanzaWordDiagnostic {
                            sentence_idx: si,
                            word_idx: wi,
                            field: (*field).into(),
                            issue: format!(
                                "field absent (keys present: {keys:?}). \
                                 Stanza's processor likely failed silently for this token."
                            ),
                        });
                    }
                    Some(v) if v.is_null() => {
                        let text = obj.get("text").and_then(|t| t.as_str()).unwrap_or("?");
                        diagnostics.push(StanzaWordDiagnostic {
                            sentence_idx: si,
                            word_idx: wi,
                            field: (*field).into(),
                            issue: format!(
                                "value is null for word '{text}'. \
                                 Stanza's processor likely failed silently."
                            ),
                        });
                    }
                    Some(v) if !v.is_string() => {
                        diagnostics.push(StanzaWordDiagnostic {
                            sentence_idx: si,
                            word_idx: wi,
                            field: (*field).into(),
                            issue: format!(
                                "expected string, got {}",
                                match v {
                                    serde_json::Value::Number(_) => "number",
                                    serde_json::Value::Bool(_) => "bool",
                                    serde_json::Value::Array(_) => "array",
                                    serde_json::Value::Object(_) => "object",
                                    _ => "unknown",
                                }
                            ),
                        });
                    }
                    _ => {} // Field present, not null, is string — OK
                }
            }

            // Check for <pad> sentinels
            if let Some(deprel) = obj.get("deprel").and_then(|v| v.as_str()) {
                if deprel.starts_with('<') && deprel.ends_with('>') {
                    diagnostics.push(StanzaWordDiagnostic {
                        sentence_idx: si,
                        word_idx: wi,
                        field: "deprel".into(),
                        issue: format!("pad sentinel value '{deprel}'"),
                    });
                }
            }

            // Check id field
            match id_val {
                None => {
                    diagnostics.push(StanzaWordDiagnostic {
                        sentence_idx: si,
                        word_idx: wi,
                        field: "id".into(),
                        issue: "field absent".into(),
                    });
                }
                Some(v) if !(v.is_number() || v.is_array()) => {
                    diagnostics.push(StanzaWordDiagnostic {
                        sentence_idx: si,
                        word_idx: wi,
                        field: "id".into(),
                        issue: format!("expected number or array, got {v}"),
                    });
                }
                _ => {}
            }
        }
    }

    diagnostics
}

/// Errors from parsing raw Stanza output.
#[derive(Debug, thiserror::Error)]
pub enum StanzaParseError {
    /// A sentence value was not a JSON array.
    #[error("sentence {sentence_idx} is not a JSON array")]
    NotAnArray {
        /// Index of the sentence in the raw output.
        sentence_idx: usize,
    },
    /// A word dict could not be parsed into [`UdWord`].
    #[error("sentence {sentence_idx} word {word_idx}: {source}")]
    WordParse {
        /// Sentence index.
        sentence_idx: usize,
        /// Word index within the sentence.
        word_idx: usize,
        /// Underlying deserialization error.
        source: serde_json::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_is_bogus_lemma_basic() {
        assert!(is_bogus_lemma("hello", "."));
        assert!(is_bogus_lemma("world", ","));
        assert!(is_bogus_lemma("cat", "–")); // en-dash is punctuation
    }

    #[test]
    fn test_is_bogus_lemma_same_text() {
        assert!(!is_bogus_lemma("hello", "hello"));
    }

    #[test]
    fn test_is_bogus_lemma_empty() {
        assert!(!is_bogus_lemma("hello", ""));
    }

    #[test]
    fn test_is_bogus_lemma_punct_text() {
        // If text itself is punctuation, the lemma being punctuation is fine
        assert!(!is_bogus_lemma(".", "."));
        assert!(!is_bogus_lemma(",", "--"));
    }

    #[test]
    fn test_is_bogus_lemma_normal() {
        assert!(!is_bogus_lemma("running", "run"));
        assert!(!is_bogus_lemma("cats", "cat"));
    }

    #[test]
    fn test_parse_raw_stanza_single_sentence() {
        let raw = vec![json!([
            {
                "id": [1],
                "text": "hello",
                "lemma": "hello",
                "upos": "INTJ",
                "head": 0,
                "deprel": "root"
            }
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        assert_eq!(resp.sentences.len(), 1);
        assert_eq!(resp.sentences[0].words.len(), 1);
        assert_eq!(resp.sentences[0].words[0].text, "hello");
        assert_eq!(resp.sentences[0].words[0].id, UdId::Single(1));
    }

    #[test]
    fn test_parse_raw_stanza_mwt() {
        let raw = vec![json!([
            {
                "id": [1, 2],
                "text": "du",
                "lemma": "",
                "upos": "X",
                "head": 0,
                "deprel": "root"
            },
            {
                "id": [1],
                "text": "de",
                "lemma": "de",
                "upos": "ADP",
                "head": 3,
                "deprel": "case"
            },
            {
                "id": [2],
                "text": "le",
                "lemma": "le",
                "upos": "DET",
                "head": 3,
                "deprel": "det"
            }
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        assert_eq!(resp.sentences[0].words.len(), 3);
        assert_eq!(resp.sentences[0].words[0].id, UdId::Range(1, 2));
        // MWT range token should keep empty lemma (not default to text)
        assert!(resp.sentences[0].words[0].lemma.is_empty());
    }

    #[test]
    fn test_parse_raw_stanza_pad_deprel() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "hello",
                "lemma": "hello",
                "upos": "INTJ",
                "head": 0,
                "deprel": "<pad>"
            }
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        assert_eq!(resp.sentences[0].words[0].deprel, "dep");
    }

    #[test]
    fn test_parse_raw_stanza_bogus_lemma() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "hello",
                "lemma": ".",
                "upos": "INTJ",
                "head": 0,
                "deprel": "root"
            }
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        // Bogus lemma should be replaced with surface form
        assert_eq!(resp.sentences[0].words[0].lemma, "hello");
    }

    #[test]
    fn test_parse_raw_stanza_default_lemma() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "hello",
                "upos": "INTJ",
                "head": 0,
                "deprel": "root"
            }
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        // Missing lemma should default to text
        assert_eq!(resp.sentences[0].words[0].lemma, "hello");
    }

    /// Regression: Stanza can emit `"lemma": null` (JSON null) instead of
    /// omitting the field entirely. The normalization must treat null the
    /// same as absent and default to the surface form.
    ///
    /// This was the root cause of the bilbo morphotag failure (2026-03-27):
    /// `Failed to parse raw Stanza output for item 4: sentence 0 word 3:
    /// missing field 'lemma'`.
    #[test]
    fn test_parse_raw_stanza_null_lemma_defaults_to_text() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "au",
                "lemma": null,
                "upos": "ADP",
                "head": 0,
                "deprel": "root"
            }
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        // null lemma should default to surface form, not fail
        assert_eq!(resp.sentences[0].words[0].lemma, "au");
    }

    /// Regression: Stanza can also emit `"upos": null` in rare cases.
    /// The normalization must handle null for all string fields.
    #[test]
    fn test_parse_raw_stanza_null_upos_does_not_crash() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "xyz",
                "lemma": "xyz",
                "upos": null,
                "head": 0,
                "deprel": "root"
            }
        ])];

        // Should not crash — null upos should default to something parseable
        let result = parse_raw_stanza_output(&raw);
        assert!(result.is_ok(), "null upos should not crash: {result:?}");
    }

    #[test]
    fn test_parse_raw_stanza_not_array() {
        let raw = vec![json!("not an array")];
        let err = parse_raw_stanza_output(&raw).unwrap_err();
        assert!(err.to_string().contains("not a JSON array"));
    }

    /// Regression: 6-word Cantonese sentence must produce 6 UD words AND 6 MOR items.
    ///
    /// Source: MOST corpus 40415b.cha, utterance with retrace.
    /// Bug: morphotag --retokenize produced "MOR item count (5) does not match
    /// alignable word count (6)".
    #[test]
    fn test_cantonese_6_words_produces_6_mors() {
        use crate::nlp::{MappingContext, map_ud_sentence};

        let raw = vec![json!([
            {"id": 1, "text": "呢", "lemma": "呢", "upos": "PART", "head": 2, "deprel": "case"},
            {"id": 2, "text": "度", "lemma": "度", "upos": "NUM", "head": 5, "deprel": "nmod"},
            {"id": 3, "text": "食飯", "lemma": "食飯", "upos": "VERB", "head": 4, "deprel": "compound"},
            {"id": 4, "text": "啦", "lemma": "啦", "upos": "NOUN", "head": 5, "deprel": "nmod"},
            {"id": 5, "text": "飯", "lemma": "飯", "upos": "NOUN", "head": 0, "deprel": "root"},
            {"id": 6, "text": "啦", "lemma": "啦", "upos": "NOUN", "head": 5, "deprel": "discourse:sp"}
        ])];

        let resp = parse_raw_stanza_output(&raw).unwrap();
        assert_eq!(resp.sentences.len(), 1);
        assert_eq!(resp.sentences[0].words.len(), 6, "Should parse 6 UD words");

        let ctx = MappingContext {
            lang: talkbank_model::model::LanguageCode::new("yue"),
        };
        let (mors, gras) = map_ud_sentence(&resp.sentences[0], &ctx).unwrap();
        assert_eq!(mors.len(), 6, "Should produce 6 MOR items from 6 UD words");

        // GRA has 7 entries for 6 MOR words — the 7th is the terminator PUNCT
        // relation (e.g., 7|5|PUNCT for "."). This is correct CHAT behavior:
        // %gra includes the terminator, %mor does not. The inject path handles
        // this correctly — the terminator GRA is stored in the GraTier and the
        // MOR count check only looks at MOR items vs alignable words.
        assert_eq!(
            gras.len(),
            7,
            "GRA should have 6 word relations + 1 terminator PUNCT"
        );
    }

    // --- diagnose_parse_failure tests ---

    #[test]
    fn test_diagnose_detects_missing_lemma() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "au",
                "upos": "ADP",
                "head": 0,
                "deprel": "root"
            }
        ])];
        let diags = diagnose_parse_failure(&raw);
        assert!(!diags.is_empty(), "Should detect missing lemma");
        assert!(diags.iter().any(|d| d.field == "lemma" && d.issue.contains("absent")));
    }

    #[test]
    fn test_diagnose_detects_null_lemma() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "au",
                "lemma": null,
                "upos": "ADP",
                "head": 0,
                "deprel": "root"
            }
        ])];
        let diags = diagnose_parse_failure(&raw);
        assert!(!diags.is_empty(), "Should detect null lemma");
        assert!(diags.iter().any(|d| d.field == "lemma" && d.issue.contains("null")));
    }

    #[test]
    fn test_diagnose_detects_pad_deprel() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "hello",
                "lemma": "hello",
                "upos": "INTJ",
                "head": 0,
                "deprel": "<pad>"
            }
        ])];
        let diags = diagnose_parse_failure(&raw);
        assert!(diags.iter().any(|d| d.field == "deprel" && d.issue.contains("pad")));
    }

    #[test]
    fn test_diagnose_clean_output_produces_no_diagnostics() {
        let raw = vec![json!([
            {
                "id": 1,
                "text": "hello",
                "lemma": "hello",
                "upos": "INTJ",
                "head": 0,
                "deprel": "root"
            }
        ])];
        let diags = diagnose_parse_failure(&raw);
        assert!(diags.is_empty(), "Clean output should produce no diagnostics: {diags:?}");
    }

    #[test]
    fn test_diagnose_skips_lemma_for_range_tokens() {
        let raw = vec![json!([
            {
                "id": [1, 2],
                "text": "du",
                "upos": "X",
                "head": 0,
                "deprel": "root"
            }
        ])];
        let diags = diagnose_parse_failure(&raw);
        // Range tokens should NOT be flagged for missing lemma
        assert!(
            !diags.iter().any(|d| d.field == "lemma"),
            "Range token should not be flagged for missing lemma: {diags:?}"
        );
    }
}
