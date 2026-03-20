//! Document structure extraction for the compatibility shim.
//!
//! Provides `extract_document_structure()` which returns a JSON representation
//! of the document with per-word morphology and grammatical relation data.
//! Used by the Python `batchalign.compat` module to support BA2-style
//! `doc[0][0].morphology` subscript access.

use pyo3::prelude::*;
use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{ChatFile, DependentTier, Line};

use crate::ParsedChat;

// ---------------------------------------------------------------------------
// Serde output types
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct UtteranceJson {
    speaker: String,
    utterance_index: usize,
    words: Vec<WordJson>,
}

#[derive(serde::Serialize)]
struct WordJson {
    text: String,
    word_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    mor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pos: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lemma: Option<String>,
    gra: Vec<GraJson>,
}

#[derive(serde::Serialize)]
struct GraJson {
    index: usize,
    head: usize,
    relation: String,
}

// ---------------------------------------------------------------------------
// PyO3 method
// ---------------------------------------------------------------------------

#[pymethods]
impl ParsedChat {
    /// Extract document structure as JSON for compatibility shim subscript access.
    ///
    /// Returns a JSON array of utterances. Each utterance contains its speaker
    /// code, utterance index, and an array of words. Each word includes the
    /// cleaned text, word index, and (when present) per-word morphology from
    /// `%mor` (full notation, POS, lemma) and grammatical relations from `%gra`.
    ///
    /// # Return schema (JSON string)
    ///
    /// ```json
    /// [
    ///   {
    ///     "speaker": "CHI",
    ///     "utterance_index": 0,
    ///     "words": [
    ///       {
    ///         "text": "hello",
    ///         "word_index": 0,
    ///         "mor": "n|hello",
    ///         "pos": "n",
    ///         "lemma": "hello",
    ///         "gra": [{"index": 1, "head": 0, "relation": "ROOT"}]
    ///       }
    ///     ]
    ///   }
    /// ]
    /// ```
    ///
    /// Words without `%mor` have `mor`, `pos`, and `lemma` set to `null`.
    /// Words without `%gra` have an empty `gra` array.
    #[pyo3(name = "extract_document_structure")]
    fn py_extract_document_structure(&self, py: Python<'_>) -> PyResult<String> {
        let inner = &self.inner;
        let result = py.detach(|| extract_structure(inner));
        result.map_err(pyo3::exceptions::PyValueError::new_err)
    }
}

// ---------------------------------------------------------------------------
// Inner logic (runs without GIL)
// ---------------------------------------------------------------------------

/// Walk the ChatFile AST and produce a JSON document structure with per-word
/// morphology and grammatical relations.
fn extract_structure(chat_file: &ChatFile) -> Result<String, String> {
    let mut utterances = Vec::new();
    let mut utt_idx = 0usize;

    for line in &chat_file.lines {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let speaker = utt.main.speaker.as_str().to_string();

        // Extract words using the same domain-aware logic as extract_words(Mor).
        let mut extracted_words = Vec::new();
        crate::extract::collect_utterance_content(
            &utt.main.content.content,
            TierDomain::Mor,
            &mut extracted_words,
        );

        // Find %mor and %gra dependent tiers.
        let mor_tier = utt.dependent_tiers.iter().find_map(|dt| match dt {
            DependentTier::Mor(m) => Some(m),
            _ => None,
        });
        let gra_tier = utt.dependent_tiers.iter().find_map(|dt| match dt {
            DependentTier::Gra(g) => Some(g),
            _ => None,
        });

        // Build per-word JSON objects, zipping words with %mor items.
        let mut words = Vec::with_capacity(extracted_words.len());
        let mut gra_chunk_offset = 0usize;

        for (w_idx, extracted) in extracted_words.iter().enumerate() {
            let (mor_str, pos_str, lemma_str, n_chunks) = if let Some(mor) = mor_tier {
                if let Some(mor_item) = mor.items.get(w_idx) {
                    let full = mor_item.to_chat_string();
                    let pos = mor_item.main.pos.as_str().to_string();
                    let lemma = mor_item.main.lemma.as_str().to_string();
                    let chunks = 1 + mor_item.post_clitics.len();
                    (Some(full), Some(pos), Some(lemma), chunks)
                } else {
                    (None, None, None, 1)
                }
            } else {
                (None, None, None, 1)
            };

            // Collect %gra relations for this word's chunks (1 main + N clitics).
            let mut gra_entries = Vec::new();
            if let Some(gra) = gra_tier {
                for chunk_i in 0..n_chunks {
                    let target_index = gra_chunk_offset + chunk_i + 1; // %gra is 1-indexed
                    if let Some(rel) = gra.relations.iter().find(|r| r.index == target_index) {
                        gra_entries.push(GraJson {
                            index: rel.index,
                            head: rel.head,
                            relation: rel.relation.to_string(),
                        });
                    }
                }
            }
            gra_chunk_offset += n_chunks;

            words.push(WordJson {
                text: extracted.text.as_str().to_string(),
                word_index: w_idx,
                mor: mor_str,
                pos: pos_str,
                lemma: lemma_str,
                gra: gra_entries,
            });
        }

        utterances.push(UtteranceJson {
            speaker,
            utterance_index: utt_idx,
            words,
        });
        utt_idx += 1;
    }

    serde_json::to_string(&utterances).map_err(|e| format!("JSON serialization failed: {e}"))
}
