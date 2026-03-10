//! Standalone #[pyfunction]s — thin wrappers around inner functions.

use pyo3::prelude::*;
use talkbank_model::WriteChat;
use talkbank_model::model::Line;

use crate::build::build_chat_inner;
use crate::metadata::serialize_extracted_words;
use crate::parse::{parse_alignment_domain, parse_lenient_pure, parse_strict_pure};
use crate::tier_ops::add_dependent_tiers_inner;

#[derive(serde::Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AlignResultJson<'a> {
    Match {
        key: &'a str,
        payload_idx: usize,
        reference_idx: usize,
    },
    ExtraPayload {
        key: &'a str,
        payload_idx: usize,
    },
    ExtraReference {
        key: &'a str,
        reference_idx: usize,
    },
}

/// Parse CHAT text and serialize it back.
///
/// This validates the round-trip: parse via tree-sitter, serialize via WriteChat.
/// Returns the serialized CHAT text string.
///
/// Raises `ValueError` if parsing fails.
#[pyfunction]
pub(crate) fn parse_and_serialize(py: Python<'_>, chat_text: &str) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        if chat_text.trim().is_empty() {
            return Ok(String::new());
        }
        let chat_file = parse_lenient_pure(chat_text)?;
        Ok(chat_file.to_chat_string())
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Extract NLP-ready words from a CHAT file (standalone version).
///
/// Parses the CHAT text strictly, then extracts words for the specified alignment
/// domain. This is the standalone `#[pyfunction]` counterpart to
/// `ParsedChat.extract_nlp_words()`.
///
/// `domain` must be one of: `"mor"`, `"wor"`, `"pho"`, `"sin"`.
///
/// # Return schema (JSON string)
///
/// ```json
/// [
///   {
///     "speaker": "PAR0",
///     "utterance_index": 0,
///     "words": [
///       {
///         "text": "hello",
///         "raw_text": "hello",
///         "form_type": null,
///         "lang_marker": false,
///         "provenance": "chat_original"
///       }
///     ]
///   }
/// ]
/// ```
///
/// See [`py_extract_nlp_words`] for full field documentation.
///
/// Raises `ValueError` if parsing fails or domain is invalid.
#[pyfunction]
pub(crate) fn extract_nlp_words(py: Python<'_>, chat_text: &str, domain: &str) -> PyResult<String> {
    let domain = parse_alignment_domain(domain)?;
    py.detach(|| -> Result<String, String> {
        let chat_file = parse_strict_pure(chat_text)?;
        let extracted = crate::extract::extract_words(&chat_file, domain);
        Ok(serialize_extracted_words(&extracted))
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Align two string sequences using Hirschberg's algorithm.
///
/// Returns a JSON string: a list of alignment results, each with:
///   - "type": "match" | "extra_payload" | "extra_reference"
///   - "key": the string that was matched or extra
///   - "payload_idx": index into payload (present for match and extra_payload)
///   - "reference_idx": index into reference (present for match and extra_reference)
///
/// Cost model: match=0, substitution=2, insertion/deletion=1.
///
/// Raises `TypeError` if inputs are not lists of strings.
#[pyfunction]
#[pyo3(name = "dp_align", signature = (payload, reference, case_insensitive=false))]
pub(crate) fn py_dp_align(
    py: Python<'_>,
    payload: Vec<String>,
    reference: Vec<String>,
    case_insensitive: bool,
) -> String {
    py.detach(|| {
        let mode = if case_insensitive {
            crate::dp_align::MatchMode::CaseInsensitive
        } else {
            crate::dp_align::MatchMode::Exact
        };

        let results = crate::dp_align::align(&payload, &reference, mode);
        let serializable: Vec<AlignResultJson<'_>> = results
            .iter()
            .map(|r| match r {
                crate::dp_align::AlignResult::Match {
                    key,
                    payload_idx,
                    reference_idx,
                } => AlignResultJson::Match {
                    key,
                    payload_idx: *payload_idx,
                    reference_idx: *reference_idx,
                },
                crate::dp_align::AlignResult::ExtraPayload { key, payload_idx } => {
                    AlignResultJson::ExtraPayload {
                        key,
                        payload_idx: *payload_idx,
                    }
                }
                crate::dp_align::AlignResult::ExtraReference { key, reference_idx } => {
                    AlignResultJson::ExtraReference {
                        key,
                        reference_idx: *reference_idx,
                    }
                }
            })
            .collect();

        serde_json::to_string(&serializable).unwrap_or_else(|e| {
            tracing::error!(error = %e, "failed to serialize dp_align output");
            "[]".to_string()
        })
    })
}

/// Build a valid CHAT file from a JSON transcript description.
///
/// This is used by ASR engines to generate CHAT output without going through
/// the Python Document/CHATFile layer.
///
/// Input JSON schema:
/// ```json
/// {
///   "langs": ["eng"],
///   "media_name": "test",       // optional
///   "media_type": "audio",      // optional: "audio" or "video"
///   "participants": [
///     {"id": "PAR0", "name": "Participant", "role": "Participant"}
///   ],
///   "utterances": [
///     {
///       "speaker": "PAR0",
///       "words": [
///         {"text": "hello", "start_ms": 100, "end_ms": 500},
///         {"text": ".", "start_ms": null, "end_ms": null}
///       ]
///     }
///   ]
/// }
/// ```
///
/// Returns a valid CHAT text string.
#[pyfunction]
pub(crate) fn build_chat(
    py: Python<'_>,
    transcript_json: talkbank_model::PythonTranscriptJson,
) -> PyResult<String> {
    py.detach(|| build_chat_inner(transcript_json).map(|cf| cf.to_chat_string()))
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Add user-defined dependent tiers to utterances in a CHAT file (standalone version).
///
/// Parses the CHAT text, adds tiers, and re-serializes. This is the standalone
/// `#[pyfunction]` counterpart to `ParsedChat.add_dependent_tiers()`.
///
/// Takes CHAT text and a JSON array of tier entries:
/// `[{"utterance_index": 0, "label": "xcoref", "content": "(1, -, 1)"}, ...]`
///
/// Each entry adds a `%label:\tcontent` dependent tier to the specified utterance.
/// If a tier with the same label already exists on that utterance, it is replaced.
///
/// Validation: labels must start with `x` and must not be standard CHAT tier
/// names (e.g., `mor`, `gra`, `wor`). All labels are validated before any
/// mutations occur. See [`py_add_dependent_tiers`] for full validation rules.
///
/// Raises `ValueError` if parsing fails, JSON is malformed, or label validation fails.
#[pyfunction]
pub(crate) fn add_dependent_tiers(
    py: Python<'_>,
    chat_text: &str,
    tiers_json: &str,
) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        let mut chat_file = parse_strict_pure(chat_text)?;
        add_dependent_tiers_inner(&mut chat_file, tiers_json).map_err(|e| e.to_string())?;
        Ok(chat_file.to_chat_string())
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Extract per-speaker timed tiers from a CHAT file for TextGrid generation.
///
/// Returns a JSON string: `{"SPEAKER": [{"text": "...", "start_ms": N, "end_ms": N}, ...], ...}`
///
/// Uses the full pipeline (TreeSitter parse + validation + alignment) so that
/// `%wor` timing is distributed into `Word.inline_bullet` fields on the AST.
///
/// When `by_word` is true, walks the aligned AST and emits each word with
/// `Timed` timing. When false, emits one entry per utterance using the main
/// tier bullet for timing and `to_content_string()` for text.
///
/// Raises `ValueError` if parsing fails.
#[derive(serde::Serialize)]
struct TimedEntry {
    text: String,
    start_ms: u64,
    end_ms: u64,
}

#[pyfunction]
pub(crate) fn extract_timed_tiers(
    py: Python<'_>,
    chat_text: &str,
    by_word: bool,
) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        // Full pipeline: TreeSitter parse + validation + alignment.
        let options = if by_word {
            talkbank_model::ParseValidateOptions::default().with_alignment()
        } else {
            talkbank_model::ParseValidateOptions::default()
        };
        let errors = talkbank_model::ErrorCollector::new();
        let chat_file =
            talkbank_transform::parse_and_validate_streaming(chat_text, options, &errors)
                .map_err(|e| format!("Pipeline error: {e}"))?;

        let mut tiers: indexmap::IndexMap<String, Vec<TimedEntry>> = indexmap::IndexMap::new();

        for line in chat_file.lines.iter() {
            let utt = match line {
                Line::Utterance(u) => u,
                _ => continue,
            };

            let speaker = utt.main.speaker.as_str().to_string();

            if by_word {
                // Prefer %wor tier (has per-word inline bullets from parsing),
                // fall back to main tier words (have inline_bullet if built from JSON).
                if let Some(wor) = utt.wor_tier() {
                    for word in wor.words() {
                        push_if_timed(word, &speaker, &mut tiers);
                    }
                } else {
                    collect_timed_words_utt(&utt.main.content.content.0, &speaker, &mut tiers);
                }
            } else if let Some(ref bullet) = utt.main.content.bullet {
                let text = utt.main.content.to_content_string_no_bullets();
                let text = text.trim();

                if !text.is_empty() {
                    tiers.entry(speaker).or_default().push(TimedEntry {
                        text: text.to_string(),
                        start_ms: bullet.timing.start_ms,
                        end_ms: bullet.timing.end_ms,
                    });
                }
            }
        }

        serde_json::to_string(&tiers).map_err(|e| format!("JSON serialization failed: {e}"))
    })
    .map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Walk UtteranceContent items and collect words that have inline_bullet timing.
fn collect_timed_words_utt(
    items: &[talkbank_model::model::content::UtteranceContent],
    speaker: &str,
    tiers: &mut indexmap::IndexMap<String, Vec<TimedEntry>>,
) {
    use talkbank_model::model::content::UtteranceContent;

    for item in items {
        match item {
            UtteranceContent::Word(w) => push_if_timed(w, speaker, tiers),
            UtteranceContent::AnnotatedWord(aw) => push_if_timed(&aw.inner, speaker, tiers),
            UtteranceContent::ReplacedWord(rw) => push_if_timed(&rw.word, speaker, tiers),
            UtteranceContent::Group(g) => {
                collect_timed_words_bracketed(&g.content.content.0, speaker, tiers);
            }
            UtteranceContent::AnnotatedGroup(g) => {
                collect_timed_words_bracketed(&g.inner.content.content.0, speaker, tiers);
            }
            UtteranceContent::PhoGroup(g) => {
                collect_timed_words_bracketed(&g.content.content.0, speaker, tiers);
            }
            UtteranceContent::SinGroup(g) => {
                collect_timed_words_bracketed(&g.content.content.0, speaker, tiers);
            }
            UtteranceContent::Quotation(g) => {
                collect_timed_words_bracketed(&g.content.content.0, speaker, tiers);
            }
            _ => {}
        }
    }
}

/// Walk BracketedItem items and collect words that have inline_bullet timing.
fn collect_timed_words_bracketed(
    items: &[talkbank_model::model::content::BracketedItem],
    speaker: &str,
    tiers: &mut indexmap::IndexMap<String, Vec<TimedEntry>>,
) {
    use talkbank_model::model::content::BracketedItem;

    for item in items {
        match item {
            BracketedItem::Word(w) => push_if_timed(w, speaker, tiers),
            BracketedItem::AnnotatedWord(aw) => push_if_timed(&aw.inner, speaker, tiers),
            BracketedItem::ReplacedWord(rw) => push_if_timed(&rw.word, speaker, tiers),
            BracketedItem::AnnotatedGroup(g) => {
                collect_timed_words_bracketed(&g.inner.content.content.0, speaker, tiers);
            }
            BracketedItem::PhoGroup(g) => {
                collect_timed_words_bracketed(&g.content.content.0, speaker, tiers);
            }
            BracketedItem::SinGroup(g) => {
                collect_timed_words_bracketed(&g.content.content.0, speaker, tiers);
            }
            BracketedItem::Quotation(g) => {
                collect_timed_words_bracketed(&g.content.content.0, speaker, tiers);
            }
            _ => {}
        }
    }
}

/// If the word has an inline bullet (timing), push it to the tiers map.
fn push_if_timed(
    word: &talkbank_model::model::Word,
    speaker: &str,
    tiers: &mut indexmap::IndexMap<String, Vec<TimedEntry>>,
) {
    if let Some(ref bullet) = word.inline_bullet {
        tiers
            .entry(speaker.to_string())
            .or_default()
            .push(TimedEntry {
                text: word.cleaned_text().to_string(),
                start_ms: bullet.timing.start_ms,
                end_ms: bullet.timing.end_ms,
            });
    }
}

/// Return the CHAT terminators (sentence-ending punctuation) as strings.
///
/// These are the grammatical terminators defined by the CHAT specification.
/// ASR-specific extras like `...` and `(.)` are NOT included -- add those
/// on the Python side.
#[pyfunction]
pub(crate) fn chat_terminators() -> Vec<String> {
    use talkbank_model::Span;
    use talkbank_model::model::content::Terminator;
    vec![
        Terminator::Period { span: Span::DUMMY },
        Terminator::Question { span: Span::DUMMY },
        Terminator::Exclamation { span: Span::DUMMY },
        Terminator::TrailingOff { span: Span::DUMMY },
        Terminator::Interruption { span: Span::DUMMY },
        Terminator::SelfInterruption { span: Span::DUMMY },
        Terminator::InterruptedQuestion { span: Span::DUMMY },
        Terminator::BrokenQuestion { span: Span::DUMMY },
        Terminator::QuotedNewLine { span: Span::DUMMY },
        Terminator::QuotedPeriodSimple { span: Span::DUMMY },
        Terminator::SelfInterruptedQuestion { span: Span::DUMMY },
        Terminator::TrailingOffQuestion { span: Span::DUMMY },
        Terminator::BreakForCoding { span: Span::DUMMY },
        // CA terminators intentionally excluded -- ASR never produces them
    ]
    .into_iter()
    .map(|t| t.to_string())
    .collect()
}

/// Return the CHAT morphological punctuation (intra-utterance separators) as strings.
///
/// These are the non-CA separators that appear in %mor tiers: vocative,
/// tag, and comma.
#[pyfunction]
pub(crate) fn chat_mor_punct() -> Vec<String> {
    use talkbank_model::Span;
    use talkbank_model::model::content::Separator;
    let items: Vec<Separator> = vec![
        Separator::Vocative { span: Span::DUMMY }, // ‡
        Separator::Tag { span: Span::DUMMY },      // „
        Separator::Comma { span: Span::DUMMY },    // ,
    ];
    items.into_iter().map(|s| s.to_string()).collect()
}

/// Align Stanza tokenizer output back to original CHAT words.
///
/// Thin PyO3 wrapper around [`batchalign_chat_ops::tokenizer_realign::align_tokens`].
/// Converts `PatchedToken` results to Python objects: plain `str` for unchanged
/// tokens, `(str, bool)` tuples for MWT hints.
#[pyfunction]
pub(crate) fn align_tokens(
    py: Python<'_>,
    original_words: Vec<String>,
    stanza_tokens: Vec<String>,
    alpha2: String,
) -> PyResult<Py<pyo3::types::PyList>> {
    use batchalign_chat_ops::tokenizer_realign::{self, PatchedToken};
    use pyo3::types::{PyBool, PyList, PyString, PyTuple};

    let patched =
        py.detach(|| tokenizer_realign::align_tokens(&original_words, &stanza_tokens, &alpha2));

    let result = PyList::empty(py);
    for tok in &patched {
        match tok {
            PatchedToken::Plain(s) => {
                result.append(PyString::new(py, s))?;
            }
            PatchedToken::Hint(s, expand) => {
                let s_any: Py<PyAny> = PyString::new(py, s).unbind().into_any();
                let b_any: Py<PyAny> = PyBool::new(py, *expand).to_owned().unbind().into_any();
                let tup = PyTuple::new(py, [s_any.bind(py), b_any.bind(py)])?;
                result.append(tup)?;
            }
        }
    }

    Ok(result.unbind())
}

/// Normalize a word list for WER comparison.
///
/// Applies compound splitting, contraction expansion, filler normalization,
/// name replacement, abbreviation expansion, and special word handling.
///
/// This replaces the Python `_conform()` function.
#[pyfunction]
pub(crate) fn wer_conform(py: Python<'_>, words: Vec<String>) -> PyResult<Vec<String>> {
    Ok(py.detach(|| batchalign_chat_ops::wer_conform::conform_words(&words)))
}

/// Normalize Cantonese text: simplified → HK traditional + domain replacements.
///
/// This is the Rust equivalent of the Python `normalize_cantonese_text()` from
/// `batchalign/inference/hk/_common.py`. Uses `zhconv` (pure Rust, OpenCC +
/// MediaWiki rulesets) for s2hk conversion, then applies a 31-entry domain
/// replacement table for Cantonese-specific character corrections.
#[pyfunction]
pub(crate) fn normalize_cantonese(py: Python<'_>, text: &str) -> String {
    py.detach(|| batchalign_chat_ops::asr_postprocess::cantonese::normalize_cantonese(text))
}

/// Normalize Cantonese text and split into per-character tokens.
///
/// Strips CJK punctuation and whitespace after normalization.
/// Used by FunASR Cantonese to align per-character timestamps.
#[pyfunction]
pub(crate) fn cantonese_char_tokens(py: Python<'_>, text: &str) -> Vec<String> {
    py.detach(|| batchalign_chat_ops::asr_postprocess::cantonese::cantonese_char_tokens(text))
}

/// Compute Word Error Rate between hypothesis and reference word lists.
///
/// Performs the full WER pipeline: dash removal, single-letter combining,
/// Chinese character decomposition, WER conforming, DP alignment, error
/// counting, and diff generation.
///
/// Returns a JSON string with keys: `wer`, `total`, `matches`, `diff`.
#[pyfunction]
#[pyo3(signature = (hypothesis, reference, langs=None))]
pub(crate) fn wer_compute(
    py: Python<'_>,
    hypothesis: Vec<String>,
    reference: Vec<String>,
    langs: Option<Vec<String>>,
) -> String {
    let langs = langs.unwrap_or_else(|| vec!["eng".to_string()]);
    let result =
        py.detach(|| batchalign_chat_ops::benchmark::compute_wer(&hypothesis, &reference, &langs));
    serde_json::json!({
        "wer": result.wer,
        "total": result.total,
        "matches": result.matches,
        "diff": result.diff,
    })
    .to_string()
}

/// Compute structured WER metrics for the Python convenience wrapper.
///
/// Returns a JSON string with keys:
/// `wer`, `cer`, `accuracy`, `matches`, `total`, `error`.
#[pyfunction]
#[pyo3(signature = (hypothesis, reference, langs=None))]
pub(crate) fn wer_metrics(
    py: Python<'_>,
    hypothesis: Vec<String>,
    reference: Vec<String>,
    langs: Option<Vec<String>>,
) -> String {
    let langs = langs.unwrap_or_else(|| vec!["eng".to_string()]);
    let result =
        py.detach(|| batchalign_chat_ops::benchmark::compute_wer(&hypothesis, &reference, &langs));
    serde_json::json!({
        "wer": result.wer,
        "cer": 0.0,
        "accuracy": 1.0 - result.wer,
        "matches": result.matches,
        "total": result.total,
        "error": result.diff,
    })
    .to_string()
}
