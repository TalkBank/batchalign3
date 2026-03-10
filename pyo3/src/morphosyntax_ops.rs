//! Morphosyntax PyO3 bridge functions.
//!
//! This module is the Rust/Python boundary used by `batchalign_core`
//! Python API.
//!
//! # Execution modes
//!
//! - [`add_morphosyntax_inner`]: per-utterance callback mode (one Python call
//!   per utterance).
//! - [`add_morphosyntax_batched_inner`]: preferred batch mode (single Python
//!   call for all collected utterance payloads).
//!
//! # Responsibility split
//!
//! - Rust (`batchalign-chat-ops`) owns CHAT parsing, payload extraction,
//!   injection, and alignment-safe mutations.
//! - Python callbacks own model inference only (`words/lang -> UD response`).
//!   They never receive raw CHAT source.

use pyo3::prelude::*;
use pyo3::types::PyString;
use talkbank_model::WriteChat;
use talkbank_model::alignment::helpers::AlignmentDomain;
use talkbank_model::model::Line;

use crate::py_json_bridge::{
    morphosyntax_batch_payload_to_object, morphosyntax_payload_to_object,
    parse_morphosyntax_batch_response, parse_morphosyntax_response,
};
use batchalign_chat_ops::morphosyntax::{
    MorphosyntaxBatchItem, collect_payloads as collect_morphosyntax_payloads,
    inject_results as inject_morphosyntax_results,
};

/// Add `%mor/%gra` to a parsed CHAT file using a per-utterance Python callback.
///
/// This is the per-utterance callback path.
///
/// Flow per utterance:
/// 1. Extract alignment-domain words.
/// 2. Build callback JSON payload (`{"words":[...]}`).
/// 3. Call Python and parse UD response.
/// 4. Map UD to CHAT `%mor/%gra` and inject.
/// 5. Optionally retokenize and preserve terminator semantics.
///
/// For throughput-oriented inference, prefer [`add_morphosyntax_batched_inner`].
///
/// # Errors
///
/// Returns `PyRuntimeError` for callback/progress failures and
/// `PyValueError` when UD JSON cannot be parsed.
pub(crate) fn add_morphosyntax_inner(
    py: Python<'_>,
    chat_file: &mut talkbank_model::model::ChatFile,
    lang: &str,
    morphosyntax_fn: &Bound<'_, pyo3::PyAny>,
    progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
    multilingual_policy: batchalign_chat_ops::morphosyntax::MultilingualPolicy,
    retokenize: bool, // kept as bool at PyO3 boundary; converted below
) -> PyResult<()> {
    use batchalign_chat_ops::morphosyntax::TokenizationMode;

    let tokenization_mode = TokenizationMode::from(retokenize);
    let total_utts = chat_file
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();

    let mut utt_idx = 0usize;

    for line in &mut chat_file.lines {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        if multilingual_policy.should_skip_non_primary() && utt.main.content.language_code.is_some()
        {
            utt_idx += 1;
            if let Some(progress) = progress_fn {
                progress.call1((utt_idx, total_utts)).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "progress_fn callback failed: {e}"
                    ))
                })?;
            }
            continue;
        }

        let mut words = Vec::new();
        crate::extract::collect_utterance_content(
            &utt.main.content.content,
            AlignmentDomain::Mor,
            &mut words,
        );

        if !words.is_empty() {
            let terminator_str = utt
                .main
                .content
                .terminator
                .as_ref()
                .map(|t| t.to_chat_string())
                .unwrap_or_else(|| ".".to_string());

            let word_texts: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
            let payload_obj = morphosyntax_payload_to_object(py, &word_texts).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build morphosyntax payload for utterance {utt_idx}: {e}"
                ))
            })?;
            let callback_result = morphosyntax_fn
                .call1((payload_obj, PyString::new(py, lang)))
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "morphosyntax_fn callback failed for utterance {utt_idx}: {e}"
                    ))
                })?;
            let ud_resp: crate::nlp::UdResponse = parse_morphosyntax_response(&callback_result)
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid UD callback response for utterance {utt_idx}: {e}"
                    ))
                })?;

            // Map UD data to CHAT structures in Rust

            if let Some(ud_sentence) = ud_resp.sentences.first() {
                let ctx = crate::nlp::MappingContext {
                    lang: talkbank_model::model::LanguageCode::new(lang),
                };

                let (mors, gra_relations) = match crate::nlp::map_ud_sentence(ud_sentence, &ctx) {
                    Ok(result) => result,
                    Err(e) => {
                        tracing::warn!(utterance = utt_idx, error = %e, "skipping utterance: morphosyntax mapping failed");
                        continue;
                    }
                };

                if tokenization_mode == TokenizationMode::StanzaRetokenize {
                    // Extract token texts for retokenization alignment

                    let tokens: Vec<String> =
                        ud_sentence.words.iter().map(|w| w.text.clone()).collect();

                    crate::retokenize::retokenize_utterance(
                        utt,
                        &words,
                        &tokens,
                        mors,
                        Some(terminator_str),
                        gra_relations,
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to retokenize utterance {utt_idx}: {e}"
                        ))
                    })?;
                } else {
                    crate::inject::inject_morphosyntax(
                        utt,
                        mors,
                        Some(terminator_str),
                        gra_relations,
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to inject morphosyntax for utterance {utt_idx}: {e}"
                        ))
                    })?;
                }
            } else {
                tracing::warn!(
                    utterance = utt_idx,
                    "NLP model returned no sentences, skipping morphosyntax"
                );
            }
        }

        utt_idx += 1;

        if let Some(progress) = progress_fn {
            progress.call1((utt_idx, total_utts)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "progress_fn callback failed: {e}"
                ))
            })?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Morphosyntax helpers (delegated to batchalign_chat_ops::morphosyntax)
// ---------------------------------------------------------------------------

/// Extract final %mor/%gra content strings for specified utterances (pure Rust).
///
/// Thin wrapper around `batchalign_chat_ops::morphosyntax::extract_strings` that
/// handles JSON deserialization of line indices and serialization of results
/// (the PyO3 boundary passes JSON strings, not typed slices).
///
/// The returned JSON preserves the same index order as `line_indices_json`.
pub(crate) fn extract_morphosyntax_strings_inner(
    chat_file: &talkbank_model::model::ChatFile,
    line_indices_json: &str,
) -> Result<String, String> {
    let indices: Vec<usize> = serde_json::from_str(line_indices_json)
        .map_err(|e| format!("Failed to parse line indices JSON: {e}"))?;
    let results = batchalign_chat_ops::morphosyntax::extract_strings(chat_file, &indices)?;
    serde_json::to_string(&results)
        .map_err(|e| format!("Failed to serialize morphosyntax strings: {e}"))
}

/// Batched morphosyntax: collect all utterances -> single Python call -> inject.
///
/// Phase 1 (pure Rust): Walk utterances, build typed payloads.
/// Phase 2 (GIL):       Serialize array -> call Python once -> deserialize array.
/// Phase 3 (pure Rust): Inject results back into each utterance.
///
/// Invariants:
/// - Callback response array length must equal payload length (strict positional
///   mapping).
/// - `line_idx` values in collected payload metadata must still reference the
///   same utterances during phase 3 injection.
/// - When `retokenize` is enabled, injection may update token boundaries, but
///   utterance identity and ordering must remain stable.
///
/// # Errors
///
/// Returns `PyRuntimeError` for callback/progress failures and
/// `PyValueError` for invalid callback JSON or response-length mismatch.
///
/// # Tracing
///
/// At `debug` level (`-vv`): logs utterance/word counts and callback round-trip.
/// At `trace` level (`-vvv`): logs the full payload JSON sent to Python.
pub(crate) fn add_morphosyntax_batched_inner(
    py: Python<'_>,
    chat_file: &mut talkbank_model::model::ChatFile,
    lang: &str,
    batch_fn: &Bound<'_, pyo3::PyAny>,
    progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
    multilingual_policy: batchalign_chat_ops::morphosyntax::MultilingualPolicy,
    retokenize: bool, // kept as bool at PyO3 boundary; converted below
) -> PyResult<()> {
    use batchalign_chat_ops::morphosyntax::TokenizationMode;

    let tokenization_mode = TokenizationMode::from(retokenize);
    let primary_lang = talkbank_model::model::LanguageCode::new(lang);

    let declared_languages: Vec<talkbank_model::model::LanguageCode> =
        if chat_file.languages.is_empty() {
            vec![primary_lang.clone()]
        } else {
            chat_file.languages.0.clone()
        };

    // --- Phase 1: Collect payloads (pure Rust) ---
    let (batch_items, total_utts) = collect_morphosyntax_payloads(
        chat_file,
        &primary_lang,
        &declared_languages,
        multilingual_policy,
    );

    let total_words: usize = batch_items
        .iter()
        .map(|(_, _, item, _)| item.words.len())
        .sum();
    tracing::debug!(
        utterances = batch_items.len(),
        total_words,
        total_utts,
        lang,
        ?tokenization_mode,
        "Phase 1: collected morphosyntax payloads"
    );

    if batch_items.is_empty() {
        if let Some(progress) = progress_fn {
            progress.call1((total_utts, total_utts)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "progress_fn callback failed: {e}"
                ))
            })?;
        }
        return Ok(());
    }

    // --- Phase 2: Single Python call (GIL held briefly) ---
    let payloads: Vec<&MorphosyntaxBatchItem> =
        batch_items.iter().map(|(_, _, item, _)| item).collect();
    let payload_obj = morphosyntax_batch_payload_to_object(py, &payloads).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Failed to build batch morphosyntax payload: {e}"
        ))
    })?;
    let callback_result = batch_fn
        .call1((payload_obj, PyString::new(py, lang)))
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "batch morphosyntax callback failed: {e}"
            ))
        })?;
    let responses: Vec<crate::nlp::UdResponse> =
        parse_morphosyntax_batch_response(&callback_result).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid batch callback response: {e}"))
        })?;

    tracing::debug!(
        response_count = responses.len(),
        "Phase 2: callback returned responses"
    );

    if responses.len() != batch_items.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Batch response length mismatch: expected {}, got {}",
            batch_items.len(),
            responses.len()
        )));
    }

    // --- Phase 3: Inject results (pure Rust) ---
    let lang_code = talkbank_model::model::LanguageCode::new(lang);
    let empty_mwt = std::collections::BTreeMap::new();
    inject_morphosyntax_results(
        chat_file,
        batch_items,
        responses,
        &lang_code,
        tokenization_mode,
        &empty_mwt,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    tracing::debug!("Phase 3: morphosyntax injection complete");

    // Report final progress
    if let Some(progress) = progress_fn {
        progress.call1((total_utts, total_utts)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("progress_fn callback failed: {e}"))
        })?;
    }

    Ok(())
}
