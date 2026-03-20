//! Translation and utterance segmentation inner functions.

use pyo3::prelude::*;
use talkbank_model::alignment::helpers::TierDomain;
use talkbank_model::model::{Line, ParseHealthTier};

use crate::py_json_bridge::{
    parse_translation_response, parse_utseg_batch_response, parse_utseg_response,
    translation_payload_to_object, utseg_batch_payload_to_object, utseg_payload_to_object,
};
use batchalign_chat_ops::utseg::{UtsegBatchItem, UtsegResponse};

pub(crate) fn add_translation_inner(
    py: Python<'_>,
    chat_file: &mut talkbank_model::model::ChatFile,
    translation_fn: &Bound<'_, pyo3::PyAny>,
    progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
) -> PyResult<()> {
    let total_utts = chat_file
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();

    let mut utt_idx = 0usize;

    for line in chat_file.lines.iter_mut() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mut words = Vec::new();
        crate::extract::collect_utterance_content(
            &utt.main.content.content,
            TierDomain::Mor,
            &mut words,
        );

        if !words.is_empty() {
            let text: String = words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let speaker = utt.main.speaker.as_str().to_string();
            let payload_obj = translation_payload_to_object(py, &text, &speaker).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build translation payload for utterance {utt_idx}: {e}"
                ))
            })?;
            let callback_result = translation_fn.call1((payload_obj,)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "translation_fn callback failed for utterance {utt_idx}: {e}"
                ))
            })?;
            let translation_text = parse_translation_response(&callback_result)
                .map(|resp| resp.translation)
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Invalid translation callback response for utterance {utt_idx}: {e}"
                    ))
                })?;
            if !translation_text.is_empty()
                && let Err(e) =
                    batchalign_chat_ops::translate::inject_translation(utt, &translation_text)
            {
                tracing::warn!(utterance = utt_idx, error = %e, "failed to inject translation");
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

pub(crate) fn add_utterance_segmentation_inner(
    py: Python<'_>,
    chat_file: &mut talkbank_model::model::ChatFile,
    segmentation_fn: &Bound<'_, pyo3::PyAny>,
    progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
) -> PyResult<()> {
    let total_utts = chat_file
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();

    let old_lines = std::mem::take(&mut chat_file.lines.0);
    let mut new_lines: Vec<Line> = Vec::with_capacity(old_lines.len());
    let mut utt_idx = 0usize;

    for line in old_lines {
        let utt = match line {
            Line::Utterance(u) => u,
            other => {
                new_lines.push(other);
                continue;
            }
        };

        let mut words = Vec::new();
        crate::extract::collect_utterance_content(
            &utt.main.content.content,
            TierDomain::Mor,
            &mut words,
        );

        if words.is_empty() || words.len() <= 1 {
            new_lines.push(Line::Utterance(utt));
        } else {
            let text: String = words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let word_strs: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
            let payload_obj = utseg_payload_to_object(py, &word_strs, &text).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to build utseg payload for utterance {utt_idx}: {e}"
                ))
            })?;
            let callback_result = segmentation_fn.call1((payload_obj,)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "segmentation_fn callback failed for utterance {utt_idx}: {e}"
                ))
            })?;
            let parsed: Result<UtsegResponse, _> =
                parse_utseg_response(&callback_result).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Invalid utseg callback response for utterance {utt_idx}: {e}"
                    ))
                });

            match parsed {
                Ok(resp) if resp.assignments.len() == words.len() => {
                    let split_utts =
                        crate::utterance_segmentation::split_utterance(*utt, &resp.assignments);
                    for split_utt in split_utts {
                        new_lines.push(Line::Utterance(Box::new(split_utt)));
                    }
                }
                Ok(resp) => {
                    tracing::warn!(
                        utterance = utt_idx,
                        expected = words.len(),
                        got = resp.assignments.len(),
                        "utseg assignment length mismatch, keeping original"
                    );
                    let mut utt = utt;
                    utt.mark_parse_taint(ParseHealthTier::Main);
                    new_lines.push(Line::Utterance(utt));
                }
                Err(e) => {
                    tracing::warn!(
                        utterance = utt_idx,
                        error = %e,
                        "utseg callback returned invalid JSON, keeping original"
                    );
                    let mut utt = utt;
                    utt.mark_parse_taint(ParseHealthTier::Main);
                    new_lines.push(Line::Utterance(utt));
                }
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

    chat_file.lines.0 = new_lines;

    Ok(())
}

/// Batched utterance segmentation: collect all utterances -> single Python call -> split.
///
/// Phase 1 (pure Rust): Walk utterances, build typed payloads.
/// Phase 2 (GIL):       Serialize array -> call Python once -> deserialize array.
/// Phase 3 (pure Rust): Take ownership of lines, rebuild with splits.
pub(crate) fn add_utterance_segmentation_batched_inner(
    py: Python<'_>,
    chat_file: &mut talkbank_model::model::ChatFile,
    batch_fn: &Bound<'_, pyo3::PyAny>,
    progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
) -> PyResult<()> {
    let total_utts = chat_file
        .lines
        .iter()
        .filter(|l| matches!(l, Line::Utterance(_)))
        .count();

    // --- Phase 1: Collect payloads (pure Rust) ---
    // Each item: (utt_ordinal, payload)
    // Only multi-word utterances get batched.
    let mut batch_items: Vec<(usize, UtsegBatchItem)> = Vec::new();
    let mut utt_idx = 0usize;

    for line in chat_file.lines.iter() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        let mut words = Vec::new();
        crate::extract::collect_utterance_content(
            &utt.main.content.content,
            TierDomain::Mor,
            &mut words,
        );

        if words.len() > 1 {
            let text: String = words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let word_texts: Vec<String> =
                words.iter().map(|w| w.text.as_str().to_string()).collect();

            batch_items.push((
                utt_idx,
                UtsegBatchItem {
                    words: word_texts,
                    text,
                },
            ));
        }

        utt_idx += 1;
    }

    // --- Phase 2: Single Python call (GIL held briefly) ---
    // Build a map of utt_ordinal -> assignments for Phase 3
    let mut assignment_map: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    let mut mismatched_utts: std::collections::HashSet<usize> = std::collections::HashSet::new();

    if !batch_items.is_empty() {
        let payloads: Vec<&UtsegBatchItem> = batch_items.iter().map(|(_, item)| item).collect();
        let payload_obj = utseg_batch_payload_to_object(py, &payloads).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build batch utseg payload: {e}"
            ))
        })?;
        let callback_result = batch_fn.call1((payload_obj,)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("batch utseg callback failed: {e}"))
        })?;
        let responses: Vec<UtsegResponse> =
            parse_utseg_batch_response(&callback_result).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid batch utseg callback response: {e}"
                ))
            })?;

        if responses.len() != batch_items.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Batch utseg response length mismatch: expected {}, got {}",
                batch_items.len(),
                responses.len()
            )));
        }

        for (resp, (utt_ordinal, item)) in responses.into_iter().zip(batch_items.into_iter()) {
            if resp.assignments.len() == item.words.len() {
                assignment_map.insert(utt_ordinal, resp.assignments);
            } else {
                tracing::warn!(
                    utterance = utt_ordinal,
                    expected = item.words.len(),
                    got = resp.assignments.len(),
                    "utseg assignment length mismatch, keeping original"
                );
                mismatched_utts.insert(utt_ordinal);
            }
        }
    }

    // --- Phase 3: Rebuild lines with splits (pure Rust) ---
    let old_lines = std::mem::take(&mut chat_file.lines.0);
    let mut new_lines: Vec<Line> = Vec::with_capacity(old_lines.len());
    let mut utt_ordinal = 0usize;

    for line in old_lines {
        let utt = match line {
            Line::Utterance(u) => u,
            other => {
                new_lines.push(other);
                continue;
            }
        };

        if let Some(assignments) = assignment_map.remove(&utt_ordinal) {
            let split_utts = crate::utterance_segmentation::split_utterance(*utt, &assignments);
            for split_utt in split_utts {
                new_lines.push(Line::Utterance(Box::new(split_utt)));
            }
        } else {
            let mut utt = utt;
            if mismatched_utts.remove(&utt_ordinal) {
                utt.mark_parse_taint(ParseHealthTier::Main);
            }
            new_lines.push(Line::Utterance(utt));
        }

        utt_ordinal += 1;

        if let Some(progress) = progress_fn {
            progress.call1((utt_ordinal, total_utts)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "progress_fn callback failed: {e}"
                ))
            })?;
        }
    }

    chat_file.lines.0 = new_lines;

    Ok(())
}
