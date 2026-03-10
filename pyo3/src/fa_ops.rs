//! Forced alignment PyO3 bridge function.
//!
//! This module drives forced alignment for the Python callback API used
//! by `batchalign_core`.
//!
//! Rust owns grouping, payload construction, timing injection, `%wor`
//! generation, and final CHAT invariants. Python owns model inference over a
//! serialized FA payload and returns timing JSON.
//!
//! # Ordering invariants
//!
//! Post-processing order is load-bearing:
//! 1. inject timings
//! 2. `postprocess_utterance_timings`
//! 3. `update_utterance_bullet`
//! 4. `add_wor_tier`
//! 5. E362 monotonicity pass
//! 6. E704 same-speaker overlap pass

use pyo3::prelude::*;
use talkbank_model::model::Line;

use crate::py_json_bridge::{fa_payload_to_object, parse_fa_response};
use batchalign_chat_ops::fa::{
    FaTimingMode, strip_e704_same_speaker_overlaps, strip_timing_from_content,
};

/// Add forced-alignment timing to a parsed CHAT file via Python callback.
///
/// Flow:
/// 1. Group utterances into audio windows.
/// 2. Build FA payload for each group and call Python.
/// 3. Parse timings and map them back to utterance-local offsets.
/// 4. Inject timings, generate `%wor`, and normalize bullets.
/// 5. Enforce utterance-level monotonicity (E362) and same-speaker overlap
///    constraints (E704).
///
/// # Errors
///
/// Returns `PyRuntimeError` when the Python callback or progress callback
/// fails, and `PyValueError` when callback JSON cannot be parsed into timing
/// responses.
pub(crate) fn add_forced_alignment_inner(
    py: Python<'_>,
    chat_file: &mut talkbank_model::model::ChatFile,
    fa_callback: &Bound<'_, pyo3::PyAny>,
    progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
    pauses: bool,
    max_group_ms: u64,
    total_audio_ms: Option<u64>,
) -> PyResult<()> {
    // Convert bool from Python boundary to typed enum for all internal Rust calls.
    let timing_mode = if pauses {
        FaTimingMode::WithPauses
    } else {
        FaTimingMode::Continuous
    };

    let groups = crate::forced_alignment::group_utterances(chat_file, max_group_ms, total_audio_ms);
    let total_groups = groups.len();

    for (group_idx, group) in groups.iter().enumerate() {
        if group.words.is_empty() {
            continue;
        }

        let payload = crate::forced_alignment::build_fa_item(group, timing_mode);
        let payload_obj = fa_payload_to_object(py, &payload).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to build FA payload for group {group_idx}: {e}"
            ))
        })?;
        let callback_result = fa_callback.call1((payload_obj,)).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "fa_callback failed for group {group_idx}: {e}"
            ))
        })?;
        let raw_response = parse_fa_response(&callback_result).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid FA callback response for group {group_idx}: {e}"
            ))
        })?;
        let response_json = serde_json::to_string(&raw_response).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to normalize FA response for group {group_idx}: {e}"
            ))
        })?;
        let word_timings = crate::forced_alignment::parse_fa_response(
            &response_json,
            &group.words,
            group.audio_start_ms(),
            timing_mode,
        )
        .map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid FA callback response for group {group_idx}: {e}"
            ))
        })?;

        if word_timings.len() != group.words.len() {
            tracing::warn!(
                words = group.words.len(),
                timings = word_timings.len(),
                "FA timing count mismatch: extra words will have no timing"
            );
        }

        let mut timing_offset_per_utt: std::collections::HashMap<
            usize,
            Vec<Option<crate::forced_alignment::WordTiming>>,
        > = std::collections::HashMap::new();

        for (word_idx, word) in group.words.iter().enumerate() {
            let timing = if word_idx < word_timings.len() {
                word_timings[word_idx]
            } else {
                None
            };
            timing_offset_per_utt
                .entry(word.utterance_index.raw())
                .or_default()
                .push(timing);
        }

        let mut utt_idx = 0;
        for line in chat_file.lines.iter_mut() {
            let utt = match line {
                Line::Utterance(u) => u,
                _ => continue,
            };

            if let Some(timings) = timing_offset_per_utt.get(&utt_idx) {
                let mut offset = 0;
                crate::forced_alignment::inject_timings_for_utterance(utt, timings, &mut offset);
            }

            utt_idx += 1;
        }

        if let Some(progress) = progress_fn {
            progress.call1((group_idx + 1, total_groups)).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "progress_fn callback failed: {e}"
                ))
            })?;
        }
    }

    let mut grouped_utt_indices: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for group in &groups {
        for &idx in &group.utterance_indices {
            grouped_utt_indices.insert(idx.raw());
        }
    }

    let mut utt_idx = 0usize;
    for line in chat_file.lines.iter_mut() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        if grouped_utt_indices.contains(&utt_idx) {
            // Postprocess BEFORE update_bullet: bounding uses the UTR bullet
            // (which captures actual speech extent), not a bullet recomputed
            // from raw word onsets where end == start.
            crate::forced_alignment::postprocess_utterance_timings(utt, timing_mode);
            crate::forced_alignment::update_utterance_bullet(utt);
            crate::forced_alignment::add_wor_tier(utt);
        }

        utt_idx += 1;
    }

    // Final pass: enforce utterance-level timestamp monotonicity.
    //
    // When the CHAT transcript's textual ordering does not match the temporal
    // order in the audio (common in multi-speaker conversations with
    // overlapping turns or backchannels written out of chronological sequence),
    // the FA pipeline can assign timestamps that are correct for each
    // individual utterance but non-monotonic across utterances — violating
    // E362.
    //
    // Root cause: UTR (DP alignment) skips utterances it can't place in text
    // order.  FA then uses proportional estimation for those untimed
    // utterances, which gives them a window near their actual audio position.
    // The FA callback correctly aligns them there.  The result: in-order
    // utterances get correct timestamps, out-of-order utterances ALSO get
    // correct timestamps, but the combination is non-monotonic.
    //
    // Remedy: strip timing from utterances that would break monotonicity.
    // Those utterances lose their alignment rather than producing invalid CHAT.
    {
        use talkbank_model::model::DependentTier;
        let mut last_start_ms: u64 = 0;
        for line in chat_file.lines.iter_mut() {
            let utt = match line {
                Line::Utterance(u) => u,
                _ => continue,
            };
            match utt.main.content.bullet.as_ref().map(|b| b.timing.start_ms) {
                Some(s) if s < last_start_ms => {
                    // Non-monotonic: strip all FA-assigned timing from this utterance.
                    tracing::warn!(
                        start_ms = s,
                        last_start_ms,
                        "stripping non-monotonic utterance timing to enforce E362"
                    );
                    utt.main.content.bullet = None;
                    strip_timing_from_content(&mut utt.main.content.content.0);
                    utt.dependent_tiers
                        .retain(|t| !matches!(t, DependentTier::Wor(_)));
                }
                Some(s) => last_start_ms = s,
                None => {}
            }
        }
    }

    // E704 pass: enforce same-speaker temporal non-overlap.
    // See `strip_e704_same_speaker_overlaps` for details.
    strip_e704_same_speaker_overlaps(chat_file);

    Ok(())
}
