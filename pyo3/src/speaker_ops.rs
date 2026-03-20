//! Speaker reassignment and utterance timing (ID-mapped when available, deterministic fallback otherwise).

use batchalign_chat_ops::speaker::{SpeakerSegment, reassign_speakers};
use pyo3::PyResult;
use talkbank_model::model::Line;

use crate::AsrWordJson;

pub(crate) fn reassign_speakers_inner(
    mut chat_file: talkbank_model::model::ChatFile,
    segments_json: &str,
    lang: &str,
) -> PyResult<talkbank_model::model::ChatFile> {
    let segments: Vec<SpeakerSegment> = serde_json::from_str(segments_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid segments JSON: {e}"))
    })?;

    if segments.is_empty() {
        return Ok(chat_file);
    }

    let mut seen_speakers: Vec<String> = Vec::new();
    for segment in &segments {
        if !seen_speakers.contains(&segment.speaker) {
            seen_speakers.push(segment.speaker.clone());
        }
    }
    let participant_ids: Vec<String> = seen_speakers
        .iter()
        .enumerate()
        .map(|(index, _)| format!("PA{index}"))
        .collect();
    reassign_speakers(&mut chat_file, &segments, lang, &participant_ids);
    Ok(chat_file)
}

pub(crate) fn add_utterance_timing_inner(
    chat_file: &mut talkbank_model::model::ChatFile,
    asr_words_json: &str,
) -> PyResult<()> {
    use talkbank_model::alignment::helpers::TierDomain;

    let asr_words: Vec<AsrWordJson> = serde_json::from_str(asr_words_json).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Invalid asr_words JSON: {e}"))
    })?;

    if asr_words.is_empty() {
        return Ok(());
    }

    let extracted = crate::extract::extract_words(chat_file, TierDomain::Wor);
    let utterance_windows = collect_utterance_windows(chat_file);
    let mut utt_timings: std::collections::HashMap<usize, Vec<Option<(u64, u64)>>> = extracted
        .iter()
        .map(|utt| (utt.utterance_index.raw(), vec![None; utt.words.len()]))
        .collect();

    // Preferred lossless path: map ASR words carrying stable transcript IDs.
    // Any ASR words without usable IDs fall back to deterministic matching
    // against remaining words.
    let mut matched_ref_coords: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    let mut unmatched_asr_indices: Vec<usize> = Vec::new();
    let mut id_mapped = 0usize;
    let mut window_mapped = 0usize;
    let mut no_window_mapped = 0usize;
    let mut deterministic_unassigned = 0usize;
    let mut no_window_fallback_used = false;

    for (asr_idx, asr) in asr_words.iter().enumerate() {
        let Some(word_id) = asr.word_id.as_deref() else {
            unmatched_asr_indices.push(asr_idx);
            continue;
        };
        let Some((utt_idx, word_idx)) = parse_stable_word_id(word_id) else {
            tracing::warn!(
                word_id,
                "invalid ASR word_id format; expected u{{n}}:w{{n}}"
            );
            unmatched_asr_indices.push(asr_idx);
            continue;
        };
        if !matched_ref_coords.insert((utt_idx, word_idx)) {
            tracing::warn!(
                word_id,
                utterance_index = utt_idx,
                utterance_word_index = word_idx,
                "duplicate ASR word_id in payload; falling back to deterministic matching for duplicate entry"
            );
            unmatched_asr_indices.push(asr_idx);
            continue;
        }
        if let Some(timings) = utt_timings.get_mut(&utt_idx) {
            if word_idx < timings.len() {
                timings[word_idx] = Some((asr.start_ms, asr.end_ms));
                id_mapped += 1;
            } else {
                tracing::warn!(
                    word_id,
                    utterance_index = utt_idx,
                    utterance_word_index = word_idx,
                    "ASR word_id points past utterance word count"
                );
                matched_ref_coords.remove(&(utt_idx, word_idx));
                unmatched_asr_indices.push(asr_idx);
            }
        } else {
            tracing::warn!(
                word_id,
                utterance_index = utt_idx,
                "ASR word_id references unknown utterance"
            );
            matched_ref_coords.remove(&(utt_idx, word_idx));
            unmatched_asr_indices.push(asr_idx);
        }
    }

    if !unmatched_asr_indices.is_empty() {
        // First fallback pass: constrain unmatched ASR words to utterances whose
        // bullets uniquely overlap those word times. This avoids global-crossing
        // ties when utterance windows are available.
        let mut unmatched_after_window_pass: Vec<usize> = Vec::new();
        let mut asr_by_window_utt: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for &asr_idx in &unmatched_asr_indices {
            if let Some(utt_idx) =
                unique_overlapping_utterance(&asr_words[asr_idx], &utterance_windows)
            {
                asr_by_window_utt.entry(utt_idx).or_default().push(asr_idx);
            } else {
                unmatched_after_window_pass.push(asr_idx);
            }
        }

        let mut constrained_utts: Vec<usize> = asr_by_window_utt.keys().copied().collect();
        constrained_utts.sort_unstable();
        for utt_idx in constrained_utts {
            let Some(mut asr_indices) = asr_by_window_utt.remove(&utt_idx) else {
                continue;
            };
            if asr_indices.is_empty() {
                continue;
            }
            asr_indices.sort_unstable_by_key(|&idx| asr_words[idx].start_ms);

            let Some(utt) = extracted
                .iter()
                .find(|u| u.utterance_index.raw() == utt_idx)
            else {
                unmatched_after_window_pass.extend(asr_indices);
                continue;
            };

            let mut ref_keys: Vec<String> = Vec::new();
            let mut ref_coords: Vec<(usize, usize)> = Vec::new();
            for (word_idx, word) in utt.words.iter().enumerate() {
                let coord = (utt_idx, word_idx);
                if matched_ref_coords.contains(&coord) {
                    continue;
                }
                ref_keys.push(word.text.to_lowercase());
                ref_coords.push(coord);
            }

            if ref_keys.is_empty() {
                unmatched_after_window_pass.extend(asr_indices);
                continue;
            }

            let matched_asr = align_unmatched_monotonic(
                &asr_words,
                &asr_indices,
                &ref_keys,
                &ref_coords,
                &mut utt_timings,
                &mut matched_ref_coords,
            );
            window_mapped += matched_asr.len();
            for asr_idx in asr_indices {
                if !matched_asr.contains(&asr_idx) {
                    unmatched_after_window_pass.push(asr_idx);
                }
            }
        }

        // Second fallback pass: use deterministic global monotonic matching only
        // when no utterance windows exist.
        if !unmatched_after_window_pass.is_empty() {
            if utterance_windows.is_empty() {
                no_window_fallback_used = true;
                let mut ref_keys: Vec<String> = Vec::new();
                let mut ref_coords: Vec<(usize, usize)> = Vec::new();
                for utt in &extracted {
                    for (word_idx, word) in utt.words.iter().enumerate() {
                        let coord = (utt.utterance_index.raw(), word_idx);
                        if matched_ref_coords.contains(&coord) {
                            continue;
                        }
                        ref_keys.push(word.text.to_lowercase());
                        ref_coords.push(coord);
                    }
                }
                let matched_global = align_unmatched_monotonic(
                    &asr_words,
                    &unmatched_after_window_pass,
                    &ref_keys,
                    &ref_coords,
                    &mut utt_timings,
                    &mut matched_ref_coords,
                );
                no_window_mapped += matched_global.len();
                deterministic_unassigned += unmatched_after_window_pass
                    .len()
                    .saturating_sub(matched_global.len());
            } else {
                deterministic_unassigned += unmatched_after_window_pass.len();
                tracing::warn!(
                    unmatched_count = unmatched_after_window_pass.len(),
                    "skipping global fallback because utterance windows exist; leaving unmatched ASR words unassigned"
                );
            }
        }
    }

    tracing::info!(
        total_asr_words = asr_words.len(),
        id_mapped,
        window_mapped,
        no_window_mapped,
        deterministic_unassigned,
        no_window_fallback_used,
        "utr timing mapping counters"
    );

    let mut utt_idx = 0usize;
    for line in chat_file.lines.iter_mut() {
        let utt = match line {
            Line::Utterance(u) => u,
            _ => continue,
        };

        if let Some(timings) = utt_timings.get(&utt_idx) {
            let fa_timings: Vec<Option<crate::forced_alignment::WordTiming>> = timings
                .iter()
                .map(|t| {
                    t.map(|(s, e)| crate::forced_alignment::WordTiming {
                        start_ms: s,
                        end_ms: e,
                    })
                })
                .collect();
            let mut offset = 0;
            crate::forced_alignment::inject_timings_for_utterance(utt, &fa_timings, &mut offset);
            crate::forced_alignment::update_utterance_bullet(utt);
            crate::forced_alignment::add_wor_tier(utt);
        }

        utt_idx += 1;
    }

    Ok(())
}

fn parse_stable_word_id(word_id: &str) -> Option<(usize, usize)> {
    let (utt_part, word_part) = word_id.split_once(':')?;
    let utt_idx = utt_part.strip_prefix('u')?.parse::<usize>().ok()?;
    let word_idx = word_part.strip_prefix('w')?.parse::<usize>().ok()?;
    Some((utt_idx, word_idx))
}

fn collect_utterance_windows(
    chat_file: &talkbank_model::model::ChatFile,
) -> std::collections::HashMap<usize, (u64, u64)> {
    let mut windows = std::collections::HashMap::new();
    let mut utt_idx = 0usize;
    for line in &chat_file.lines {
        let Line::Utterance(utt) = line else {
            continue;
        };
        if let Some(bullet) = &utt.main.content.bullet {
            windows.insert(utt_idx, (bullet.timing.start_ms, bullet.timing.end_ms));
        }
        utt_idx += 1;
    }
    windows
}

fn unique_overlapping_utterance(
    asr: &AsrWordJson,
    utterance_windows: &std::collections::HashMap<usize, (u64, u64)>,
) -> Option<usize> {
    let asr_start = asr.start_ms;
    let asr_end = asr.end_ms;

    let mut matched: Option<usize> = None;
    for (&utt_idx, &(utt_start, utt_end)) in utterance_windows {
        if asr_end < utt_start || asr_start > utt_end {
            continue;
        }
        if matched.is_some() {
            return None;
        }
        matched = Some(utt_idx);
    }
    matched
}

fn align_unmatched_monotonic(
    asr_words: &[AsrWordJson],
    asr_indices: &[usize],
    ref_keys: &[String],
    ref_coords: &[(usize, usize)],
    utt_timings: &mut std::collections::HashMap<usize, Vec<Option<(u64, u64)>>>,
    matched_ref_coords: &mut std::collections::HashSet<(usize, usize)>,
) -> std::collections::HashSet<usize> {
    let mut matched_asr_indices: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    if asr_indices.is_empty() || ref_keys.is_empty() {
        return matched_asr_indices;
    }

    let mut asr_cursor = 0usize;
    for (reference_idx, ref_key) in ref_keys.iter().enumerate() {
        while asr_cursor < asr_indices.len() {
            let asr_idx = asr_indices[asr_cursor];
            if asr_words[asr_idx].word.to_lowercase() == *ref_key {
                break;
            }
            asr_cursor += 1;
        }
        if asr_cursor >= asr_indices.len() {
            break;
        }

        let asr_idx = asr_indices[asr_cursor];
        let asr = &asr_words[asr_idx];
        let (utt_idx, word_idx) = ref_coords[reference_idx];
        if let Some(timings) = utt_timings.get_mut(&utt_idx)
            && word_idx < timings.len()
            && timings[word_idx].is_none()
        {
            timings[word_idx] = Some((asr.start_ms, asr.end_ms));
            matched_ref_coords.insert((utt_idx, word_idx));
            matched_asr_indices.insert(asr_idx);
        }
        asr_cursor += 1;
    }
    matched_asr_indices
}
