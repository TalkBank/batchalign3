//! Incremental forced alignment processing.
//!
//! Compares a "before" file (with existing timings) against an "after" file
//! (user-edited) and only re-aligns FA groups that still need worker or cache
//! work after stable `%wor` timing is copied forward from the old file.
//!
//! Like full-file FA, this module now depends on the transport-neutral FA
//! worker adapter instead of assembling a concrete worker payload inline. That
//! keeps the incremental path and full-file path on the same migration path as
//! the worker protocol evolves from V1 payloads to V2 prepared artifacts.

use crate::api::DurationMs;
use crate::cache::CacheBackend;
use crate::error::ServerError;
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::types::results::FaResult;
use crate::types::traces::{FaGroupTrace, TimingTrace, ViolationTrace};
use batchalign_chat_ops::diff::UtteranceDelta;
use batchalign_chat_ops::diff::preserve::{TierKind, copy_dependent_tiers};
use batchalign_chat_ops::fa::{
    FaGroup, WordTiming, apply_fa_results, cache_key, collect_existing_fa_word_timings,
    group_utterances, refresh_existing_alignment_for_utterance,
};
use batchalign_chat_ops::parse::{is_dummy, is_no_align, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{CacheKey, ChatFile, Line, Utterance};
use tracing::{info, warn};

use super::transport::{FaWorkerBatch, FaWorkerTransport};
use super::{CACHE_TASK, collect_final_timings, process_fa};

/// Process a CHAT file through forced alignment incrementally.
///
/// Compares `before_text` (previous file with timings) against `after_text`
/// (user-edited version) and only re-aligns FA groups that contain changed
/// utterances. Unchanged groups preserve their existing timings.
///
/// Falls back to full processing if no "before" is available.
pub(crate) async fn process_fa_incremental(
    before_text: &str,
    after_text: &str,
    audio: &AudioContext<'_>,
    worker_lang: &crate::api::LanguageCode3,
    services: PipelineServices<'_>,
    fa_params: &FaParams,
    progress: Option<&ProgressSender>,
) -> Result<FaResult, ServerError> {
    use batchalign_chat_ops::diff::{DiffSummary, diff_chat};

    let (before_file, _) = parse_lenient(before_text);
    let (after_file, _) = parse_lenient(after_text);

    let deltas = diff_chat(&before_file, &after_file);
    let summary = DiffSummary::from_deltas(&deltas);

    info!(
        unchanged = summary.unchanged,
        words_changed = summary.words_changed,
        inserted = summary.inserted,
        deleted = summary.deleted,
        "Incremental FA diff"
    );

    // If there is no unchanged, speaker-only-changed, or timing-only region to
    // preserve from the previous file, the incremental path has nothing to
    // reuse and should fall back to the regular full-file align path.
    if summary.unchanged == 0 && summary.speaker_changed == 0 && summary.timing_only == 0 {
        return process_fa(
            after_text,
            audio,
            worker_lang,
            services,
            fa_params,
            progress,
        )
        .await;
    }

    // Group the "after" file's utterances
    let (mut chat_file, parse_errors) = parse_lenient(after_text);

    if is_dummy(&chat_file) || is_no_align(&chat_file) {
        return Ok(FaResult {
            chat_text: to_chat_string(&chat_file),
            groups: Vec::new(),
            pre_injection_timings: Vec::new(),
            timing_mode: fa_params.timing_mode,
            violations: Vec::new(),
        });
    }

    if let Err(errors) =
        validate_to_level(&chat_file, parse_errors.len(), ValidityLevel::MainTierValid)
    {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return Err(ServerError::Validation(format!(
            "align pre-validation failed: {}",
            msgs.join("; ")
        )));
    }

    let reusable_after_indices = reuse_stable_wor_timing_from_before(
        &before_file,
        &mut chat_file,
        &deltas,
        fa_params.wor_tier.should_write(),
    );

    let groups = group_utterances(
        &chat_file,
        fa_params.max_group_ms.0,
        audio.total_audio_ms.map(|ms| ms.0),
    );
    if groups.is_empty() {
        return Ok(FaResult {
            chat_text: to_chat_string(&chat_file),
            groups: Vec::new(),
            pre_injection_timings: Vec::new(),
            timing_mode: fa_params.timing_mode,
            violations: Vec::new(),
        });
    }

    // Determine which groups still need re-alignment after stable `%wor`
    // regions from the "before" file were copied into the edited file.
    let mut group_needs_realign: Vec<bool> = Vec::with_capacity(groups.len());
    let mut realign_count = 0usize;
    let mut reused_group_count = 0usize;
    for group in &groups {
        let needs = group
            .utterance_indices
            .iter()
            .any(|idx| !reusable_after_indices.contains(&idx.raw()));
        if needs {
            realign_count += 1;
        } else {
            reused_group_count += 1;
        }
        group_needs_realign.push(needs);
    }

    info!(
        total_groups = groups.len(),
        realign_groups = realign_count,
        reused_groups = reused_group_count,
        "Incremental FA: selective group re-alignment with stable %wor reuse"
    );

    // Build cache keys and timing storage for all groups
    let word_texts: Vec<Vec<String>> = groups
        .iter()
        .map(|g| g.words.iter().map(|w| w.text.clone()).collect())
        .collect();

    let cache_keys: Vec<CacheKey> = groups
        .iter()
        .zip(word_texts.iter())
        .map(|(g, words)| {
            cache_key(
                words,
                audio.audio_identity,
                g.audio_start_ms(),
                g.audio_end_ms(),
                fa_params.timing_mode,
                fa_params.engine,
            )
        })
        .collect();

    let mut all_timings: Vec<Option<Vec<Option<WordTiming>>>> = vec![None; groups.len()];

    // Reused groups already have current main-tier word timing in `chat_file`.
    // Everything else still needs a cache lookup or worker call.
    let key_strings: Vec<String> = cache_keys.iter().map(|k| k.as_str().to_string()).collect();
    let cached = if fa_params.cache_policy.should_skip() {
        std::collections::HashMap::new()
    } else {
        match services
            .cache
            .get_batch(&key_strings, CACHE_TASK.as_str(), services.engine_version)
            .await
        {
            Ok(map) => map,
            Err(e) => {
                warn!(error = %e, "FA cache batch lookup failed");
                std::collections::HashMap::new()
            }
        }
    };

    // Populate reused groups and cache hits.
    let mut miss_indices: Vec<usize> = Vec::new();
    for (i, key) in cache_keys.iter().enumerate() {
        if !group_needs_realign[i]
            && let Some(timings) = collect_preserved_group_timings(&chat_file, &groups[i])
        {
            all_timings[i] = Some(timings);
            continue;
        }

        if let Some(cached_data) = cached.get(key.as_str()) {
            match serde_json::from_value::<Vec<Option<WordTiming>>>(cached_data.clone()) {
                Ok(timings) => {
                    all_timings[i] = Some(timings);
                    continue;
                }
                Err(e) => {
                    warn!(error = %e, group = i, "Failed to deserialize cached FA timings");
                }
            }
        }
        miss_indices.push(i);
    }

    let reused_or_cached_groups = groups.len() - miss_indices.len();
    if reused_or_cached_groups > 0 || !miss_indices.is_empty() {
        info!(
            reused_or_cached = reused_or_cached_groups,
            misses = miss_indices.len(),
            "FA incremental partition"
        );
    }

    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::Aligning,
            Some(reused_or_cached_groups as i64),
            Some(groups.len() as i64),
        ));
    }

    let transport = FaWorkerTransport::production(services);

    // Send miss groups through the shared FA worker transport adapter.
    if !miss_indices.is_empty() {
        let parsed_results = transport
            .infer_groups(FaWorkerBatch {
                word_texts: &word_texts,
                groups: &groups,
                miss_indices: &miss_indices,
                audio_path: audio.audio_path,
                worker_lang: worker_lang.into(),
                engine: fa_params.engine,
                timing_mode: fa_params.timing_mode,
            })
            .await?;

        for (parsed_idx, parsed_result) in parsed_results.iter().enumerate() {
            let miss_idx = parsed_result.group_index;
            let timings = parsed_result.timings.clone();

            let ba_version = env!("CARGO_PKG_VERSION");
            if let Ok(cache_data) = serde_json::to_value(&timings)
                && let Err(error) = services
                    .cache
                    .put_batch(
                        &[(cache_keys[miss_idx].as_str().to_string(), cache_data)],
                        CACHE_TASK.as_str(),
                        services.engine_version,
                        ba_version,
                    )
                    .await
            {
                warn!(error = %error, "Failed to cache FA result (non-fatal)");
            }

            all_timings[miss_idx] = Some(timings);

            if let Some(tx) = progress {
                let done = reused_or_cached_groups + parsed_idx + 1;
                let _ = tx.send(ProgressUpdate::new(
                    FileStage::Aligning,
                    Some(done as i64),
                    Some(groups.len() as i64),
                ));
            }
        }
    }

    // Apply all results
    let final_timings = collect_final_timings(all_timings, "incremental forced alignment")?;

    let pre_injection_timings: Vec<Vec<Option<TimingTrace>>> = final_timings
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|t| {
                    t.as_ref().map(|wt| TimingTrace {
                        start_ms: wt.start_ms as i64,
                        end_ms: wt.end_ms as i64,
                    })
                })
                .collect()
        })
        .collect();

    apply_fa_results(
        &mut chat_file,
        &groups,
        &final_timings,
        fa_params.timing_mode,
        fa_params.wor_tier.should_write(),
    );

    let violations = if let Err(errors) = validate_output(&chat_file, "align") {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        warn!(errors = ?msgs, "align post-validation warnings (non-fatal)");
        errors
            .iter()
            .map(|e| ViolationTrace {
                code: format!("L{}", e.level as u8),
                message: e.message.clone(),
                utterance_index: None,
            })
            .collect()
    } else {
        Vec::new()
    };

    let group_traces: Vec<FaGroupTrace> = groups
        .iter()
        .map(|g| FaGroupTrace {
            audio_start_ms: DurationMs(g.audio_start_ms()),
            audio_end_ms: DurationMs(g.audio_end_ms()),
            utterance_indices: g.utterance_indices.iter().map(|idx| idx.0).collect(),
            words: g.words.iter().map(|w| w.text.clone()).collect(),
        })
        .collect();

    Ok(FaResult {
        chat_text: to_chat_string(&chat_file),
        groups: group_traces,
        pre_injection_timings,
        timing_mode: fa_params.timing_mode,
        violations,
    })
}

/// Copy reusable `%wor` timing from the "before" file into the edited file.
///
/// Only utterances whose words are unchanged are candidates. That includes
/// plain unchanged utterances, speaker-only changes, and timing-only edits
/// where a rerun should restore timing from the durable `%wor` layer instead of
/// trusting the edited utterance bullet. Each reused utterance receives the
/// `%wor` tier from the "before" file and is then refreshed back onto the main
/// tier so later grouping sees current utterance bullets and word timings.
fn reuse_stable_wor_timing_from_before(
    before_file: &ChatFile,
    after_file: &mut ChatFile,
    deltas: &[UtteranceDelta],
    write_wor: bool,
) -> std::collections::HashSet<usize> {
    let mut reused = std::collections::HashSet::new();

    for delta in deltas {
        let (before_idx, after_idx) = match delta {
            UtteranceDelta::Unchanged {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::TimingOnly {
                before_idx,
                after_idx,
            }
            | UtteranceDelta::SpeakerChanged {
                before_idx,
                after_idx,
            } => (*before_idx, *after_idx),
            _ => continue,
        };

        copy_dependent_tiers(
            before_file,
            before_idx,
            after_file,
            after_idx,
            &[TierKind::Wor],
        );

        let Some(utterance) = get_utterance_mut(after_file, after_idx.raw()) else {
            continue;
        };
        if refresh_existing_alignment_for_utterance(utterance, write_wor) {
            reused.insert(after_idx.raw());
        }
    }

    reused
}

/// Collect current timings for a preserved FA group from the CHAT AST.
///
/// The caller should use this only for groups whose utterances have already
/// been refreshed from stable `%wor` timing. The returned vector matches the
/// same word order used by FA extraction and injection.
pub(super) fn collect_preserved_group_timings(
    chat_file: &ChatFile,
    group: &FaGroup,
) -> Option<Vec<Option<WordTiming>>> {
    let mut timings = Vec::new();

    for utt_idx in &group.utterance_indices {
        let utterance = get_utterance(chat_file, utt_idx.raw())?;
        timings.extend(collect_existing_fa_word_timings(utterance));
    }

    if timings.len() != group.words.len() {
        return None;
    }

    Some(timings)
}

/// Borrow one utterance immutably by utterance ordinal.
pub(super) fn get_utterance(chat_file: &ChatFile, idx: usize) -> Option<&Utterance> {
    let mut current = 0usize;
    for line in &chat_file.lines {
        if let Line::Utterance(utterance) = line {
            if current == idx {
                return Some(utterance);
            }
            current += 1;
        }
    }
    None
}

/// Borrow one utterance mutably by utterance ordinal.
fn get_utterance_mut(chat_file: &mut ChatFile, idx: usize) -> Option<&mut Utterance> {
    let mut current = 0usize;
    for line in &mut chat_file.lines {
        if let Line::Utterance(utterance) = line {
            if current == idx {
                return Some(utterance);
            }
            current += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_chat_ops::diff::diff_chat;

    fn parse_chat(text: &str) -> ChatFile {
        batchalign_chat_ops::parse::parse_lenient(text).0
    }

    fn chat_with_wor(words0: &str, words1: &str) -> String {
        format!(
            "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tCHI Target_Child\n@ID:\teng|test|CHI|||||Target_Child|||\n*CHI:\t{words0}\n%wor:\thello \u{15}100_500\u{15} world \u{15}600_1000\u{15} .\n*CHI:\t{words1}\n%wor:\tgoodbye \u{15}1500_2000\u{15} .\n@End\n"
        )
    }

    #[test]
    fn reuse_stable_wor_timing_from_before_only_marks_unchanged_utterances() {
        let before = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        let mut after = parse_chat(&chat_with_wor("hello world .", "farewell ."));
        let deltas = diff_chat(&before, &after);

        let reused = reuse_stable_wor_timing_from_before(&before, &mut after, &deltas, true);
        assert!(reused.contains(&0));
        assert!(!reused.contains(&1));

        let utt0 = get_utterance(&after, 0).expect("missing utterance 0");
        assert_eq!(collect_existing_fa_word_timings(utt0).len(), 2);
        assert!(utt0.main.content.bullet.is_some());
    }

    #[test]
    fn collect_preserved_group_timings_reads_refreshed_main_tier_timing() {
        let before = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        let mut after = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        let deltas = diff_chat(&before, &after);
        let reused = reuse_stable_wor_timing_from_before(&before, &mut after, &deltas, true);
        assert_eq!(reused.len(), 2);

        let groups = group_utterances(&after, 20_000, Some(4_000));
        let timings = collect_preserved_group_timings(&after, &groups[0])
            .expect("group timings should exist");
        assert_eq!(timings.len(), groups[0].words.len());
        assert!(timings.iter().all(|timing| timing.is_some()));
    }

    #[test]
    fn reuse_stable_wor_timing_from_before_marks_timing_only_utterances() {
        let mut before = parse_chat(&chat_with_wor("hello world .", "goodbye ."));
        batchalign_chat_ops::fa::refresh_existing_alignment(&mut before, true);
        let before_text = batchalign_chat_ops::serialize::to_chat_string(&before);
        let before = parse_chat(&before_text);
        let mut after = parse_chat(&before_text);

        let utt0 = get_utterance_mut(&mut after, 0).expect("missing utterance 0");
        utt0.main.content.bullet = None;

        let deltas = diff_chat(&before, &after);
        assert!(matches!(deltas[0], UtteranceDelta::TimingOnly { .. }));

        let reused = reuse_stable_wor_timing_from_before(&before, &mut after, &deltas, true);
        assert!(
            reused.contains(&0),
            "timing-only utterance should be reused"
        );

        let utt0 = get_utterance(&after, 0).expect("missing utterance 0");
        assert!(utt0.main.content.bullet.is_some());
        assert!(
            collect_existing_fa_word_timings(utt0)
                .iter()
                .all(|timing| timing.is_some())
        );
    }
}
