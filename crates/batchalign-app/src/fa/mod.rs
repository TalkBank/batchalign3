//! Server-side forced alignment orchestrator.
//!
//! Owns the full CHAT lifecycle for FA jobs:
//! parse → group → cache check → infer (audio chunks) → DP-align → inject →
//! postprocess → %wor → monotonicity/E704 → serialize.
//!
//! # Call path
//!
//! `batchalign-cli`/API submission
//! → `runner::dispatch_fa_infer`
//! → [`process_fa`]
//! → `batchalign_chat_ops::fa::{group_utterances, parse_fa_response, apply_fa_results}`
//! → FA worker transport adapter
//! → validation + serialization.
//!
//! # Key differences from morphosyntax/utseg/translate/coref
//!
//! - **Per-file, not cross-file**: Each file has its own audio, so no cross-file batching.
//! - **Multiple groups per file**: Utterances are grouped by time window; each group is one infer item.
//! - **Audio access**: Workers need the audio file path and time range, not just text.
//! - **DP alignment in Rust**: Model output is aligned to transcript words via Hirschberg.
//!
//! # Invariants for contributors
//!
//! - FA worker timestamps are chunk-relative; `parse_fa_response` must convert
//!   them to file-absolute ms with `audio_start_ms`.
//! - `apply_fa_results` ordering is load-bearing:
//!   inject → postprocess → utterance bullet update → `%wor` generation
//!   → monotonicity (E362) → same-speaker overlap enforcement (E704).
//! - Cache keys must include audio identity + time window + text + timing mode
//!   + engine; changing dimensions changes cache compatibility.

mod transport;

use crate::cache::CacheBackend;
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use batchalign_chat_ops::fa::{
    WordTiming, apply_fa_results, cache_key, find_reusable_utterance_indices, group_utterances,
    has_reusable_wor_timing, refresh_existing_alignment, refresh_reusable_utterances,
};
use batchalign_chat_ops::parse::{is_dummy, is_no_align, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{CacheKey, CacheTaskName};
use tracing::{info, warn};

use crate::api::{ChatText, DurationMs};
use crate::error::ServerError;
use crate::runner::util::{FileStage, ProgressSender, ProgressUpdate};
use crate::types::results::FaResult;
use crate::types::traces::{FaGroupTrace, TimingTrace, ViolationTrace};
use crate::workflow::PerFileWorkflow;
use crate::workflow::fa::{ForcedAlignmentWorkflow, ForcedAlignmentWorkflowRequest};
use transport::{FaWorkerBatch, FaWorkerTransport};

/// Cache task name for FA results.
const CACHE_TASK: CacheTaskName = CacheTaskName::ForcedAlignment;

pub(super) fn collect_final_timings(
    all_timings: Vec<Option<Vec<Option<WordTiming>>>>,
    context: &str,
) -> Result<Vec<Vec<Option<WordTiming>>>, ServerError> {
    let missing_groups: Vec<usize> = all_timings
        .iter()
        .enumerate()
        .filter_map(|(index, timings)| timings.is_none().then_some(index))
        .collect();
    if !missing_groups.is_empty() {
        return Err(ServerError::Validation(format!(
            "{context} completed without timings for group(s): {missing_groups:?}"
        )));
    }

    // Safety: the None check above returned Err for any missing groups,
    // so all remaining elements are guaranteed Some.
    Ok(all_timings.into_iter().flatten().collect())
}

// ---------------------------------------------------------------------------
// Per-file FA processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the forced alignment pipeline.
///
/// Returns a structured [`FaResult`] containing the serialized CHAT text,
/// group info, timing data, and validation results.  The caller decides
/// which parts to persist (file output, trace cache, etc.).
///
/// Algorithm outline:
/// 1. Parse leniently and run pre-validation (`MainTierValid`).
/// 2. Group utterances into FA windows.
/// 3. Resolve cache hits/misses per group.
/// 4. Send miss groups through the FA worker transport adapter.
/// 5. Parse responses and align to transcript words in Rust.
/// 6. Apply timings + postprocessing (`apply_fa_results`).
/// 7. Run full post-validation and serialize.
pub(crate) async fn process_fa(
    chat_text: &str,
    audio: &AudioContext<'_>,
    worker_lang: &crate::api::LanguageCode3,
    services: PipelineServices<'_>,
    fa_params: &FaParams,
    progress: Option<&ProgressSender>,
) -> Result<FaResult, ServerError> {
    ForcedAlignmentWorkflow
        .run(ForcedAlignmentWorkflowRequest {
            chat_text: ChatText::from(chat_text),
            audio,
            worker_lang,
            services,
            params: fa_params,
            progress,
        })
        .await
}

pub(crate) async fn run_fa_impl(
    chat_text: &str,
    audio: &AudioContext<'_>,
    worker_lang: &crate::api::LanguageCode3,
    services: PipelineServices<'_>,
    fa_params: &FaParams,
    progress: Option<&ProgressSender>,
) -> Result<FaResult, ServerError> {
    // 1. Parse
    let (mut chat_file, parse_errors) = parse_lenient(chat_text);
    if !parse_errors.is_empty() {
        warn!(
            num_errors = parse_errors.len(),
            "Parse errors in FA input (continuing with recovery)"
        );
    }

    // 1b. Skip dummy files
    if is_dummy(&chat_file) {
        return Ok(FaResult {
            chat_text: to_chat_string(&chat_file),
            groups: Vec::new(),
            pre_injection_timings: Vec::new(),
            timing_mode: fa_params.timing_mode,
            violations: Vec::new(),
        });
    }

    // 1c. Skip files with @Options: NoAlign
    if is_no_align(&chat_file) {
        return Ok(FaResult {
            chat_text: to_chat_string(&chat_file),
            groups: Vec::new(),
            pre_injection_timings: Vec::new(),
            timing_mode: fa_params.timing_mode,
            violations: Vec::new(),
        });
    }

    // 1d. Pre-validation gate (L2: MainTierValid)
    if let Err(errors) =
        validate_to_level(&chat_file, parse_errors.len(), ValidityLevel::MainTierValid)
    {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return Err(ServerError::Validation(format!(
            "align pre-validation failed: {}",
            msgs.join("; ")
        )));
    }

    // 1e. Cheap rerun path: if the file already has complete, reusable `%wor`
    // timing, rebuild main-tier bullets and optionally regenerate `%wor`
    // without sending audio back through FA.
    if has_reusable_wor_timing(&chat_file) {
        info!("FA fast path: reusing existing %wor timing");
        refresh_existing_alignment(&mut chat_file, fa_params.wor_tier.should_write());
        return Ok(FaResult {
            chat_text: to_chat_string(&chat_file),
            groups: Vec::new(),
            pre_injection_timings: Vec::new(),
            timing_mode: fa_params.timing_mode,
            violations: Vec::new(),
        });
    }

    // 1f. Per-utterance partial reuse: when some (but not all) utterances have
    // clean %wor, refresh those and track them so their FA groups can be skipped.
    let reusable_indices = find_reusable_utterance_indices(&chat_file);
    if !reusable_indices.is_empty() {
        info!(
            reusable = reusable_indices.len(),
            "FA partial reuse: refreshing utterances with clean %wor"
        );
        refresh_reusable_utterances(
            &mut chat_file,
            &reusable_indices,
            fa_params.wor_tier.should_write(),
        );
    }

    // 2. Group utterances
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

    info!(
        num_groups = groups.len(),
        total_words = groups.iter().map(|g| g.words.len()).sum::<usize>(),
        "FA grouping complete"
    );

    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::CheckingCache,
            Some(0),
            Some(groups.len() as i64),
        ));
    }

    // 3. For each group: compute cache key, check cache
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

    // 4. Cache lookup
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
                warn!(error = %e, "FA cache batch lookup failed (treating all as misses)");
                std::collections::HashMap::new()
            }
        }
    };

    // 5. Partition into reused (from %wor), cache hits, and misses
    let mut all_timings: Vec<Option<Vec<Option<WordTiming>>>> = vec![None; groups.len()];
    let mut miss_indices: Vec<usize> = Vec::new();
    let mut reused_group_count = 0usize;

    for (i, key) in cache_keys.iter().enumerate() {
        // Tier 1: group fully reusable from %wor (all utterances have clean timing)
        if !reusable_indices.is_empty()
            && groups[i]
                .utterance_indices
                .iter()
                .all(|idx| reusable_indices.contains(&idx.raw()))
            && let Some(timings) =
                incremental::collect_preserved_group_timings(&chat_file, &groups[i])
        {
            all_timings[i] = Some(timings);
            reused_group_count += 1;
            continue;
        }

        // Tier 2: cache hit
        if let Some(cached_data) = cached.get(key.as_str()) {
            match serde_json::from_value::<Vec<Option<WordTiming>>>(cached_data.clone()) {
                Ok(timings) => {
                    all_timings[i] = Some(timings);
                    continue;
                }
                Err(e) => {
                    warn!(error = %e, group = i, "Failed to deserialize cached FA timings (re-computing)");
                }
            }
        }

        // Tier 3: cache miss
        miss_indices.push(i);
    }

    let cache_hits = groups.len() - miss_indices.len() - reused_group_count;
    let reused_or_cached_groups = reused_group_count + cache_hits;
    if cache_hits > 0 || reused_group_count > 0 {
        info!(
            reused = reused_group_count,
            cache_hits = cache_hits,
            misses = miss_indices.len(),
            "FA partition (reused from %wor / cache hits / misses)"
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

    // 6. Dispatch miss groups through the FA worker transport adapter
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

            // Cache the result
            let ba_version = env!("CARGO_PKG_VERSION");
            if let Ok(cache_data) = serde_json::to_value(&timings)
                && let Err(e) = services
                    .cache
                    .put_batch(
                        &[(cache_keys[miss_idx].as_str().to_string(), cache_data)],
                        CACHE_TASK.as_str(),
                        services.engine_version,
                        ba_version,
                    )
                    .await
            {
                warn!(error = %e, "Failed to cache FA result (non-fatal)");
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

    // 8. Apply all results
    if let Some(tx) = progress {
        let _ = tx.send(ProgressUpdate::new(
            FileStage::ApplyingResults,
            Some(groups.len() as i64),
            Some(groups.len() as i64),
        ));
    }

    let final_timings = collect_final_timings(all_timings, "forced alignment")?;

    // Snapshot pre-injection timings (before apply_fa_results consumes them)
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

    // 9. Post-validation check (warn only — cross-speaker overlap is normal in
    //    conversation data, and enforce_monotonicity/E704 already stripped
    //    same-speaker issues).
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

    // 10. Build group traces
    let group_traces: Vec<FaGroupTrace> = groups
        .iter()
        .map(|g| FaGroupTrace {
            audio_start_ms: DurationMs(g.audio_start_ms()),
            audio_end_ms: DurationMs(g.audio_end_ms()),
            utterance_indices: g.utterance_indices.iter().map(|idx| idx.0).collect(),
            words: g.words.iter().map(|w| w.text.clone()).collect(),
        })
        .collect();

    // 11. Serialize and return structured result
    Ok(FaResult {
        chat_text: to_chat_string(&chat_file),
        groups: group_traces,
        pre_injection_timings,
        timing_mode: fa_params.timing_mode,
        violations,
    })
}

mod incremental;
pub(crate) use incremental::process_fa_incremental;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_task_name_is_stable() {
        assert_eq!(CACHE_TASK.as_str(), "forced_alignment");
    }

    #[test]
    fn collect_final_timings_rejects_missing_groups() {
        let error = collect_final_timings(vec![Some(Vec::new()), None], "forced alignment")
            .expect_err("missing timing groups should fail");
        assert!(
            error
                .to_string()
                .contains("completed without timings for group(s): [1]")
        );
    }
}
