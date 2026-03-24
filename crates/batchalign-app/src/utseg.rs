//! Server-side utterance segmentation orchestrator.
//!
//! Owns the full CHAT lifecycle for utseg jobs:
//! parse → collect payloads → cache check → infer → apply splits → cache put → serialize.
//!
//! Python workers receive only `(words, text) → UtsegResponse` via the infer protocol —
//! pure Stanza constituency parsing with zero CHAT awareness.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::api::{ChatText, EngineVersion, LanguageCode3};
use crate::cache::{CacheBackend, UtteranceCache};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_utseg_request_v2};
use crate::worker::text_result_v2::parse_utseg_result_v2;
use batchalign_chat_ops::parse::{is_dummy, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::utseg::{
    UtsegBatchItem, UtsegResponse, apply_utseg_results, cache_key, collect_utseg_payloads,
};
use batchalign_chat_ops::utseg_compute;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{CacheKey, CacheTaskName, ChatFile, LanguageCode};
use tracing::{info, warn};

use crate::error::ServerError;
use crate::infer_retry::dispatch_execute_v2_with_retry;
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::pipeline::text_infer::{CachedTextPipelineHooks, run_cached_text_pipeline};
use crate::workflow::text_batch::{
    TextBatchFileInput, TextBatchFileResult, TextBatchFileResults, TextBatchOperation,
    TextBatchWorkflow, TextBatchWorkflowRequest, TextPerFileWorkflowRequest,
};

/// Cache task name — matches Python's `utterance_segmentation`.
const CACHE_TASK: CacheTaskName = CacheTaskName::UtteranceSegmentation;

/// Command-specific parameters for the utseg workflow family.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UtsegWorkflowParams {
    /// Cache policy used for utterance segmentation.
    pub cache_policy: CachePolicy,
}

/// Typed workflow operation for utseg.
pub(crate) struct UtsegOperation;

/// Trait-oriented workflow wrapper for utseg.
pub(crate) type UtsegWorkflow = TextBatchWorkflow<UtsegOperation>;

#[async_trait]
impl TextBatchOperation for UtsegOperation {
    type Shared<'a>
        = PipelineServices<'a>
    where
        Self: 'a;

    type Params<'a>
        = UtsegWorkflowParams
    where
        Self: 'a;

    async fn run_single(
        chat_text: ChatText<'_>,
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        params: Self::Params<'_>,
    ) -> Result<String, ServerError> {
        run_utseg_impl(
            chat_text.as_ref(),
            lang,
            shared.pool,
            shared.cache,
            shared.engine_version,
            params.cache_policy,
        )
        .await
    }

    async fn run_batch(
        files: &[TextBatchFileInput],
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        params: Self::Params<'_>,
    ) -> TextBatchFileResults {
        run_utseg_batch_impl(
            files,
            lang,
            shared.pool,
            shared.cache,
            shared.engine_version,
            params.cache_policy,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Per-file utseg processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the utseg pipeline.
///
/// Returns the serialized CHAT text with utterances split as needed.
pub async fn process_utseg(
    chat_text: &str,
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    cache_policy: CachePolicy,
) -> Result<String, ServerError> {
    UtsegWorkflow::new()
        .run_per_file(TextPerFileWorkflowRequest {
            chat_text: ChatText::from(chat_text),
            lang,
            shared: PipelineServices::new(pool, cache, engine_version),
            params: UtsegWorkflowParams { cache_policy },
        })
        .await
}

// ---------------------------------------------------------------------------
// Cross-file batch utseg processing
// ---------------------------------------------------------------------------

/// Process multiple CHAT files, pooling all cache misses into a single
/// `batch_infer` call for maximum throughput.
///
/// Returns `(filename, Ok(output_text) | Err(error_msg))` for each file.
pub(crate) async fn process_utseg_batch(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    cache_policy: CachePolicy,
) -> TextBatchFileResults {
    UtsegWorkflow::new()
        .run_batch_files(TextBatchWorkflowRequest {
            files,
            lang,
            shared: PipelineServices::new(pool, cache, engine_version),
            params: UtsegWorkflowParams { cache_policy },
        })
        .await
}

async fn run_utseg_impl(
    chat_text: &str,
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    cache_policy: CachePolicy,
) -> Result<String, ServerError> {
    run_cached_text_pipeline(
        chat_text,
        lang,
        PipelineServices::new(pool, cache, engine_version),
        cache_policy,
        CachedTextPipelineHooks {
            command: "utseg",
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_utseg_payloads,
            partition: partition_single_file,
            infer: infer_single_file,
            integrate: integrate_assignments,
            cache_put: cache_put_single_file,
            apply: apply_utseg_results,
        },
    )
    .await
}

async fn run_utseg_batch_impl(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    cache_policy: CachePolicy,
) -> TextBatchFileResults {
    let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
        .expect("tree-sitter CHAT grammar must load");
    let mut results: TextBatchFileResults = Vec::with_capacity(files.len());

    // 1. Parse all files and collect payloads
    let mut parsed_files: Vec<ChatFile> = Vec::with_capacity(files.len());
    let mut parse_error_counts: Vec<usize> = Vec::with_capacity(files.len());
    for file in files {
        let filename = file.filename.as_ref();
        let (chat_file, parse_errors) = parse_lenient(&parser, file.chat_text.as_ref());
        if !parse_errors.is_empty() {
            warn!(
                filename = %filename,
                num_errors = parse_errors.len(),
                "Parse errors (continuing with recovery)"
            );
        }
        parse_error_counts.push(parse_errors.len());
        parsed_files.push(chat_file);
    }

    // 2. Collect payloads from each file, tracking provenance
    struct FileMissInfo {
        items: Vec<(usize, UtsegBatchItem)>,
        keys: Vec<CacheKey>,
        global_start: usize,
    }

    let mut all_misses: Vec<(usize, UtsegBatchItem)> = Vec::new();
    let mut per_file_info: Vec<Option<FileMissInfo>> = Vec::with_capacity(files.len());
    let mut per_file_cached: Vec<HashMap<usize, Vec<usize>>> = Vec::with_capacity(files.len());
    let mut validation_errors: Vec<Option<String>> = vec![None; files.len()];

    for (file_idx, parsed_file) in parsed_files.iter().enumerate() {
        // Skip dummy files — they pass through unchanged
        if is_dummy(parsed_file) {
            per_file_info.push(None);
            per_file_cached.push(HashMap::new());
            continue;
        }

        // Pre-validation gate (L1: StructurallyComplete)
        if let Err(errors) = validate_to_level(
            parsed_file,
            parse_error_counts[file_idx],
            ValidityLevel::StructurallyComplete,
        ) {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            let error_summary = format!("utseg pre-validation failed: {}", msgs.join("; "));
            warn!(
                filename = %files[file_idx].filename,
                errors = %error_summary,
                chat_text = %files[file_idx].chat_text,
                "utseg pre-validation failed — dumping CHAT for diagnosis"
            );
            validation_errors[file_idx] = Some(error_summary);
            per_file_info.push(None);
            per_file_cached.push(HashMap::new());
            continue;
        }

        let batch_items = collect_utseg_payloads(parsed_file);

        if batch_items.is_empty() {
            per_file_info.push(None);
            per_file_cached.push(HashMap::new());
            continue;
        }

        // Cache lookup
        let lang_code = LanguageCode::new(lang.as_ref());
        let (cached_assignments, miss_keys, misses) = if cache_policy.should_skip() {
            let keys: Vec<CacheKey> = batch_items
                .iter()
                .map(|(_, item)| cache_key(&item.words, &lang_code))
                .collect();
            (HashMap::new(), keys, batch_items)
        } else {
            partition_by_cache(&batch_items, &lang_code, cache, engine_version).await
        };

        per_file_cached.push(cached_assignments);

        if misses.is_empty() {
            per_file_info.push(None);
        } else {
            let global_start = all_misses.len();
            per_file_info.push(Some(FileMissInfo {
                items: misses.clone(),
                keys: miss_keys,
                global_start,
            }));
            all_misses.extend(misses);
        }
    }

    // 3. Single batch_infer across all files
    let all_utseg_responses = if all_misses.is_empty() {
        Vec::new()
    } else {
        match infer_batch(pool, &all_misses, lang).await {
            Ok(responses) => responses,
            Err(e) => {
                warn!(error = %e, "Batch infer failed for all files");
                for (file_idx, file) in files.iter().enumerate() {
                    if per_file_info
                        .get(file_idx)
                        .and_then(|f| f.as_ref())
                        .is_some()
                    {
                        results.push(TextBatchFileResult::err(
                            file.filename.clone(),
                            format!("Batch infer failed: {e}"),
                        ));
                    } else {
                        // Apply cached assignments and serialize
                        let chat_file = &mut parsed_files[file_idx];
                        let cached = &per_file_cached[file_idx];
                        if !cached.is_empty() {
                            apply_utseg_results(chat_file, cached);
                        }
                        results.push(TextBatchFileResult::ok(file.filename.clone(), to_chat_string(chat_file)));
                    }
                }
                return results;
            }
        }
    };

    // 4. Distribute responses back to files and apply
    for (file_idx, file) in files.iter().enumerate() {
        let filename = file.filename.as_ref();
        // Skip files that failed pre-validation
        if let Some(ref err) = validation_errors[file_idx] {
            results.push(TextBatchFileResult::err(file.filename.clone(), err.clone()));
            continue;
        }

        let chat_file = &mut parsed_files[file_idx];
        let mut assignment_map = std::mem::take(&mut per_file_cached[file_idx]);

        if let Some(ref fm) = per_file_info[file_idx] {
            let global_start = fm.global_start;
            let count = fm.items.len();

            let file_responses: Vec<UtsegResponse> =
                all_utseg_responses[global_start..global_start + count].to_vec();

            for ((utt_ordinal, item), resp) in fm.items.iter().zip(file_responses.iter()) {
                if resp.assignments.len() == item.words.len() {
                    assignment_map.insert(*utt_ordinal, resp.assignments.clone());
                } else {
                    warn!(
                        filename = %filename,
                        utterance = utt_ordinal,
                        expected = item.words.len(),
                        got = resp.assignments.len(),
                        "utseg assignment length mismatch, keeping original"
                    );
                }
            }

            // Cache store for this file's misses
            cache_put_entries(
                cache,
                &fm.keys,
                &fm.items,
                &file_responses,
                lang,
                engine_version,
            )
            .await;
        }

        // Apply all assignments (cached + inferred)
        if !assignment_map.is_empty() {
            apply_utseg_results(chat_file, &assignment_map);
        }

        // Post-validation check (warn only — always serialize output for debugging).
        if let Err(errors) = validate_output(chat_file, "utseg") {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            warn!(filename = %filename, errors = ?msgs, "utseg post-validation warnings (non-fatal)");
        }

        results.push(TextBatchFileResult::ok(file.filename.clone(), to_chat_string(chat_file)));
    }

    results
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Result of partitioning batch items by cache.
type PartitionResult = (
    HashMap<usize, Vec<usize>>,
    Vec<CacheKey>,
    Vec<(usize, UtsegBatchItem)>,
);

/// Partition batch items into cache hits and misses.
///
/// Returns `(cached_assignments, miss_keys, misses)`.
async fn partition_by_cache(
    batch_items: &[(usize, UtsegBatchItem)],
    lang: &LanguageCode,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
) -> PartitionResult {
    let keys: Vec<CacheKey> = batch_items
        .iter()
        .map(|(_, item)| cache_key(&item.words, lang))
        .collect();

    let key_strings: Vec<String> = keys.iter().map(|k| k.as_str().to_string()).collect();
    let cached = match cache
        .get_batch(&key_strings, CACHE_TASK.as_str(), engine_version)
        .await
    {
        Ok(map) => map,
        Err(e) => {
            warn!(error = %e, "Cache batch lookup failed (treating all as misses)");
            HashMap::new()
        }
    };

    let mut hits = HashMap::new();
    let mut miss_keys = Vec::new();
    let mut misses = Vec::new();

    for ((utt_ordinal, item), key) in batch_items.iter().zip(keys.into_iter()) {
        if let Some(cached_data) = cached.get(key.as_str()) {
            // Extract assignments from cached JSON
            if let Some(assignments) = cached_data
                .get("assignments")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_u64().map(|n| n as usize))
                        .collect::<Vec<_>>()
                })
                && assignments.len() == item.words.len()
            {
                hits.insert(*utt_ordinal, assignments);
                continue;
            }
            // Cache entry was invalid/mismatched — treat as miss
            miss_keys.push(key);
            misses.push((*utt_ordinal, item.clone()));
        } else {
            miss_keys.push(key);
            misses.push((*utt_ordinal, item.clone()));
        }
    }

    (hits, miss_keys, misses)
}

#[allow(clippy::type_complexity)]
fn partition_single_file<'a>(
    batch_items: &'a [(usize, UtsegBatchItem)],
    lang: &'a LanguageCode3,
    cache: &'a UtteranceCache,
    engine_version: &'a EngineVersion,
    cache_policy: CachePolicy,
) -> std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = (
                    HashMap<usize, Vec<usize>>,
                    Vec<CacheKey>,
                    Vec<(usize, UtsegBatchItem)>,
                ),
            > + Send
            + 'a,
    >,
> {
    Box::pin(async move {
        let lang_code = LanguageCode::new(lang.as_ref());
        if cache_policy.should_skip() {
            let keys: Vec<CacheKey> = batch_items
                .iter()
                .map(|(_, item)| cache_key(&item.words, &lang_code))
                .collect();
            (HashMap::new(), keys, batch_items.to_vec())
        } else {
            partition_by_cache(batch_items, &lang_code, cache, engine_version).await
        }
    })
}

/// Send batch items to a worker for constituency inference via batched
/// `execute_v2`.
async fn infer_batch(
    pool: &WorkerPool,
    items: &[(usize, UtsegBatchItem)],
    lang: &LanguageCode3,
) -> Result<Vec<UtsegResponse>, ServerError> {
    let payload_items: Vec<_> = items.iter().map(|(_, item)| item.clone()).collect();
    let artifacts = PreparedArtifactRuntimeV2::new("utseg_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create utseg V2 artifact runtime: {error}"
        ))
    })?;
    let request_ids = PreparedTextRequestIdsV2::for_task("utseg");
    let request = build_utseg_request_v2(artifacts.store(), &request_ids, lang, &payload_items)
        .map_err(|error| {
            ServerError::Validation(format!("failed to build utseg V2 worker request: {error}"))
        })?;

    info!(
        num_items = items.len(),
        lang = %lang,
        "Dispatching utseg execute_v2 batch"
    );

    let response = dispatch_execute_v2_with_retry(pool, lang, &request).await?;
    let result = parse_utseg_result_v2(&response)
        .map_err(|error| ServerError::Validation(format!("invalid utseg V2 result: {error}")))?;
    if result.items.len() != items.len() {
        return Err(ServerError::Validation(format!(
            "utseg V2 returned {} items for {} requests",
            result.items.len(),
            items.len()
        )));
    }

    let mut utseg_responses = Vec::with_capacity(result.items.len());
    for (i, item_result) in result.items.iter().enumerate() {
        if let Some(error) = &item_result.error {
            warn!(item = i, error = %error, "Infer error for item (using empty response)");
            utseg_responses.push(UtsegResponse {
                assignments: Vec::new(),
            });
            continue;
        }

        if let Some(trees) = &item_result.trees {
            let num_words = items[i].1.words.len();
            let assignments = utseg_compute::compute_assignments(trees, num_words);
            utseg_responses.push(UtsegResponse { assignments });
            continue;
        }

        warn!(
            item = i,
            "Utseg V2 returned no trees and no error (using empty response)"
        );
        utseg_responses.push(UtsegResponse {
            assignments: Vec::new(),
        });
    }

    Ok(utseg_responses)
}

fn infer_single_file<'a>(
    pool: &'a WorkerPool,
    items: &'a [(usize, UtsegBatchItem)],
    lang: &'a LanguageCode3,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Vec<UtsegResponse>, ServerError>> + Send + 'a>,
> {
    Box::pin(async move { infer_batch(pool, items, lang).await })
}

fn integrate_assignments(
    assignment_map: &mut HashMap<usize, Vec<usize>>,
    misses: &[(usize, UtsegBatchItem)],
    responses: &[UtsegResponse],
) {
    for ((utt_ordinal, item), resp) in misses.iter().zip(responses.iter()) {
        if resp.assignments.len() == item.words.len() {
            assignment_map.insert(*utt_ordinal, resp.assignments.clone());
        } else {
            warn!(
                utterance = utt_ordinal,
                expected = item.words.len(),
                got = resp.assignments.len(),
                "utseg assignment length mismatch, keeping original"
            );
        }
    }
}

/// Store utseg results in cache.
async fn cache_put_entries(
    cache: &UtteranceCache,
    keys: &[CacheKey],
    items: &[(usize, UtsegBatchItem)],
    responses: &[UtsegResponse],
    _lang: &LanguageCode3,
    engine_version: &EngineVersion,
) {
    let ba_version = env!("CARGO_PKG_VERSION");
    let cache_entries: Vec<(String, serde_json::Value)> = keys
        .iter()
        .zip(items.iter().zip(responses.iter()))
        .filter(|(_, ((_, item), resp))| {
            // Only cache valid responses with matching lengths
            resp.assignments.len() == item.words.len() && !resp.assignments.is_empty()
        })
        .map(|(key, (_, resp))| {
            let data = serde_json::json!({
                "assignments": resp.assignments,
            });
            (key.as_str().to_string(), data)
        })
        .collect();

    if !cache_entries.is_empty()
        && let Err(e) = cache
            .put_batch(
                &cache_entries,
                CACHE_TASK.as_str(),
                engine_version,
                ba_version,
            )
            .await
    {
        warn!(error = %e, "Failed to store utseg cache entries (non-fatal)");
    }
}

fn cache_put_single_file<'a>(
    cache: &'a UtteranceCache,
    keys: &'a [CacheKey],
    items: &'a [(usize, UtsegBatchItem)],
    responses: &'a [UtsegResponse],
    lang: &'a LanguageCode3,
    engine_version: &'a EngineVersion,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
    Box::pin(async move {
        cache_put_entries(cache, keys, items, responses, lang, engine_version).await;
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_cache() -> (UtteranceCache, tempfile::TempDir) {
        let dir = tempfile::TempDir::new().unwrap();
        let cache = UtteranceCache::sqlite(Some(dir.path().to_path_buf()))
            .await
            .unwrap();
        (cache, dir)
    }

    #[tokio::test]
    async fn test_partition_by_cache_all_misses() {
        let (cache, _dir) = test_cache().await;
        let items = vec![(
            0,
            UtsegBatchItem {
                words: vec!["hello".into(), "world".into()],
                text: "hello world".into(),
            },
        )];
        let eng = LanguageCode::new("eng");
        let ev = EngineVersion::from("1.0");
        let (hits, miss_keys, misses) = partition_by_cache(&items, &eng, &cache, &ev).await;
        assert!(hits.is_empty());
        assert_eq!(miss_keys.len(), 1);
        assert_eq!(misses.len(), 1);
    }

    #[tokio::test]
    async fn test_partition_by_cache_with_hit() {
        let (cache, _dir) = test_cache().await;

        let eng = LanguageCode::new("eng");
        let words = vec!["hello".to_string(), "world".to_string()];
        let key = cache_key(&words, &eng);
        let data = serde_json::json!({"assignments": [0, 0]});
        cache
            .put_batch(
                &[(key.as_str().to_string(), data)],
                CACHE_TASK.as_str(),
                "1.0",
                "0.1.0",
            )
            .await
            .unwrap();

        let items = vec![(
            0,
            UtsegBatchItem {
                words: words.clone(),
                text: "hello world".into(),
            },
        )];
        let ev = EngineVersion::from("1.0");
        let (hits, miss_keys, misses) = partition_by_cache(&items, &eng, &cache, &ev).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[&0], vec![0, 0]);
        assert!(miss_keys.is_empty());
        assert!(misses.is_empty());
    }
}
