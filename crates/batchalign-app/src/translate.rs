//! Server-side translation orchestrator.
//!
//! Owns the full CHAT lifecycle for translate jobs:
//! parse → collect payloads → cache check → infer → inject %xtra → cache put → serialize.
//!
//! Python workers receive only `(text) → TranslateResponse` via the infer protocol —
//! pure Google Translate / Seamless inference with zero CHAT awareness.

use std::collections::HashMap;

use async_trait::async_trait;

use crate::api::{ChatText, EngineVersion, LanguageCode3};
use crate::cache::{CacheBackend, UtteranceCache};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_translate_request_v2};
use crate::worker::text_result_v2::parse_translate_result_v2;
use batchalign_chat_ops::parse::{is_dummy, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::translate::{
    TranslateBatchItem, TranslateResponse, apply_translate_results, cache_key, chat_punct_chars,
    collect_translate_payloads, postprocess_translation, preprocess_for_translate,
};
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

/// Cache task name — matches Python's `translation`.
const CACHE_TASK: CacheTaskName = CacheTaskName::Translation;

/// Default target language for translation.
const DEFAULT_TGT_LANG: &str = "eng";

/// Command-specific parameters for the translate workflow family.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TranslateWorkflowParams {
    /// Cache policy used for translation.
    pub cache_policy: CachePolicy,
}

/// Typed workflow operation for translate.
pub(crate) struct TranslateOperation;

/// Trait-oriented workflow wrapper for translate.
pub(crate) type TranslateWorkflow = TextBatchWorkflow<TranslateOperation>;

#[async_trait]
impl TextBatchOperation for TranslateOperation {
    type Shared<'a>
        = PipelineServices<'a>
    where
        Self: 'a;

    type Params<'a>
        = TranslateWorkflowParams
    where
        Self: 'a;

    async fn run_single(
        chat_text: ChatText<'_>,
        lang: &LanguageCode3,
        shared: Self::Shared<'_>,
        params: Self::Params<'_>,
    ) -> Result<String, ServerError> {
        run_translate_impl(
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
        run_translate_batch_impl(
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
// Per-file translate processing
// ---------------------------------------------------------------------------

/// Process a single CHAT file through the translation pipeline.
///
/// Returns the serialized CHAT text with `%xtra` tiers injected.
pub async fn process_translate(
    chat_text: &str,
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    cache_policy: CachePolicy,
) -> Result<String, ServerError> {
    TranslateWorkflow::new()
        .run_per_file(TextPerFileWorkflowRequest {
            chat_text: ChatText::from(chat_text),
            lang,
            shared: PipelineServices {
                pool,
                cache,
                engine_version,
            },
            params: TranslateWorkflowParams { cache_policy },
        })
        .await
}

// ---------------------------------------------------------------------------
// Cross-file batch translate processing
// ---------------------------------------------------------------------------

/// Process multiple CHAT files, pooling all cache misses into a single
/// `batch_infer` call for maximum throughput.
///
/// Returns `(filename, Ok(output_text) | Err(error_msg))` for each file.
pub(crate) async fn process_translate_batch(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    cache_policy: CachePolicy,
) -> TextBatchFileResults {
    TranslateWorkflow::new()
        .run_batch_files(TextBatchWorkflowRequest {
            files,
            lang,
            shared: PipelineServices {
                pool,
                cache,
                engine_version,
            },
            params: TranslateWorkflowParams { cache_policy },
        })
        .await
}

async fn run_translate_impl(
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
        PipelineServices {
            pool,
            cache,
            engine_version,
        },
        cache_policy,
        CachedTextPipelineHooks {
            command: "translate",
            validity: ValidityLevel::StructurallyComplete,
            collect: collect_translate_payloads,
            partition: partition_single_file,
            infer: infer_single_file,
            integrate: integrate_translations,
            cache_put: cache_put_single_file,
            apply: apply_translate_results,
        },
    )
    .await
}

async fn run_translate_batch_impl(
    files: &[TextBatchFileInput],
    lang: &LanguageCode3,
    pool: &WorkerPool,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    cache_policy: CachePolicy,
) -> TextBatchFileResults {
    let tgt_lang = DEFAULT_TGT_LANG;
    let mut results: TextBatchFileResults = Vec::with_capacity(files.len());

    // 1. Parse all files and collect payloads
    let mut parsed_files: Vec<ChatFile> = Vec::with_capacity(files.len());
    let mut parse_error_counts: Vec<usize> = Vec::with_capacity(files.len());
    for file in files {
        let filename = file.filename.as_ref();
        let (chat_file, parse_errors) = parse_lenient(file.chat_text.as_ref());
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
        items: Vec<(usize, TranslateBatchItem)>,
        keys: Vec<CacheKey>,
        global_start: usize,
    }

    let mut all_misses: Vec<(usize, TranslateBatchItem)> = Vec::new();
    let mut per_file_info: Vec<Option<FileMissInfo>> = Vec::with_capacity(files.len());
    let mut per_file_cached: Vec<HashMap<usize, String>> = Vec::with_capacity(files.len());
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
            validation_errors[file_idx] = Some(format!(
                "translate pre-validation failed: {}",
                msgs.join("; ")
            ));
            per_file_info.push(None);
            per_file_cached.push(HashMap::new());
            continue;
        }

        let batch_items = collect_translate_payloads(parsed_file);

        if batch_items.is_empty() {
            per_file_info.push(None);
            per_file_cached.push(HashMap::new());
            continue;
        }

        // Cache lookup
        let src_lang_code = LanguageCode::new(lang.as_ref());
        let tgt_lang_code = LanguageCode::new(tgt_lang);
        let (cached_translations, miss_keys, misses) = if cache_policy.should_skip() {
            let keys: Vec<CacheKey> = batch_items
                .iter()
                .map(|(_, item)| cache_key(&item.text, &src_lang_code, &tgt_lang_code))
                .collect();
            (HashMap::new(), keys, batch_items)
        } else {
            partition_by_cache(
                &batch_items,
                &src_lang_code,
                &tgt_lang_code,
                cache,
                engine_version,
            )
            .await
        };

        per_file_cached.push(cached_translations);

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
    let all_translate_responses = if all_misses.is_empty() {
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
                        // Apply cached translations and serialize
                        let chat_file = &mut parsed_files[file_idx];
                        let cached = &per_file_cached[file_idx];
                        if !cached.is_empty() {
                            apply_translate_results(chat_file, cached);
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
        let mut translation_map = std::mem::take(&mut per_file_cached[file_idx]);

        if let Some(ref fm) = per_file_info[file_idx] {
            let global_start = fm.global_start;
            let count = fm.items.len();

            let file_responses: Vec<TranslateResponse> =
                all_translate_responses[global_start..global_start + count].to_vec();

            for ((line_idx, _item), resp) in fm.items.iter().zip(file_responses.iter()) {
                if !resp.translation.is_empty() {
                    translation_map.insert(*line_idx, resp.translation.clone());
                }
            }

            // Cache store for this file's misses
            cache_put_entries(
                cache,
                &fm.keys,
                &file_responses,
                lang,
                tgt_lang,
                engine_version,
            )
            .await;
        }

        // Apply all translations (cached + inferred)
        if !translation_map.is_empty() {
            apply_translate_results(chat_file, &translation_map);
        }

        // Post-validation check (warn only — always serialize output for debugging).
        if let Err(errors) = validate_output(chat_file, "translate") {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            warn!(filename = %filename, errors = ?msgs, "translate post-validation warnings (non-fatal)");
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
    HashMap<usize, String>,           // cached: line_idx → translation
    Vec<CacheKey>,                    // miss keys
    Vec<(usize, TranslateBatchItem)>, // misses
);

/// Partition batch items into cache hits and misses.
///
/// Returns `(cached_translations, miss_keys, misses)`.
async fn partition_by_cache(
    batch_items: &[(usize, TranslateBatchItem)],
    src_lang: &LanguageCode,
    tgt_lang: &LanguageCode,
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
) -> PartitionResult {
    let keys: Vec<CacheKey> = batch_items
        .iter()
        .map(|(_, item)| cache_key(&item.text, src_lang, tgt_lang))
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

    let mut hits: HashMap<usize, String> = HashMap::new();
    let mut miss_keys = Vec::new();
    let mut misses = Vec::new();

    for ((line_idx, item), key) in batch_items.iter().zip(keys.into_iter()) {
        if let Some(cached_data) = cached.get(key.as_str()) {
            if let Some(translation) = cached_data.get("translation").and_then(|v| v.as_str())
                && !translation.is_empty()
            {
                hits.insert(*line_idx, translation.to_string());
                continue;
            }
            // Cache entry was invalid/empty — treat as miss
            miss_keys.push(key);
            misses.push((*line_idx, item.clone()));
        } else {
            miss_keys.push(key);
            misses.push((*line_idx, item.clone()));
        }
    }

    (hits, miss_keys, misses)
}

#[allow(clippy::type_complexity)]
fn partition_single_file<'a>(
    batch_items: &'a [(usize, TranslateBatchItem)],
    lang: &'a LanguageCode3,
    cache: &'a UtteranceCache,
    engine_version: &'a EngineVersion,
    cache_policy: CachePolicy,
) -> std::pin::Pin<
    Box<
        dyn std::future::Future<
                Output = (
                    HashMap<usize, String>,
                    Vec<CacheKey>,
                    Vec<(usize, TranslateBatchItem)>,
                ),
            > + Send
            + 'a,
    >,
> {
    Box::pin(async move {
        let tgt_lang = LanguageCode::new(DEFAULT_TGT_LANG);
        let src_lang = LanguageCode::new(lang.as_ref());
        if cache_policy.should_skip() {
            let keys: Vec<CacheKey> = batch_items
                .iter()
                .map(|(_, item)| cache_key(&item.text, &src_lang, &tgt_lang))
                .collect();
            (HashMap::new(), keys, batch_items.to_vec())
        } else {
            partition_by_cache(batch_items, &src_lang, &tgt_lang, cache, engine_version).await
        }
    })
}

/// Send batch items to a worker for translation inference via batched
/// `execute_v2`.
///
/// Applies pre-processing (Chinese space removal) before sending to Python
/// and post-processing (punct spacing, quote normalization) on the raw response.
async fn infer_batch(
    pool: &WorkerPool,
    items: &[(usize, TranslateBatchItem)],
    lang: &LanguageCode3,
) -> Result<Vec<TranslateResponse>, ServerError> {
    let src_lang_code = LanguageCode::new(lang.as_ref());

    // Pre-process text before sending to Python
    let preprocessed_items: Vec<TranslateBatchItem> = items
        .iter()
        .map(|(_, item)| TranslateBatchItem {
            text: preprocess_for_translate(&item.text, &src_lang_code),
        })
        .collect();
    let artifacts = PreparedArtifactRuntimeV2::new("translate_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create translate V2 artifact runtime: {error}"
        ))
    })?;
    let request_ids = PreparedTextRequestIdsV2::for_task("translate");
    let target_lang = LanguageCode3::eng();
    let request = build_translate_request_v2(
        artifacts.store(),
        &request_ids,
        lang,
        &target_lang,
        &preprocessed_items,
    )
    .map_err(|error| {
        ServerError::Validation(format!(
            "failed to build translate V2 worker request: {error}"
        ))
    })?;

    info!(
        num_items = items.len(),
        lang = %lang,
        "Dispatching translate execute_v2 batch"
    );

    let response = dispatch_execute_v2_with_retry(pool, lang, &request).await?;
    let result = parse_translate_result_v2(&response).map_err(|error| {
        ServerError::Validation(format!("invalid translate V2 result: {error}"))
    })?;
    if result.items.len() != items.len() {
        return Err(ServerError::Validation(format!(
            "translate V2 returned {} items for {} requests",
            result.items.len(),
            items.len()
        )));
    }

    // Get punctuation chars for post-processing
    let punct_strings = chat_punct_chars();
    let punct_refs: Vec<&str> = punct_strings.iter().map(|s| s.as_str()).collect();

    let mut translate_responses = Vec::with_capacity(result.items.len());
    for (i, item_result) in result.items.iter().enumerate() {
        if let Some(error) = &item_result.error {
            warn!(item = i, error = %error, "Infer error for item (using empty response)");
            translate_responses.push(TranslateResponse {
                translation: String::new(),
            });
            continue;
        }

        if let Some(raw_translation) = &item_result.raw_translation {
            let processed = postprocess_translation(raw_translation, &punct_refs);
            translate_responses.push(TranslateResponse {
                translation: processed,
            });
            continue;
        }

        warn!(
            item = i,
            "Translate V2 returned no raw_translation and no error (using empty response)"
        );
        translate_responses.push(TranslateResponse {
            translation: String::new(),
        });
    }

    Ok(translate_responses)
}

fn infer_single_file<'a>(
    pool: &'a WorkerPool,
    items: &'a [(usize, TranslateBatchItem)],
    lang: &'a LanguageCode3,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Vec<TranslateResponse>, ServerError>> + Send + 'a>,
> {
    Box::pin(async move { infer_batch(pool, items, lang).await })
}

fn integrate_translations(
    translation_map: &mut HashMap<usize, String>,
    misses: &[(usize, TranslateBatchItem)],
    responses: &[TranslateResponse],
) {
    for ((line_idx, _item), resp) in misses.iter().zip(responses.iter()) {
        if !resp.translation.is_empty() {
            translation_map.insert(*line_idx, resp.translation.clone());
        }
    }
}

/// Store translation results in cache.
async fn cache_put_entries(
    cache: &UtteranceCache,
    keys: &[CacheKey],
    responses: &[TranslateResponse],
    _src_lang: &LanguageCode3,
    _tgt_lang: &str,
    engine_version: &EngineVersion,
) {
    let ba_version = env!("CARGO_PKG_VERSION");
    let cache_entries: Vec<(String, serde_json::Value)> = keys
        .iter()
        .zip(responses.iter())
        .filter(|(_, resp)| !resp.translation.is_empty())
        .map(|(key, resp)| {
            let data = serde_json::json!({
                "translation": resp.translation,
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
        warn!(error = %e, "Failed to store translate cache entries (non-fatal)");
    }
}

fn cache_put_single_file<'a>(
    cache: &'a UtteranceCache,
    keys: &'a [CacheKey],
    _items: &'a [(usize, TranslateBatchItem)],
    responses: &'a [TranslateResponse],
    lang: &'a LanguageCode3,
    engine_version: &'a EngineVersion,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>> {
    Box::pin(async move {
        cache_put_entries(
            cache,
            keys,
            responses,
            lang,
            DEFAULT_TGT_LANG,
            engine_version,
        )
        .await;
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
        let eng = LanguageCode::new("eng");
        let spa = LanguageCode::new("spa");
        let items = vec![(
            0,
            TranslateBatchItem {
                text: "hello world".into(),
            },
        )];
        let ev = EngineVersion::from("1.0");
        let (hits, miss_keys, misses) = partition_by_cache(&items, &eng, &spa, &cache, &ev).await;
        assert!(hits.is_empty());
        assert_eq!(miss_keys.len(), 1);
        assert_eq!(misses.len(), 1);
    }

    #[tokio::test]
    async fn test_partition_by_cache_with_hit() {
        let (cache, _dir) = test_cache().await;

        let eng = LanguageCode::new("eng");
        let spa = LanguageCode::new("spa");
        let text = "hello world";
        let key = cache_key(text, &eng, &spa);
        let data = serde_json::json!({"translation": "hola mundo"});
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
            TranslateBatchItem {
                text: text.to_string(),
            },
        )];
        let ev = EngineVersion::from("1.0");
        let (hits, miss_keys, misses) = partition_by_cache(&items, &eng, &spa, &cache, &ev).await;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[&0], "hola mundo");
        assert!(miss_keys.is_empty());
        assert!(misses.is_empty());
    }
}
