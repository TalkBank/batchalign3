//! Cross-file batch morphosyntax processing and cache helpers.

use std::collections::HashMap;

use crate::api::EngineVersion;
use crate::cache::{CacheBackend, UtteranceCache};
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::workflow::text_batch::{
    TextBatchFileInput, TextBatchFileResult, TextBatchFileResults,
};
use batchalign_chat_ops::morphosyntax::{
    BatchItemWithPosition, MwtDict, TokenizationMode, cache_key, clear_morphosyntax,
    collect_payloads, declared_languages, extract_strings, inject_from_cache, inject_results,
    validate_mor_alignment,
};
use batchalign_chat_ops::nlp::UdResponse;
use batchalign_chat_ops::parse::{is_dummy, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{CacheKey, ChatFile, LanguageCode};
use tracing::warn;

use super::CACHE_TASK;
use super::worker::{cache_put_entries, infer_batch};

// ---------------------------------------------------------------------------
// Cross-file batch morphosyntax processing
// ---------------------------------------------------------------------------

/// Process multiple CHAT files, pooling all cache misses into a single
/// `batch_infer` call for maximum throughput.
///
/// Returns `(filename, Ok(output_text) | Err(error_msg))` for each file.
///
/// This function preserves per-file correctness boundaries while sharing one
/// model call: parse/collect per file, aggregate misses globally, then
/// repartition responses back by file before injection and validation.
pub(crate) async fn run_morphosyntax_batch_impl(
    files: &[TextBatchFileInput],
    services: PipelineServices<'_>,
    params: &MorphosyntaxParams<'_>,
) -> TextBatchFileResults {
    let primary_lang = LanguageCode::new(params.lang.as_ref());
    let mut results: TextBatchFileResults = Vec::with_capacity(files.len());

    // 1. Parse all files
    let mut parsed_files: Vec<ChatFile> = Vec::with_capacity(files.len());
    let mut dummy_flags: Vec<bool> = Vec::with_capacity(files.len());
    let mut validation_errors: Vec<Option<String>> = Vec::with_capacity(files.len());
    for file in files {
        let filename = file.filename.as_ref();
        let (mut chat_file, parse_errors) = parse_lenient(file.chat_text.as_ref());
        if !parse_errors.is_empty() {
            warn!(
                filename = %filename,
                num_errors = parse_errors.len(),
                "Parse errors (continuing with recovery)"
            );
        }
        let dummy = is_dummy(&chat_file);
        if !dummy {
            // Pre-validation gate (L2: MainTierValid)
            if let Err(errors) =
                validate_to_level(&chat_file, parse_errors.len(), ValidityLevel::MainTierValid)
            {
                let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                validation_errors.push(Some(format!(
                    "morphotag pre-validation failed: {}",
                    msgs.join("; ")
                )));
                dummy_flags.push(true); // treat as skip
                parsed_files.push(chat_file);
                continue;
            }
            clear_morphosyntax(&mut chat_file);
        }
        validation_errors.push(None);
        dummy_flags.push(dummy);
        parsed_files.push(chat_file);
    }

    // 2. Collect payloads from each file, tracking provenance
    struct FileMissInfo {
        item_count: usize,
        keys: Vec<CacheKey>,
        global_start: usize,
    }

    let mut all_misses: Vec<BatchItemWithPosition> = Vec::new();
    let mut per_file_info: Vec<Option<FileMissInfo>> = Vec::with_capacity(files.len());

    for file_idx in 0..parsed_files.len() {
        // Skip dummy files entirely — they pass through unchanged
        if dummy_flags[file_idx] {
            per_file_info.push(None);
            continue;
        }

        let langs = declared_languages(&parsed_files[file_idx], &primary_lang);
        let (batch_items, _total) = collect_payloads(
            &parsed_files[file_idx],
            &primary_lang,
            &langs,
            params.multilingual_policy,
        );

        if batch_items.is_empty() {
            per_file_info.push(None);
            continue;
        }

        // Warn when Cantonese input appears to be per-character without --retokenize.
        let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
        if !retokenize && params.lang.as_ref() == "yue" {
            let per_char_count = batch_items
                .iter()
                .flat_map(|(_, _, item, _)| item.words.iter())
                .filter(|w| w.chars().count() == 1 && w.chars().all(|c| c > '\u{2E80}'))
                .count();
            let total_words: usize = batch_items
                .iter()
                .map(|(_, _, item, _)| item.words.len())
                .sum();
            if total_words > 0 && per_char_count * 100 / total_words > 80 {
                warn!(
                    "Cantonese input appears to be per-character tokens \
                     ({per_char_count}/{total_words} single-CJK words). \
                     Consider --retokenize for word-level analysis."
                );
            }
        }

        // Cache lookup
        let (hits, miss_keys, misses) = if params.cache_policy.should_skip() {
            let keys: Vec<CacheKey> = batch_items
                .iter()
                .map(|(_, _, item, _)| {
                    let retokenize =
                        params.tokenization_mode == TokenizationMode::StanzaRetokenize;
                    cache_key(&item.words, &item.lang, params.mwt, retokenize)
                })
                .collect();
            (HashMap::new(), keys, batch_items)
        } else {
            let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
            partition_by_cache(
                &batch_items,
                services.cache,
                services.engine_version,
                params.mwt,
                retokenize,
            )
            .await
        };

        // Inject cache hits immediately
        if !hits.is_empty()
            && let Err(e) = inject_cache_hits(&mut parsed_files[file_idx], &hits)
        {
            warn!(
                filename = %files[file_idx].filename,
                error = %e,
                "Cache injection failed (non-fatal)"
            );
        }

        if misses.is_empty() {
            per_file_info.push(None);
        } else {
            let global_start = all_misses.len();
            let item_count = misses.len();
            per_file_info.push(Some(FileMissInfo {
                item_count,
                keys: miss_keys,
                global_start,
            }));
            all_misses.extend(misses);
        }
    }

    // 3. Single batch_infer across all files
    let all_ud_responses = if all_misses.is_empty() {
        Vec::new()
    } else {
        let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
        match infer_batch(services.pool, &all_misses, params.lang, params.mwt, retokenize).await {
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
                        results.push(TextBatchFileResult::ok(
                            file.filename.clone(),
                            to_chat_string(&parsed_files[file_idx]),
                        ));
                    }
                }
                return results;
            }
        }
    };

    // 4. Distribute responses back to files and inject
    for (file_idx, file) in files.iter().enumerate() {
        let filename = file.filename.as_ref();
        // Skip files that failed pre-validation
        if let Some(ref err) = validation_errors[file_idx] {
            results.push(TextBatchFileResult::err(file.filename.clone(), err.clone()));
            continue;
        }

        let chat_file = &mut parsed_files[file_idx];

        if let Some(ref fm) = per_file_info[file_idx] {
            let global_start = fm.global_start;
            let count = fm.item_count;

            let file_responses: Vec<UdResponse> =
                all_ud_responses[global_start..global_start + count].to_vec();
            let file_items: Vec<BatchItemWithPosition> =
                all_misses[global_start..global_start + count].to_vec();
            let miss_line_indices: Vec<usize> = file_items.iter().map(|(idx, ..)| *idx).collect();

            match inject_results(
                chat_file,
                file_items,
                file_responses,
                &primary_lang,
                params.tokenization_mode,
                params.mwt,
            ) {
                Ok(_retokenize_traces) => {}
                Err(e) => {
                    results.push(TextBatchFileResult::err(
                        file.filename.clone(),
                        format!("Result injection failed: {e}"),
                    ));
                    continue;
                }
            }

            // Validate alignment
            let alignment_warnings = validate_mor_alignment(chat_file);
            for w in &alignment_warnings {
                warn!(filename = %filename, warning = %w, "Morphosyntax alignment mismatch");
            }

            match extract_strings(chat_file, &miss_line_indices) {
                Ok(entries) => {
                    cache_put_entries(services.cache, &fm.keys, &entries, services.engine_version)
                        .await;
                }
                Err(e) => {
                    warn!(filename = %filename, error = %e, "Cache extraction failed (non-fatal)");
                }
            }
        }

        // Post-validation check (warn only — always serialize output so it can
        // be inspected for debugging).
        if !dummy_flags[file_idx]
            && let Err(errors) = validate_output(chat_file, "morphotag")
        {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            warn!(filename = %filename, errors = ?msgs, "morphotag post-validation warnings (non-fatal)");
        }

        results.push(TextBatchFileResult::ok(file.filename.clone(), to_chat_string(chat_file)));
    }

    results
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Inject cache hits into a ChatFile.
pub(crate) fn inject_cache_hits(
    chat_file: &mut ChatFile,
    hits: &HashMap<usize, serde_json::Value>,
) -> Result<(), ServerError> {
    // Convert hits map to the CachedMorphosyntaxEntry JSON format
    let entries: Vec<serde_json::Value> = hits
        .iter()
        .map(|(line_idx, data)| {
            // data from cache should be {mor: "...", gra: "..."} or similar
            let mut entry = serde_json::Map::new();
            entry.insert("line_idx".to_string(), serde_json::json!(line_idx));
            if let Some(obj) = data.as_object() {
                if let Some(mor) = obj.get("mor") {
                    entry.insert("mor".to_string(), mor.clone());
                }
                if let Some(gra) = obj.get("gra") {
                    entry.insert("gra".to_string(), gra.clone());
                }
            }
            serde_json::Value::Object(entry)
        })
        .collect();

    let cache_json = serde_json::to_string(&entries)
        .map_err(|e| ServerError::Validation(format!("Failed to serialize cache hits: {e}")))?;
    inject_from_cache(chat_file, &cache_json)
        .map_err(|e| ServerError::Validation(format!("Cache injection failed: {e}")))
}

/// Partition batch items into cache hits and misses.
///
/// Returns `(hits_map, miss_keys, misses)`.
pub(crate) async fn partition_by_cache(
    batch_items: &[BatchItemWithPosition],
    cache: &UtteranceCache,
    engine_version: &EngineVersion,
    mwt: &MwtDict,
    retokenize: bool,
) -> (
    HashMap<usize, serde_json::Value>,
    Vec<CacheKey>,
    Vec<BatchItemWithPosition>,
) {
    let keys: Vec<CacheKey> = batch_items
        .iter()
        .map(|(_, _, item, _)| cache_key(&item.words, &item.lang, mwt, retokenize))
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

    for (item, key) in batch_items.iter().zip(keys.into_iter()) {
        if let Some(cached_data) = cached.get(key.as_str()) {
            hits.insert(item.0, cached_data.clone());
        } else {
            miss_keys.push(key);
            misses.push(item.clone());
        }
    }

    (hits, miss_keys, misses)
}
