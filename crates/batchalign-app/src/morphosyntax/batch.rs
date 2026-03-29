//! Cross-file batch morphosyntax processing and cache helpers.

use std::collections::HashMap;

use crate::api::{DisplayPath, EngineVersion};
use crate::cache::{CacheBackend, UtteranceCache};
use crate::error::ServerError;
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};
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
    progress_tx: Option<tokio::sync::mpsc::Sender<crate::types::worker_v2::ProgressEventV2>>,
    group_timeout: std::time::Duration,
) -> TextBatchFileResults {
    let primary_lang = LanguageCode::new(params.lang.as_ref());

    // 1. Parse all files in parallel.
    //
    // Tree-sitter parsers are !Send, so we use spawn_blocking with a
    // per-task parser. Each file's parse + validate + clear is independent.
    let file_texts: Vec<(String, String)> = files
        .iter()
        .map(|f| (f.filename.to_string(), f.chat_text.to_string()))
        .collect();

    struct ParsedFile {
        chat_file: ChatFile,
        is_dummy: bool,
        validation_error: Option<String>,
    }

    let num_files = file_texts.len();
    let parsed_results: Vec<ParsedFile> = tokio::task::spawn_blocking(move || {
        use rayon::prelude::*;
        file_texts
            .par_iter()
            .map(|(filename, chat_text)| {
                // Each rayon thread gets its own parser (tree-sitter is !Send).
                let thread_parser =
                    batchalign_chat_ops::parse::TreeSitterParser::new()
                        .expect("tree-sitter CHAT grammar must load");
                let (mut chat_file, parse_errors) = parse_lenient(&thread_parser, chat_text);
                if !parse_errors.is_empty() {
                    // Log inside spawn_blocking is fine — tracing is thread-safe.
                    warn!(
                        filename = %filename,
                        num_errors = parse_errors.len(),
                        "Parse errors (continuing with recovery)"
                    );
                }
                let dummy = is_dummy(&chat_file);
                if !dummy {
                    if let Err(errors) = validate_to_level(
                        &chat_file,
                        parse_errors.len(),
                        ValidityLevel::MainTierValid,
                    ) {
                        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                        return ParsedFile {
                            chat_file,
                            is_dummy: true, // treat as skip
                            validation_error: Some(format!(
                                "morphotag pre-validation failed: {}",
                                msgs.join("; ")
                            )),
                        };
                    }
                    clear_morphosyntax(&mut chat_file);
                }
                ParsedFile {
                    chat_file,
                    is_dummy: dummy,
                    validation_error: None,
                }
            })
            .collect()
    })
    .await
    .expect("parse task must not panic");

    tracing::info!(num_files, "Parsed all files in parallel");

    let mut parsed_files: Vec<ChatFile> = Vec::with_capacity(num_files);
    let mut dummy_flags: Vec<bool> = Vec::with_capacity(num_files);
    let mut validation_errors: Vec<Option<String>> = Vec::with_capacity(num_files);
    for pf in parsed_results {
        parsed_files.push(pf.chat_file);
        dummy_flags.push(pf.is_dummy);
        validation_errors.push(pf.validation_error);
    }

    // 2. Collect payloads from each file, tracking provenance.
    //
    // Phase 2 parallelism: the CPU-bound work (collect_payloads,
    // declared_languages, cache_key, Cantonese warning) runs in parallel
    // via rayon inside spawn_blocking. The async cache lookups run
    // sequentially afterward since they need the SQLite cache.
    struct FileMissInfo {
        item_count: usize,
        keys: Vec<CacheKey>,
        global_start: usize,
    }

    // 2a. CPU-bound payload collection — rayon in spawn_blocking.
    //
    // For each non-dummy file: extract batch items, compute cache keys,
    // and emit the Cantonese per-character warning.
    struct CollectedPayload {
        batch_items: Vec<BatchItemWithPosition>,
        keys: Vec<CacheKey>,
        /// Serialized debug JSON for the debug dumper.
        debug_json: Vec<serde_json::Value>,
    }

    let primary_lang_owned = primary_lang.clone();
    let multilingual_policy = params.multilingual_policy;
    let retokenize_flag = params.tokenization_mode == TokenizationMode::StanzaRetokenize;
    let lang_str = params.lang.as_ref().to_string();
    // Clone MwtDict into the blocking closure (BTreeMap is Send).
    let mwt_owned = params.mwt.clone();

    // spawn_blocking requires 'static, so we clone the ChatFiles for the
    // read-only collection phase. The originals stay for cache-hit injection
    // in the async loop afterward.
    let collected_payloads: Vec<Option<CollectedPayload>> =
        tokio::task::spawn_blocking({
            let parsed_files_clone: Vec<ChatFile> = parsed_files.clone();
            let dummy_flags_clone = dummy_flags.clone();
            move || {
                use rayon::prelude::*;
                parsed_files_clone
                    .par_iter()
                    .enumerate()
                    .map(|(file_idx, chat_file)| {
                        if dummy_flags_clone[file_idx] {
                            return None;
                        }

                        let langs =
                            declared_languages(chat_file, &primary_lang_owned);
                        let (batch_items, _total) = collect_payloads(
                            chat_file,
                            &primary_lang_owned,
                            &langs,
                            multilingual_policy,
                        );

                        if batch_items.is_empty() {
                            return None;
                        }

                        // Cantonese per-character warning (CPU-bound check).
                        if !retokenize_flag && lang_str == "yue" {
                            let per_char_count = batch_items
                                .iter()
                                .flat_map(|(_, _, item, _)| item.words.iter())
                                .filter(|w| {
                                    w.chars().count() == 1
                                        && w.chars().all(|c| c > '\u{2E80}')
                                })
                                .count();
                            let total_words: usize = batch_items
                                .iter()
                                .map(|(_, _, item, _)| item.words.len())
                                .sum();
                            if total_words > 0
                                && per_char_count * 100 / total_words > 80
                            {
                                warn!(
                                    "Cantonese input appears to be per-character tokens \
                                     ({per_char_count}/{total_words} single-CJK words). \
                                     Consider --retokenize for word-level analysis."
                                );
                            }
                        }

                        // Compute cache keys for each batch item.
                        let keys: Vec<CacheKey> = batch_items
                            .iter()
                            .map(|(_, _, item, _)| {
                                cache_key(
                                    &item.words,
                                    &item.lang,
                                    &mwt_owned,
                                    retokenize_flag,
                                )
                            })
                            .collect();

                        // Build debug JSON (used by DebugDumper after we
                        // return to the async context).
                        let debug_json: Vec<serde_json::Value> = batch_items
                            .iter()
                            .map(|(li, uo, item, words)| {
                                serde_json::json!({
                                    "line_idx": li,
                                    "utt_ordinal": uo,
                                    "item_words": &item.words,
                                    "extracted_words": words.iter().map(|w| w.text.as_ref()).collect::<Vec<_>>(),
                                    "word_count": words.len(),
                                })
                            })
                            .collect();

                        Some(CollectedPayload {
                            batch_items,
                            keys,
                            debug_json,
                        })
                    })
                    .collect()
            }
        })
        .await
        .expect("payload collection task must not panic");

    tracing::info!(num_files, "Collected payloads from all files in parallel");

    // 2b. Async cache lookups + cache-hit injection (sequential — needs
    // async SQLite cache and mutable ChatFile access).
    //
    // Instrumented with counters and timers so we can measure whether
    // the cache actually helps in practice.
    let mut all_misses: Vec<BatchItemWithPosition> = Vec::new();
    let mut per_file_info: Vec<Option<FileMissInfo>> = Vec::with_capacity(files.len());
    let mut cache_total_items: usize = 0;
    let mut cache_hits: usize = 0;
    let mut cache_injection_failures: usize = 0;
    let cache_lookup_start = tokio::time::Instant::now();

    for (file_idx, collected) in collected_payloads.into_iter().enumerate() {
        let collected = match collected {
            Some(c) => c,
            None => {
                per_file_info.push(None);
                continue;
            }
        };

        // Debug dump (runs on the async executor — cheap I/O).
        let filename = &files[file_idx].filename;
        services
            .debug_dumper
            .dump_morphosyntax_extracted(filename, &collected.debug_json);

        let CollectedPayload {
            batch_items,
            keys,
            debug_json: _,
        } = collected;

        // Cache lookup — async, uses SQLite.
        let (hits, miss_keys, misses) = if params.cache_policy.should_skip() {
            (HashMap::new(), keys, batch_items)
        } else {
            // partition_by_cache recomputes keys internally, but we already
            // have them. For now, call the existing function to avoid
            // duplicating its cache-get logic. The key computation is cheap
            // relative to the SQLite I/O.
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

        // Track cache metrics.
        cache_total_items += hits.len() + misses.len();
        cache_hits += hits.len();

        // Inject cache hits immediately.
        if !hits.is_empty() {
            if let Err(e) = inject_cache_hits(&mut parsed_files[file_idx], &hits) {
                cache_injection_failures += hits.len();
                warn!(
                    filename = %files[file_idx].filename,
                    error = %e,
                    "Cache injection failed (non-fatal)"
                );
            }
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
    let cache_lookup_ms = cache_lookup_start.elapsed().as_millis() as u64;
    let cache_misses = cache_total_items - cache_hits;
    let effective_hits = cache_hits - cache_injection_failures;
    let hit_rate_pct = if cache_total_items > 0 {
        effective_hits * 100 / cache_total_items
    } else {
        0
    };
    tracing::warn!(
        cache_total_items,
        cache_hits,
        effective_hits,
        cache_misses,
        cache_injection_failures,
        cache_lookup_ms,
        hit_rate_pct,
        "Cache metrics: lookup phase"
    );

    // 3. Batch infer grouped by per-item language — all languages in parallel.
    //
    // Multilingual CHAT files (e.g., @Languages: fra, eng) produce batch
    // items with different per-item languages. Each language group must be
    // dispatched to a worker loaded with the correct Stanza model — sending
    // French text to an English MWT pipeline produces corrupt Range tokens.
    //
    // Language groups are dispatched **concurrently** since each language uses
    // a separate worker process. This is the primary throughput lever for
    // multilingual batches: a 5-language batch runs ~5x faster than serial.
    //
    // BA2 parity: BA2 used stanza.MultilingualPipeline for this. BA3 groups
    // by language and dispatches each group to a separate single-language
    // worker, which achieves the same correctness without MultilingualPipeline.
    let all_ud_responses = if all_misses.is_empty() {
        Vec::new()
    } else {
        let retokenize = params.tokenization_mode == TokenizationMode::StanzaRetokenize;

        // Group items by their per-item language, preserving original indices.
        let mut by_lang: HashMap<
            LanguageCode,
            Vec<(usize, &BatchItemWithPosition)>,
        > = HashMap::new();
        for (global_idx, item) in all_misses.iter().enumerate() {
            let item_lang = &item.2.lang;
            by_lang
                .entry(item_lang.clone())
                .or_default()
                .push((global_idx, item));
        }

        // Prepare per-language dispatch inputs (owned data for the async tasks).
        struct LangDispatch {
            lang3: crate::api::LanguageCode3,
            items: Vec<BatchItemWithPosition>,
            /// Original global indices so results can be placed back.
            global_indices: Vec<usize>,
        }

        // Partition language groups into supported (dispatch to workers)
        // and unsupported (skip with warning).  This prevents spawning
        // workers for languages Stanza cannot process, which would either
        // crash the worker or deadlock the pool.
        let mut dispatches: Vec<LangDispatch> = Vec::new();
        let mut skipped_indices: Vec<(usize, String)> = Vec::new();

        for (lang, lang_items) in &by_lang {
            let lang3 = crate::api::LanguageCode3::try_new(lang.as_ref())
                .unwrap_or_else(|_| params.lang.clone());

            // Check language support via the capability registry (populated
            // from worker's resources.json), falling back to the hardcoded
            // table when the registry hasn't been populated yet.
            let lang_supported = if let Some(reg) = services.pool.stanza_registry() {
                reg.supports_morphosyntax(lang.as_ref())
            } else {
                batchalign_chat_ops::morphosyntax::stanza_languages::is_stanza_supported(lang)
            };
            if !lang_supported
            {
                tracing::warn!(
                    lang = %lang3,
                    items = lang_items.len(),
                    "Skipping unsupported language — utterances will have empty morphosyntax"
                );
                for (global_idx, _) in lang_items {
                    skipped_indices.push((*global_idx, lang3.to_string()));
                }
                continue;
            }

            let items: Vec<BatchItemWithPosition> =
                lang_items.iter().map(|(_, item)| (*item).clone()).collect();
            let global_indices: Vec<usize> =
                lang_items.iter().map(|(idx, _)| *idx).collect();
            dispatches.push(LangDispatch { lang3, items, global_indices });
        }

        if !skipped_indices.is_empty() {
            tracing::info!(
                skipped = skipped_indices.len(),
                "Skipped utterances with unsupported languages"
            );
        }

        // Dispatch language groups with bounded concurrency.
        //
        // Each language group needs up to `max_workers_per_key` workers.
        // Unbounded `join_all` would try to spawn workers for all languages
        // simultaneously, exceeding `max_total_workers` and deadlocking.
        //
        // Instead, we use a semaphore to limit the number of concurrent
        // language groups to `max_total_workers / max_workers_per_key`.
        // When a group finishes and releases its workers, the next group
        // starts — no deadlock, full utilization, all groups eventually
        // process.  This is the same pattern FA pipeline uses for per-file
        // concurrency (JoinSet + Semaphore).
        let max_per_key = services.pool.max_workers_per_key().max(1);
        let max_total = services.pool.effective_max_total_workers().max(1);
        let max_concurrent_groups = (max_total / max_per_key).max(1);

        tracing::info!(
            language_groups = dispatches.len(),
            max_concurrent_groups,
            max_total_workers = max_total,
            max_workers_per_key = max_per_key,
            "Dispatching morphosyntax language groups with bounded concurrency"
        );

        let lang_sem = std::sync::Arc::new(tokio::sync::Semaphore::new(max_concurrent_groups));

        let mut all_responses: Vec<Option<UdResponse>> = vec![None; all_misses.len()];
        let mut batch_error: Option<ServerError> = None;

        // Build futures that each acquire a semaphore permit before dispatching.
        let futures: Vec<_> = dispatches
            .iter()
            .map(|d| {
                let sem = lang_sem.clone();
                let ptx = progress_tx.clone();
                let gt = group_timeout;
                async move {
                    let _permit = sem.acquire().await.map_err(|_| {
                        ServerError::Validation("language group semaphore closed".into())
                    })?;
                    tracing::info!(
                        lang = %d.lang3,
                        items = d.items.len(),
                        available_permits = sem.available_permits(),
                        "Acquired semaphore permit for language group"
                    );
                    let result = tokio::time::timeout(
                        gt,
                        infer_batch(services.pool, &d.items, &d.lang3, params.mwt, retokenize, ptx.as_ref()),
                    ).await;
                    match result {
                        Ok(inner) => inner,
                        Err(_) => {
                            tracing::error!(
                                lang = %d.lang3,
                                items = d.items.len(),
                                timeout_s = gt.as_secs(),
                                "Language group timed out — producing empty responses"
                            );
                            Err(ServerError::Validation(format!(
                                "language group {} timed out after {}s",
                                d.lang3, gt.as_secs()
                            )))
                        }
                    }
                }
            })
            .collect();

        let outcomes = futures::future::join_all(futures).await;

        for (dispatch, outcome) in dispatches.iter().zip(outcomes) {
            match outcome {
                Ok(responses) => {
                    for (global_idx, ud) in dispatch.global_indices.iter().zip(responses) {
                        all_responses[*global_idx] = Some(ud);
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        lang = %dispatch.lang3,
                        error = %e,
                        "Batch infer failed for language group"
                    );
                    let dump_items: Vec<_> = dispatch.items.iter().map(|(li, uo, item, _)| {
                        serde_json::json!({
                            "line_idx": li,
                            "utt_ordinal": uo,
                            "words": &item.words,
                            "lang": item.lang.as_ref(),
                        })
                    }).collect();
                    services.debug_dumper.dump_morphosyntax_failed_batch(
                        &format!("batch_failure_{}", dispatch.lang3),
                        &dump_items,
                        &e,
                    );
                    batch_error = Some(e);
                }
            }
        }

        if let Some(ref e) = batch_error {
            tracing::warn!(
                error = %e,
                "One or more language groups failed — continuing with \
                 successful groups. Utterances needing failed languages \
                 will get empty morphosyntax results."
            );
        }

        // Fill missing responses with empty UdResponse for items whose
        // language group failed.  This allows files that span multiple
        // languages to still get results for the successful languages,
        // rather than poisoning the entire batch.
        let collected: Vec<UdResponse> = all_responses
            .into_iter()
            .map(|r| r.unwrap_or_else(|| UdResponse { sentences: Vec::new() }))
            .collect();
        services
            .debug_dumper
            .dump_morphosyntax_ud_responses("batch", &collected);
        collected
    };

    // 4. Distribute responses back to files and inject — rayon parallel.
    //
    // Each file's injection is independent: inject_results, validate,
    // extract_strings, serialize. The tree-sitter parser is !Send, so
    // each rayon thread creates its own (same pattern as the parse phase).
    //
    // Cache puts are async (SQLite), so we collect the cache data during
    // the parallel phase and perform the puts sequentially afterward.
    use batchalign_chat_ops::morphosyntax::MorphosyntaxStringsEntry;

    /// Data collected per file during the parallel injection phase that
    /// needs async cache writes afterward.
    struct CachePutData {
        keys: Vec<CacheKey>,
        entries: Vec<MorphosyntaxStringsEntry>,
    }

    // Build per-file input bundles that own all the data each rayon task
    // needs, avoiding shared mutable state across threads.
    struct InjectionInput {
        filename: DisplayPath,
        chat_file: ChatFile,
        is_dummy: bool,
        validation_error: Option<String>,
        miss_info: Option<FileMissInfo>,
        file_responses: Vec<UdResponse>,
        file_items: Vec<BatchItemWithPosition>,
    }

    // Consume parsed_files, per_file_info, dummy_flags, and validation_errors
    // by zipping into owned bundles. Each InjectionInput owns its ChatFile so
    // rayon threads can mutate them independently.
    let injection_inputs: Vec<InjectionInput> = parsed_files
        .into_iter()
        .zip(per_file_info.into_iter())
        .zip(dummy_flags.into_iter())
        .zip(validation_errors.into_iter())
        .enumerate()
        .map(|(file_idx, (((chat_file, miss_info), is_dummy), validation_error))| {
            let (file_responses, file_items) = if let Some(ref fm) = miss_info {
                let gs = fm.global_start;
                let cnt = fm.item_count;
                (
                    all_ud_responses[gs..gs + cnt].to_vec(),
                    all_misses[gs..gs + cnt].to_vec(),
                )
            } else {
                (Vec::new(), Vec::new())
            };
            InjectionInput {
                filename: files[file_idx].filename.clone(),
                chat_file,
                is_dummy,
                validation_error,
                miss_info,
                file_responses,
                file_items,
            }
        })
        .collect();

    let primary_lang_inject = primary_lang.clone();
    let tokenization_mode_inject = params.tokenization_mode;
    let mwt_inject = params.mwt.clone();

    // Run injection + serialization in parallel via rayon.
    let injection_results: Vec<(TextBatchFileResult, Option<CachePutData>)> =
        tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;
            injection_inputs
                .into_par_iter()
                .map(|input| {
                    let InjectionInput {
                        filename,
                        mut chat_file,
                        is_dummy,
                        validation_error,
                        miss_info,
                        file_responses,
                        file_items,
                    } = input;

                    // Skip files that failed pre-validation.
                    if let Some(err) = validation_error {
                        return (
                            TextBatchFileResult::err(filename, err),
                            None,
                        );
                    }

                    let mut cache_data: Option<CachePutData> = None;

                    if let Some(fm) = miss_info {
                        let miss_line_indices: Vec<usize> =
                            file_items.iter().map(|(idx, ..)| *idx).collect();

                        // Each rayon thread gets its own parser (tree-sitter
                        // is !Send).
                        let thread_parser =
                            batchalign_chat_ops::parse::TreeSitterParser::new()
                                .expect("tree-sitter CHAT grammar must load");

                        match inject_results(
                            &thread_parser,
                            &mut chat_file,
                            file_items,
                            file_responses,
                            &primary_lang_inject,
                            tokenization_mode_inject,
                            &mwt_inject,
                        ) {
                            Ok(_retokenize_traces) => {}
                            Err(e) => {
                                return (
                                    TextBatchFileResult::err(
                                        filename,
                                        format!("Result injection failed: {e}"),
                                    ),
                                    None,
                                );
                            }
                        }

                        // Validate alignment.
                        let alignment_warnings =
                            validate_mor_alignment(&chat_file);
                        for w in &alignment_warnings {
                            warn!(
                                filename = %filename,
                                warning = %w,
                                "Morphosyntax alignment mismatch"
                            );
                        }

                        // Extract cache strings — the actual async cache put
                        // happens after spawn_blocking returns.
                        match extract_strings(&chat_file, &miss_line_indices) {
                            Ok(entries) => {
                                cache_data = Some(CachePutData {
                                    keys: fm.keys,
                                    entries,
                                });
                            }
                            Err(e) => {
                                warn!(
                                    filename = %filename,
                                    error = %e,
                                    "Cache extraction failed (non-fatal)"
                                );
                            }
                        }
                    }

                    // Post-validation check (warn only — always serialize
                    // output so it can be inspected for debugging).
                    if !is_dummy {
                        if let Err(errors) =
                            validate_output(&chat_file, "morphotag")
                        {
                            let msgs: Vec<String> =
                                errors.iter().map(|e| e.to_string()).collect();
                            warn!(
                                filename = %filename,
                                errors = ?msgs,
                                "morphotag post-validation warnings (non-fatal)"
                            );
                        }
                    }

                    let result = TextBatchFileResult::ok(
                        filename,
                        to_chat_string(&chat_file),
                    );
                    (result, cache_data)
                })
                .collect()
        })
        .await
        .expect("injection task must not panic");

    tracing::info!(num_files, "Injected and serialized all files in parallel");

    // 4b. Async cache puts — sequential, needs SQLite.
    let cache_put_start = tokio::time::Instant::now();
    let mut cache_put_count: usize = 0;
    for (_result, cache_data) in &injection_results {
        if let Some(cd) = cache_data {
            cache_put_count += cd.keys.len();
            cache_put_entries(
                services.cache,
                &cd.keys,
                &cd.entries,
                services.engine_version,
            )
            .await;
        }
    }
    let cache_put_ms = cache_put_start.elapsed().as_millis() as u64;
    tracing::warn!(
        cache_put_count,
        cache_put_ms,
        "Cache metrics: put phase"
    );

    // Collect the final results in file order.
    let results: TextBatchFileResults = injection_results
        .into_iter()
        .map(|(result, _)| result)
        .collect();

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
