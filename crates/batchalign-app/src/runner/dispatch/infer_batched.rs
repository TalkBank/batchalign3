//! Batched text NLP dispatch (morphotag, utseg, translate, coref, compare).

use std::collections::HashMap;

use crate::api::{LanguageCode3, ReleasedCommand};
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::recipe_runner::runtime::{primary_output_artifact, write_text_output_artifact};
use crate::runner::DispatchHostContext;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};
use tracing::warn;

use crate::store::{RunnerJobSnapshot, unix_now};

use super::super::util::{FileRunTracker, FileStage, set_file_progress};
use super::BatchedInferDispatchPlan;
use super::compare_pipeline::dispatch_compare;

/// Parse CHAT text, apply merge_abbreviations transform, re-serialize.
pub(in crate::runner) fn apply_merge_abbrev(chat_text: &str) -> String {
    let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
        .expect("tree-sitter CHAT grammar must load");
    let (mut file, _) = batchalign_chat_ops::parse::parse_lenient(&parser, chat_text);
    batchalign_chat_ops::merge_abbrev::merge_abbreviations(&mut file);
    batchalign_chat_ops::serialize::to_chat_string(&file)
}

/// Dispatch files via the server-side infer path.
///
/// Reads all CHAT files, runs processing in Rust (parse ->
/// cache -> infer -> inject -> serialize), and records results per file.
pub(crate) async fn dispatch_batched_infer(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    services: PipelineServices<'_>,
    plan: BatchedInferDispatchPlan,
) {
    let BatchedInferDispatchPlan {
        kernel_plan,
        tokenization_mode,
        multilingual_policy,
        cache_policy,
        should_merge_abbrev,
        mwt,
    } = plan;
    let job_id = &job.identity.job_id;
    let correlation_id = &*job.identity.correlation_id;
    let file_list = &job.pending_files;
    let fallback_lang = LanguageCode3::eng();
    let lang: &LanguageCode3 = job.dispatch.lang.as_resolved().unwrap_or(&fallback_lang);
    let command = job.dispatch.command;
    debug_assert_eq!(kernel_plan.file_parallelism_hint, 1);

    let started_at = unix_now();
    let sink = host.sink().clone();

    let stage = FileStage::for_batch_command(command);

    // Mark all files as processing, open their batch attempts, and publish the
    // initial stage label.
    for file in file_list {
        let filename = file.filename.as_ref();
        FileRunTracker::new(sink.as_ref(), job_id, filename)
            .begin_first_attempt(WorkUnitKind::BatchInfer, started_at, stage)
            .await;
    }

    // Read all CHAT file contents (and optional "before" texts for incremental)
    let mut file_texts: Vec<TextBatchFileInput> = Vec::with_capacity(file_list.len());
    let mut before_texts: HashMap<String, String> = HashMap::new();
    let mut read_errors: Vec<(usize, String)> = Vec::new();

    for file in file_list {
        let file_index = file.file_index;
        let filename = file.filename.as_ref();
        let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename);

        // Transition to Reading while doing I/O so the frontend shows activity.
        lifecycle.stage(FileStage::Reading).await;

        let read_path: std::path::PathBuf =
            if job.filesystem.paths_mode && file_index < job.filesystem.source_paths.len() {
                job.filesystem.source_paths[file_index]
                    .assume_shared_filesystem()
                    .as_path()
                    .to_owned()
            } else {
                job.filesystem.staging_dir.join("input").join(filename)
                    .as_path()
                    .to_owned()
            };
        let before_path = if !job.filesystem.before_paths.is_empty()
            && file_index < job.filesystem.before_paths.len()
        {
            Some(job.filesystem.before_paths[file_index]
                .assume_shared_filesystem())
        } else {
            None
        };
        match tokio::fs::read_to_string(&read_path).await {
            Ok(content) => {
                file_texts.push(TextBatchFileInput::new(filename.to_string(), content));
                // Read the corresponding "before" file if available
                if let Some(bp) = before_path
                    && let Ok(before_content) = tokio::fs::read_to_string(&bp).await
                {
                    before_texts.insert(filename.to_string(), before_content);
                }
            }
            Err(e) => {
                let err_msg = format!("Failed to read input: {e}");
                lifecycle
                    .fail(&err_msg, FailureCategory::InputMissing, unix_now())
                    .await;
                read_errors.push((file_index, filename.to_string()));
            }
        }
    }

    if file_texts.is_empty() {
        return;
    }

    // Publish the batch total so frontends can display "0/N" while inference runs.
    let total_files = file_texts.len() as i64;
    for file in &file_texts {
        set_file_progress(
            sink.as_ref(),
            job_id,
            file.filename.as_ref(),
            stage,
            Some(0),
            Some(total_files),
        )
        .await;
    }

    // Run the appropriate server-side orchestrator
    let results = match command {
        ReleasedCommand::Morphotag => {
            let mor_params = MorphosyntaxParams {
                lang,
                tokenization_mode,
                cache_policy,
                multilingual_policy,
                mwt: &mwt,
            };
            // Use incremental processing when before texts are available
            if !before_texts.is_empty() {
                let mut results: TextBatchFileResults = Vec::new();
                for file in &file_texts {
                    let filename = file.filename.as_ref();
                    let after_text = file.chat_text.as_ref();
                    let result = if let Some(before_text) = before_texts.get(filename) {
                        crate::morphosyntax::process_morphosyntax_incremental(
                            before_text,
                            after_text,
                            services,
                            &mor_params,
                        )
                        .await
                    } else {
                        crate::morphosyntax::process_morphosyntax(after_text, services, &mor_params)
                            .await
                    };
                    match result {
                        Ok(text) => {
                            results.push(TextBatchFileResult::ok(file.filename.clone(), text))
                        }
                        Err(e) => results.push(TextBatchFileResult::err(
                            file.filename.clone(),
                            e.to_string(),
                        )),
                    }
                }
                results
            } else {
                // Create a progress channel for batch-level monitoring.
                // The drain task logs progress and publishes via the event sink.
                let (progress_tx, mut progress_rx) =
                    tokio::sync::mpsc::channel::<crate::types::worker_v2::ProgressEventV2>(64);

                let progress_job_id = job_id.clone();
                let _progress_sink = sink.clone();
                let drain_handle = tokio::spawn(async move {
                    use crate::runner::util::batch_progress::BatchInferProgress;
                    let mut progress = BatchInferProgress::new();
                    let mut last_log = tokio::time::Instant::now();

                    while let Some(event) = progress_rx.recv().await {
                        // The stage field carries the language code for morphosyntax.
                        let lang = &event.stage;
                        if !progress.language_groups.contains_key(lang) {
                            progress.register_group(lang, event.total as u64);
                        }
                        progress.update_group(lang, event.completed as u64);

                        // Rate-limited logging (at most once per 2 seconds).
                        let now = tokio::time::Instant::now();
                        if now.duration_since(last_log).as_secs() >= 2 {
                            last_log = now;
                            tracing::info!(
                                job_id = %progress_job_id,
                                summary = %progress.summary(),
                                "Batch morphosyntax progress"
                            );
                        }
                    }

                    // Final log when channel closes (all language groups done).
                    tracing::info!(
                        job_id = %progress_job_id,
                        summary = %progress.summary(),
                        "Batch morphosyntax complete"
                    );
                });

                let results = crate::morphosyntax::run_morphosyntax_batch_impl(
                    &file_texts,
                    services,
                    &mor_params,
                    Some(progress_tx),
                )
                .await;

                // The sender was cloned per-language-group in batch.rs and
                // dropped when each future completed.  The original sender
                // was moved into run_morphosyntax_batch_impl and dropped
                // when it returned.  The channel is now closed, so the
                // drain task will finish after processing remaining events.
                let _ = drain_handle.await;

                results
            }
        }
        ReleasedCommand::Utseg => {
            crate::utseg::process_utseg_batch(
                &file_texts,
                lang,
                services.pool,
                services.cache,
                services.engine_version,
                cache_policy,
            )
            .await
        }
        ReleasedCommand::Translate => {
            crate::translate::process_translate_batch(
                &file_texts,
                lang,
                services.pool,
                services.cache,
                services.engine_version,
                cache_policy,
            )
            .await
        }
        ReleasedCommand::Coref => {
            crate::coref::process_coref_batch(&file_texts, lang, services.pool).await
        }
        ReleasedCommand::Compare => {
            dispatch_compare(
                job,
                host,
                services,
                &file_texts,
                cache_policy,
                &mwt,
                should_merge_abbrev,
            )
            .await;
            return; // compare handles its own result recording
        }
        _ => {
            warn!(command = %command, "Unsupported batched infer command");
            return;
        }
    };

    let finished_at = unix_now();

    // Record results per file, reporting Writing progress as each file completes.
    let total_results = results.len() as i64;
    for (result_idx, file_result) in results.into_iter().enumerate() {
        let filename = file_result.filename;
        let result = file_result.result;
        let lifecycle = FileRunTracker::new(sink.as_ref(), job_id, filename.as_ref());
        // Find the file_index for this filename
        let file_index = file_list
            .iter()
            .find(|file| file.filename == filename)
            .map(|file| file.file_index)
            .unwrap_or(0);

        match result {
            Ok(output_chat) => {
                // Signal the Writing stage with a per-file counter.
                set_file_progress(
                    sink.as_ref(),
                    job_id,
                    filename.as_ref(),
                    FileStage::Writing,
                    Some(result_idx as i64 + 1),
                    Some(total_results),
                )
                .await;

                // Optionally merge abbreviations before writing
                let output_text = if should_merge_abbrev {
                    apply_merge_abbrev(output_chat.as_ref())
                } else {
                    output_chat.into_string()
                };

                // Write output
                let primary_output = primary_output_artifact(command, &filename);

                if let Err(e) = write_text_output_artifact(
                    &job.filesystem,
                    file_index,
                    &primary_output.display_path,
                    &output_text,
                )
                .await
                {
                    warn!(
                        job_id = %job_id,
                        correlation_id = %correlation_id,
                        filename = %filename,
                        error = %e,
                        "Failed to write infer output"
                    );
                }

                lifecycle
                    .complete_with_result(
                        primary_output.display_path.clone(),
                        primary_output.content_type,
                        finished_at,
                    )
                    .await;
            }
            Err(err) => {
                let err_msg = err.into_message();
                lifecycle
                    .fail(&err_msg, FailureCategory::ProviderTerminal, finished_at)
                    .await;
            }
        }
    }
    let _ = correlation_id; // mark used for non-compare paths
}
