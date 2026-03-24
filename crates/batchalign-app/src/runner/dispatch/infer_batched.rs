//! Batched text NLP dispatch (morphotag, utseg, translate, coref, compare).

use std::collections::HashMap;
use std::sync::Arc;

use crate::api::{ContentType, LanguageCode3, ReleasedCommand};
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::workflow::text_batch::{TextBatchFileInput, TextBatchFileResult, TextBatchFileResults};
use tracing::warn;

use crate::store::{JobStore, RunnerJobSnapshot, unix_now};

use super::super::util::{FileRunTracker, FileStage, apply_result_filename, set_file_progress};
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
pub(in crate::runner) async fn dispatch_batched_infer(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    services: PipelineServices<'_>,
    plan: BatchedInferDispatchPlan,
) {
    let BatchedInferDispatchPlan {
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

    let started_at = unix_now();

    let stage = FileStage::for_batch_command(command);

    // Mark all files as processing, open their batch attempts, and publish the
    // initial stage label.
    for file in file_list {
        let filename = file.filename.as_ref();
        FileRunTracker::new(store, job_id, filename)
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
        let lifecycle = FileRunTracker::new(store, job_id, filename);

        // Transition to Reading while doing I/O so the frontend shows activity.
        lifecycle.stage(FileStage::Reading).await;

        let read_path =
            if job.filesystem.paths_mode && file_index < job.filesystem.source_paths.len() {
                job.filesystem.source_paths[file_index].clone()
            } else {
                job.filesystem.staging_dir.join("input").join(filename)
            };
        let before_path = if !job.filesystem.before_paths.is_empty()
            && file_index < job.filesystem.before_paths.len()
        {
            Some(job.filesystem.before_paths[file_index].clone())
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
            store,
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
                        Ok(text) => results.push(TextBatchFileResult::ok(file.filename.clone(), text)),
                        Err(e) => results.push(TextBatchFileResult::err(file.filename.clone(), e.to_string())),
                    }
                }
                results
            } else {
                crate::morphosyntax::process_morphosyntax_batch(
                    &file_texts,
                    services,
                    &mor_params,
                )
                .await
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
            crate::coref::process_coref_batch(
                &file_texts,
                lang,
                services.pool,
            )
            .await
        }
        ReleasedCommand::Compare => {
            dispatch_compare(
                job,
                store,
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
        let lifecycle = FileRunTracker::new(store, job_id, filename.as_ref());
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
                    store,
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
                let write_path = if job.filesystem.paths_mode
                    && file_index < job.filesystem.output_paths.len()
                {
                    apply_result_filename(&job.filesystem.output_paths[file_index], &filename)
                } else {
                    job.filesystem.staging_dir.join("output").join(&*filename)
                };

                if let Some(parent) = write_path.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }

                if let Err(e) = tokio::fs::write(&write_path, &output_text).await {
                    warn!(
                        job_id = %job_id,
                        correlation_id = %correlation_id,
                        filename = %filename,
                        error = %e,
                        "Failed to write infer output"
                    );
                }

                lifecycle
                    .complete_with_result(filename.clone(), ContentType::Chat, finished_at)
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
