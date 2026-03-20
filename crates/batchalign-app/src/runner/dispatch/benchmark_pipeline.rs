//! Benchmark dispatch built on the Rust-owned transcribe and compare pipelines.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use batchalign_chat_ops::morphosyntax::MwtDict;
use tracing::warn;

use crate::api::{ContentType, EngineVersion, NumWorkers, RevAiJobId, UnixTimestamp};
use crate::benchmark::{BenchmarkRequest, gold_chat_path_for_audio, process_benchmark};
use crate::cache::UtteranceCache;
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::scheduling::{FailureCategory, RetryPolicy, WorkUnitKind};
use crate::store::{JobStore, RunnerJobSnapshot, unix_now};
use crate::transcribe::TranscribeOptions;
use crate::worker::pool::WorkerPool;

use super::super::util::{
    FileRunTracker, FileStage, FileTaskOutcome, apply_result_filename, classify_server_error,
    drain_supervised_file_tasks, is_retryable_worker_failure, spawn_progress_forwarder,
    spawn_supervised_file_task, user_facing_error,
};
use super::BenchmarkDispatchPlan;
use super::infer_batched::apply_merge_abbrev;

/// Shared runtime dependencies for top-level benchmark dispatch.
///
/// Benchmark reuses the transcribe and compare stacks, so the runtime bundle is
/// the same worker/cache context plus the file-level concurrency cap.
pub(in crate::runner) struct BenchmarkDispatchRuntime {
    /// Worker pool used for the benchmark's ASR requests.
    pub pool: Arc<WorkerPool>,
    /// Shared utterance cache used by the compare-side morphosyntax phase.
    pub cache: Arc<UtteranceCache>,
    /// Current engine version string for cache partitioning.
    pub engine_version: EngineVersion,
    /// Optional preflight Rev.AI job ids keyed by original audio path.
    pub rev_job_ids: Arc<HashMap<PathBuf, RevAiJobId>>,
    /// Maximum number of file tasks to run concurrently for this job.
    pub num_workers: NumWorkers,
}

/// Shared per-file benchmark dependencies.
///
/// Benchmark dispatch needs the same server/runtime state for every file in the
/// job. Grouping that state here keeps the per-file function focused on file
/// lifecycle rather than on a wide orchestration signature.
struct BenchmarkFileContext<'a> {
    /// Immutable runner snapshot for the current job.
    job: &'a RunnerJobSnapshot,
    /// Store handle used for lifecycle updates and artifact writes.
    store: &'a Arc<JobStore>,
    /// Shared cache/worker services for transcribe + compare.
    services: PipelineServices<'a>,
    /// Rev.AI preflight job ids keyed by the original provider audio path.
    rev_job_ids: &'a HashMap<PathBuf, RevAiJobId>,
    /// Compare-side cache policy.
    cache_policy: CachePolicy,
    /// MWT dictionary shared with the compare pipeline.
    mwt: &'a MwtDict,
    /// Whether output should pass through merge-abbrev before persistence.
    should_merge_abbrev: bool,
}

/// Dispatch benchmark through the Rust-owned benchmark pipeline.
pub(in crate::runner) async fn dispatch_benchmark_infer(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    runtime: BenchmarkDispatchRuntime,
    plan: BenchmarkDispatchPlan,
) {
    let BenchmarkDispatchPlan {
        base_options,
        cache_policy,
        mwt,
        should_merge_abbrev,
    } = plan;

    let file_sem = Arc::new(tokio::sync::Semaphore::new(runtime.num_workers.0.max(1)));
    let mut tasks = Vec::new();

    for file in &job.pending_files {
        if job.cancel_token.is_cancelled() {
            break;
        }

        let permit = file_sem.clone().acquire_owned().await.unwrap();
        let store = store.clone();
        let pool = runtime.pool.clone();
        let cache = runtime.cache.clone();
        let job = job.clone();
        let engine_version = runtime.engine_version.clone();
        let mut opts = base_options.clone();
        let file = file.clone();
        let mwt = mwt.clone();
        let rev_job_ids = runtime.rev_job_ids.clone();
        let filename = file.filename.clone();

        tasks.push(spawn_supervised_file_task(
            filename,
            "benchmark file task",
            async move {
                let _permit = permit;
                let services = PipelineServices {
                    pool: &pool,
                    cache: &cache,
                    engine_version: &engine_version,
                };
                let context = BenchmarkFileContext {
                    job: &job,
                    store: &store,
                    services,
                    rev_job_ids: rev_job_ids.as_ref(),
                    cache_policy,
                    mwt: &mwt,
                    should_merge_abbrev,
                };
                process_one_benchmark_file(&file, &mut opts, context).await
            },
        ));
    }

    let abnormal_exits =
        drain_supervised_file_tasks(store, &job.identity.job_id, &job.cancel_token, tasks).await;
    if abnormal_exits > 0 {
        warn!(
            job_id = %job.identity.job_id,
            abnormal_exits,
            "Supervised benchmark file tasks exited abnormally"
        );
    }
}

async fn process_one_benchmark_file(
    file: &crate::store::PendingJobFile,
    opts: &mut TranscribeOptions,
    context: BenchmarkFileContext<'_>,
) -> FileTaskOutcome {
    // This dispatch stays fully in Rust once the raw ASR worker capability is
    // available: resolve the gold transcript, normalize media, run the Rust
    // transcribe+compare composition, then persist the hypothesis CHAT and CSV
    // metrics artifacts.
    let BenchmarkFileContext {
        job,
        store,
        services,
        rev_job_ids,
        cache_policy,
        mwt,
        should_merge_abbrev,
    } = context;
    let job_id = &job.identity.job_id;
    let file_index = file.file_index;
    let filename = file.filename.as_ref();
    let lifecycle = FileRunTracker::new(store, job_id, filename);
    let started_at = unix_now();

    lifecycle
        .begin_first_attempt(
            WorkUnitKind::FileInfer,
            started_at,
            FileStage::ResolvingAudio,
        )
        .await;

    let original_audio_path =
        resolve_benchmark_original_audio_path(&job.filesystem, file_index, filename);
    let audio_path = original_audio_path.clone();

    opts.rev_job_id = rev_job_ids.get(&original_audio_path).cloned();

    let gold_path = gold_chat_path_for_audio(&original_audio_path.to_string_lossy());
    let gold_text = match tokio::fs::read_to_string(&gold_path).await {
        Ok(text) => text,
        Err(err) => {
            let err_msg =
                format!("Failed to read benchmark reference transcript {gold_path}: {err}");
            lifecycle
                .fail(&err_msg, FailureCategory::InputMissing, unix_now())
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    let audio_path = match crate::ensure_wav::ensure_wav(&audio_path, None).await {
        Ok(path) => path,
        Err(err) => {
            let err_msg = format!("Media conversion failed for {filename}: {err}");
            lifecycle
                .fail(&err_msg, FailureCategory::Validation, unix_now())
                .await;
            return FileTaskOutcome::TerminalStateRecorded;
        }
    };

    opts.media_name = audio_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned());

    let retry_policy = RetryPolicy::default();

    for attempt_number in 1..=retry_policy.max_attempts {
        if attempt_number > 1 {
            lifecycle
                .restart_attempt(WorkUnitKind::FileInfer, unix_now(), FileStage::Benchmarking)
                .await;
        } else {
            lifecycle.stage(FileStage::Benchmarking).await;
        }

        let progress_tx =
            spawn_progress_forwarder(store.clone(), job_id.clone(), filename.to_string());

        match process_benchmark(BenchmarkRequest {
            audio_path: &audio_path,
            gold_text: &gold_text,
            lang: &job.dispatch.lang.resolve_or(&crate::api::LanguageCode3::from("eng")),
            services,
            transcribe_options: opts,
            cache_policy,
            mwt,
            progress: Some(&progress_tx),
        })
        .await
        {
            Ok((mut chat_output, csv_output)) => {
                lifecycle.stage(FileStage::Writing).await;
                let finished_at = unix_now();

                if should_merge_abbrev {
                    chat_output = apply_merge_abbrev(&chat_output);
                }

                let output_filename = Path::new(filename)
                    .with_extension("cha")
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned();

                let write_path = if job.filesystem.paths_mode
                    && file_index < job.filesystem.output_paths.len()
                {
                    apply_result_filename(
                        &job.filesystem.output_paths[file_index],
                        &output_filename,
                    )
                } else {
                    job.filesystem.staging_dir.join("output").join(&output_filename)
                };

                if let Some(parent) = write_path.parent() {
                    let _ = tokio::fs::create_dir_all(parent).await;
                }
                if let Err(err) = tokio::fs::write(&write_path, &chat_output).await {
                    warn!(error = %err, "Failed to write benchmark CHAT output");
                }

                let csv_path = write_path.with_extension("compare.csv");
                if let Err(err) = tokio::fs::write(&csv_path, &csv_output).await {
                    warn!(error = %err, "Failed to write benchmark CSV output");
                }

                lifecycle
                    .complete_with_result(output_filename.into(), ContentType::Chat, finished_at)
                    .await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
            Err(err) => {
                let finished_at = unix_now();
                let category = classify_server_error(&err);
                let raw_msg = format!("Benchmark failed: {err}");
                warn!(
                    job_id = %job_id,
                    filename,
                    category = %category,
                    raw_error = %raw_msg,
                    "Benchmark error (raw)"
                );
                let err_msg = user_facing_error(category, "Benchmark", filename, &raw_msg);
                let has_retry_budget = attempt_number < retry_policy.max_attempts;

                if matches!(&err, crate::error::ServerError::Worker(_))
                    && is_retryable_worker_failure(category)
                    && has_retry_budget
                {
                    let retry_number = attempt_number;
                    let backoff_ms = retry_policy.backoff_for_retry(retry_number);
                    let retry_at = UnixTimestamp(finished_at.0 + (backoff_ms.0 as f64 / 1000.0));
                    lifecycle
                        .retry(retry_at, category, &err_msg, finished_at)
                        .await;
                    continue;
                }

                lifecycle.fail(&err_msg, category, finished_at).await;
                return FileTaskOutcome::TerminalStateRecorded;
            }
        }
    }

    FileTaskOutcome::MissingTerminalState
}

fn resolve_benchmark_original_audio_path(
    filesystem: &crate::store::RunnerFilesystemConfig,
    file_index: usize,
    filename: &str,
) -> PathBuf {
    filesystem
        .source_paths
        .get(file_index)
        .cloned()
        .unwrap_or_else(|| filesystem.source_dir.join(filename))
}

#[cfg(test)]
mod tests {
    use super::resolve_benchmark_original_audio_path;
    use crate::store::RunnerFilesystemConfig;
    use std::path::PathBuf;

    fn filesystem_config(source_paths: Vec<&str>, source_dir: &str) -> RunnerFilesystemConfig {
        RunnerFilesystemConfig {
            paths_mode: false,
            source_paths: source_paths.into_iter().map(PathBuf::from).collect(),
            output_paths: Vec::new(),
            before_paths: Vec::new(),
            staging_dir: PathBuf::from("/tmp/staging"),
            media_mapping: String::new(),
            media_subdir: String::new(),
            source_dir: PathBuf::from(source_dir),
        }
    }

    #[test]
    fn resolve_benchmark_original_audio_path_prefers_explicit_source_path() {
        let filesystem = filesystem_config(vec!["/tmp/input/clip.mp3"], "/tmp/source");

        let path = resolve_benchmark_original_audio_path(&filesystem, 0, "clip.mp3");

        assert_eq!(path, PathBuf::from("/tmp/input/clip.mp3"));
    }

    #[test]
    fn resolve_benchmark_original_audio_path_falls_back_to_source_dir() {
        let filesystem = filesystem_config(Vec::new(), "/tmp/source");

        let path = resolve_benchmark_original_audio_path(&filesystem, 0, "clip.mp3");

        assert_eq!(path, PathBuf::from("/tmp/source/clip.mp3"));
    }
}
