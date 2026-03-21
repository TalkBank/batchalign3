//! Per-job async task — port of `JobStore._run_job` from Python.
//!
//! Each job runs as a `tokio::spawn` task. It acquires a semaphore permit,
//! then processes files concurrently (bounded by `compute_job_workers`) via
//! the WorkerPool.

pub(crate) mod debug_dumper;
mod dispatch;
mod policy;
pub(crate) mod util;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use crate::api::{ContentType, EngineVersion, FileName, JobId, RevAiJobId, UnixTimestamp};
use crate::host_memory::{HostMemoryCoordinator, HostMemoryError};
use crate::cache::UtteranceCache;
use crate::pipeline::PipelineServices;
use crate::revai::{RevAiPreflightPlan, preflight_submit_audio_paths};
use crate::scheduling::{DurationMs, FailureCategory, RetryPolicy, WorkUnitKind};
use crate::worker::InferTask;
use crate::worker::pool::WorkerPool;
use crate::worker::target::task_name as infer_task_name;
use tracing::{error, info, warn};

use crate::api::JobStatus;
use crate::queue::QueueBackend;
use crate::store::{JobStore, LeaseRenewalOutcome, RunnerJobSnapshot, unix_now};
use crate::workflow::{RunnerDispatchKind, command_runner_dispatch_kind};

use dispatch::{
    BatchedInferDispatchPlan, BenchmarkDispatchPlan, BenchmarkDispatchRuntime, FaDispatchPlan,
    FaDispatchRuntime, MediaAnalysisDispatchPlan, MediaAnalysisDispatchRuntime,
    TranscribeDispatchPlan, TranscribeDispatchRuntime, dispatch_batched_infer,
    dispatch_benchmark_infer, dispatch_fa_infer, dispatch_media_analysis_v2,
    dispatch_transcribe_infer,
};
use policy::{command_requires_infer, infer_task_for_command, result_filename_for_command};
use util::{
    FileRunTracker, FileStage, apply_result_filename, collect_preflight_audio_paths,
    compute_job_workers, force_terminal_file_states, preflight_validate_media, should_preflight,
};

/// Shared dependencies needed to build per-job runner tasks.
#[derive(Clone)]
pub(crate) struct RunnerContext {
    store: Arc<JobStore>,
    pool: Arc<WorkerPool>,
    cache: Arc<UtteranceCache>,
    infer_tasks: Vec<InferTask>,
    engine_versions: BTreeMap<String, String>,
    test_echo_mode: bool,
    queue: Arc<dyn QueueBackend>,
}

impl RunnerContext {
    /// Create the shared dependency bundle for per-job runner tasks.
    pub(crate) fn new(
        store: Arc<JobStore>,
        pool: Arc<WorkerPool>,
        cache: Arc<UtteranceCache>,
        infer_tasks: Vec<InferTask>,
        engine_versions: BTreeMap<String, String>,
        test_echo_mode: bool,
        queue: Arc<dyn QueueBackend>,
    ) -> Self {
        Self {
            store,
            pool,
            cache,
            infer_tasks,
            engine_versions,
            test_echo_mode,
            queue,
        }
    }
}

/// Build the full future that owns one background job lifecycle.
pub(crate) async fn job_task(job_id: JobId, context: RunnerContext) {
    let store_for_release = context.store.clone();
    let lease_store = context.store.clone();
    let lease_job_id = job_id.clone();
    let correlation_id = context
        .store
        .runner_snapshot(&job_id)
        .await
        .map(|snapshot| snapshot.identity.correlation_id)
        .unwrap_or_else(|| job_id.to_string().into());
    let lease_task = tokio::spawn(async move {
        let interval = std::time::Duration::from_secs(JobStore::LOCAL_LEASE_HEARTBEAT_S);
        loop {
            tokio::time::sleep(interval).await;
            if lease_store.renew_job_lease(&lease_job_id).await == LeaseRenewalOutcome::Stop {
                break;
            }
        }
    });
    if let Err(e) = run_job(&job_id, &context).await {
        error!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            error = %e,
            "Job failed with server error"
        );
    }
    lease_task.abort();
    store_for_release.release_runner_claim(&job_id).await;
}

/// Run the server-owned processing lifecycle for one job.
async fn run_job(job_id: &JobId, context: &RunnerContext) -> Result<(), crate::error::ServerError> {
    let store = &context.store;
    let pool = &context.pool;
    let cache = &context.cache;
    let infer_tasks = &context.infer_tasks;
    let engine_versions = &context.engine_versions;
    let queue = context.queue.as_ref();
    let test_echo_mode = context.test_echo_mode;

    let Some(job) = store.runner_snapshot(job_id).await else {
        return Ok(());
    };
    let job = Arc::new(job);
    let correlation_id = job.identity.correlation_id.clone();

    if job.cancel_token.is_cancelled() {
        return Ok(());
    }

    let context = format!("job {job_id} pre-start");
    let memory_gate_policy = RetryPolicy {
        max_attempts: 1,
        initial_backoff_ms: DurationMs(30_000),
        max_backoff_ms: DurationMs(120_000),
        backoff_multiplier: 2,
    };

    // Acquire semaphore (blocks if too many concurrent jobs)
    let _permit = store.acquire_job_slot().await.expect("semaphore closed");

    // Check cancellation after acquiring permit
    if job.cancel_token.is_cancelled() {
        return Ok(());
    }

    // Collect file list to process
    let mut file_list = job.pending_files.clone();
    let command = job.dispatch.command.clone();

    // Compute the requested per-job file parallelism without host-memory math,
    // then let the host-memory coordinator clamp it to what the machine can
    // safely support right now.
    let requested_workers = compute_job_workers(&command, file_list.len(), store.config());
    let coordinator = HostMemoryCoordinator::from_server_config(store.config());
    let job_label = format!(
        "job-execution:{}:{}:{}",
        job_id,
        command,
        job.dispatch.lang.to_worker_language()
    );
    let planner_command = command.clone();
    let timeout = Duration::from_secs(store.config().memory_gate_timeout_s);
    let poll_interval = Duration::from_secs(store.config().memory_gate_poll_s.max(1));
    let execution_plan = tokio::task::spawn_blocking(move || {
        coordinator.wait_for_job_execution_plan(
            &planner_command,
            requested_workers,
            &job_label,
            timeout,
            poll_interval,
        )
    })
    .await
    .map_err(|error| {
        crate::error::ServerError::Persistence(format!(
            "host-memory planner task failed for job {job_id}: {error}"
        ))
    })?;
    let execution_plan = match execution_plan {
        Ok(plan) => plan,
        Err(error @ (HostMemoryError::CapacityRejected { .. } | HostMemoryError::TimedOut { .. })) => {
            let retry_at = UnixTimestamp(
                unix_now().0 + (memory_gate_policy.backoff_for_retry(1).0 as f64 / 1000.0),
            );
            store.requeue_job_after_memory_gate(job_id, retry_at).await;
            store.bump_counter(|c| c.deferred_work_units += 1).await;
            queue.notify();
            warn!(
                job_id = %job_id,
                correlation_id = %correlation_id,
                requested_workers = requested_workers.0,
                error = %error,
                retry_at = %retry_at,
                context = %context,
                "Re-queueing job after host-memory capacity rejection"
            );
            return Ok(());
        }
        Err(error) => {
            return Err(crate::error::ServerError::Persistence(format!(
                "host-memory coordinator failed for job {job_id}: {error}"
            )));
        }
    };
    let num_workers = execution_plan.granted_workers;
    let _job_memory_lease = execution_plan.lease;

    // Mark as running only after job execution memory has been reserved.
    store.mark_job_running(job_id).await;

    // Record on job and DB
    store.record_job_worker_count(job_id, num_workers.0).await;

    info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        num_files = file_list.len(),
        requested_workers = requested_workers.0,
        num_workers = num_workers.0,
        reserved_mb = execution_plan.reserved_mb.0,
        command = %command,
        "Processing files"
    );

    // Pre-validate media files (paths_mode only) to fail fast before worker dispatch.
    let media_failures = preflight_validate_media(
        &file_list,
        &job.filesystem.source_paths,
        job.filesystem.paths_mode,
    )
    .await;

    // Mark invalid files as errors immediately and collect the valid file list.
    file_list = if media_failures.is_empty() {
        file_list
    } else {
        let failed_indices =
            record_preflight_media_failures(store, job_id, &file_list, &media_failures).await;
        file_list
            .into_iter()
            .filter(|file| !failed_indices.contains(&file.file_index))
            .collect()
    };

    // Pre-scale workers before dispatch to avoid sequential spawn overhead.
    // For --lang auto, use "auto" as the worker lang — this matches the key
    // that dispatch_execute_v2 will use for the GPU worker. Engine overrides
    // from the job options are passed so the pre-scaled worker matches the
    // exact key that dispatch will look up.
    if *num_workers > 1 {
        let pre_scale_lang = job.dispatch.lang.to_worker_language();
        let job_engine_overrides = job.dispatch.options.common().engine_overrides_json();
        pool.pre_scale_with_overrides(
            &command,
            pre_scale_lang,
            num_workers.0,
            &job_engine_overrides,
        )
        .await;
    }

    // Preflight: pre-submit audio files to Rev.AI for parallel server-side processing.
    // This collects Rev.AI job IDs that individual file tasks will poll instead of
    // re-uploading, reducing total wall-clock time by 2-5x for large batches.
    let rev_job_ids: Arc<HashMap<PathBuf, RevAiJobId>> = {
        if should_preflight(&command, Some(&job.dispatch.options)) {
            let audio_paths = collect_preflight_audio_paths(&command, &job, &file_list).await;

            if !audio_paths.is_empty() {
                info!(
                    job_id = %job_id,
                    correlation_id = %correlation_id,
                    num_audio_files = audio_paths.len(),
                    "Starting Rev.AI preflight submission"
                );

                let preflight_plan = RevAiPreflightPlan {
                    audio_paths,
                    lang: job.dispatch.lang.clone(),
                    num_speakers: job.dispatch.num_speakers,
                    max_concurrent: 16usize,
                };

                match preflight_submit_audio_paths(&preflight_plan).await {
                    Ok(response) => {
                        if !response.errors.is_empty() {
                            warn!(
                                job_id = %job_id,
                                num_errors = response.errors.len(),
                                "Preflight had partial errors (will fall back to per-file)"
                            );
                        }
                        info!(
                            job_id = %job_id,
                            num_submitted = response.job_ids.len(),
                            "Preflight submission complete"
                        );
                        Arc::new(response.job_ids.into_iter().collect())
                    }
                    Err(e) => {
                        warn!(
                            job_id = %job_id,
                            error = %e,
                            "Preflight failed, falling back to per-file submission"
                        );
                        Arc::new(HashMap::new())
                    }
                }
            } else {
                Arc::new(HashMap::new())
            }
        } else {
            Arc::new(HashMap::new())
        }
    };

    // Choose between infer path or per-file dispatch.
    let all_chat = file_list.iter().all(|file| file.has_chat);
    let infer_task = infer_task_for_command(&command);
    let infer_supported = infer_task.is_some_and(|task| infer_tasks.contains(&task));
    let use_infer = all_chat && infer_supported;

    if command_requires_infer(&command, all_chat) && !use_infer {
        let required_task = infer_task.map(infer_task_name).unwrap_or("unknown");
        let err_msg = format!(
            "Rust-first dispatch requires infer task '{}' for '{}' (all_chat={}). \
             Worker advertises infer_tasks: {:?}",
            required_task, command, all_chat, infer_tasks
        );
        warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
        let failed_at = unix_now();
        store.fail_job(job_id, &err_msg, failed_at).await;
        return Ok(());
    }

    // Special case: transcribe/transcribe_s with server-side ASR orchestration.
    // These commands take audio input (not CHAT), so they do not go through the
    // standard `use_infer` path which requires all_chat=true.
    let runner_dispatch_kind = command_runner_dispatch_kind(&command);
    let use_transcribe_infer = matches!(
        runner_dispatch_kind,
        Some(RunnerDispatchKind::TranscribeAudioInfer)
    ) && infer_tasks.contains(&InferTask::Asr);
    let use_benchmark_infer = matches!(
        runner_dispatch_kind,
        Some(RunnerDispatchKind::BenchmarkAudioInfer)
    ) && infer_tasks.contains(&InferTask::Asr);
    let use_media_analysis_infer = matches!(
        runner_dispatch_kind,
        Some(RunnerDispatchKind::MediaAnalysisV2)
    ) && infer_task.is_some_and(|task| infer_tasks.contains(&task));

    if test_echo_mode {
        dispatch_test_echo_files(&job, store, &file_list).await;
    } else if use_transcribe_infer {
        let engine_version = EngineVersion::from(
            engine_versions
                .get("asr")
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );

        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side transcribe orchestrator"
        );

        let Some(plan) = TranscribeDispatchPlan::from_job(&job) else {
            warn!(
                job_id = %job_id,
                correlation_id = %correlation_id,
                command = %command,
                "Transcribe dispatch plan could not be built from job options"
            );
            return Ok(());
        };

        dispatch_transcribe_infer(
            &job,
            store,
            TranscribeDispatchRuntime {
                pool: pool.clone(),
                cache: cache.clone(),
                engine_version: engine_version.clone(),
                rev_job_ids: rev_job_ids.clone(),
                num_workers,
            },
            plan,
        )
        .await;
    } else if use_benchmark_infer {
        let engine_version = EngineVersion::from(
            engine_versions
                .get("asr")
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );

        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side benchmark orchestrator"
        );

        let Some(plan) = BenchmarkDispatchPlan::from_job(&job) else {
            warn!(
                job_id = %job_id,
                correlation_id = %correlation_id,
                command = %command,
                "Benchmark dispatch plan could not be built from job options"
            );
            return Ok(());
        };

        dispatch_benchmark_infer(
            &job,
            store,
            BenchmarkDispatchRuntime {
                pool: pool.clone(),
                cache: cache.clone(),
                engine_version: engine_version.clone(),
                rev_job_ids: rev_job_ids.clone(),
                num_workers,
            },
            plan,
        )
        .await;
    } else if use_media_analysis_infer {
        let Some(infer_task) = infer_task else {
            tracing::error!("use_media_analysis_infer set but infer_task is None — logic error");
            return Ok(());
        };
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(infer_task))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );
        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side media-analysis V2 path"
        );

        let Some(plan) = MediaAnalysisDispatchPlan::from_job(&job) else {
            warn!(
                job_id = %job_id,
                correlation_id = %correlation_id,
                command = %command,
                "Media-analysis dispatch plan could not be built from job options"
            );
            return Ok(());
        };

        dispatch_media_analysis_v2(
            &job,
            store,
            MediaAnalysisDispatchRuntime {
                pool: pool.clone(),
                num_workers,
            },
            plan,
        )
        .await;
    } else if use_infer {
        // --- Server-side infer path ---
        // The server owns CHAT parse/cache/inject/serialize.
        // Python workers provide pure Stanza inference only.
        let Some(infer_task) = infer_task else {
            // use_infer requires infer_task.is_some() — this branch is unreachable
            // but we avoid a panic by returning early with an error log.
            tracing::error!("use_infer set but infer_task is None — logic error");
            return Ok(());
        };
        let engine_version = EngineVersion::from(
            engine_versions
                .get(infer_task_name(infer_task))
                .map(|s| s.as_str())
                .unwrap_or("unknown"),
        );

        info!(
            job_id = %job_id,
            correlation_id = %correlation_id,
            command = %command,
            engine_version = %engine_version,
            "Using server-side infer path"
        );

        if command == "align" {
            // FA is per-file (each file has its own audio) — not cross-file batchable.
            // Files are processed concurrently, bounded by num_workers.
            let Some(plan) = FaDispatchPlan::from_job(&job) else {
                warn!(
                    job_id = %job_id,
                    correlation_id = %correlation_id,
                    command = %command,
                    "FA dispatch plan could not be built from job options"
                );
                return Ok(());
            };

            dispatch_fa_infer(
                &job,
                store,
                FaDispatchRuntime {
                    pool: pool.clone(),
                    cache: cache.clone(),
                    engine_version: engine_version.clone(),
                    rev_job_ids: rev_job_ids.clone(),
                    num_workers,
                },
                plan,
            )
            .await;
        } else {
            let plan = BatchedInferDispatchPlan::from_job(&job);
            dispatch_batched_infer(
                &job,
                store,
                PipelineServices {
                    pool,
                    cache,
                    engine_version: &engine_version,
                },
                plan,
            )
            .await;
        }
    } else {
        let err_msg = format!(
            "No released dispatch path remains for command '{}' (all_chat={}, infer_task={:?}, infer_supported={}). Legacy process-path fallback is retired.",
            command, all_chat, infer_task, infer_supported
        );
        warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
        store.fail_job(job_id, &err_msg, unix_now()).await;
        return Ok(());
    }

    // Force unfinished files to terminal status
    let forced_errors = force_terminal_file_states(store, job_id).await;

    // Set final job status
    let Some(completion) = store.completion_snapshot(job_id).await else {
        return Ok(());
    };

    let completed_at = unix_now();
    let final_status = if completion.cancelled {
        JobStatus::Cancelled
    } else if forced_errors > 0 || completion.all_failed {
        JobStatus::Failed
    } else {
        JobStatus::Completed
    };

    store.finalize_job(job_id, final_status, completed_at).await;

    info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        status = %final_status,
        "Job finished"
    );

    Ok(())
}

async fn dispatch_test_echo_files(
    job: &RunnerJobSnapshot,
    store: &JobStore,
    file_list: &[crate::store::PendingJobFile],
) {
    let job_id = &job.identity.job_id;

    for file in file_list {
        if job.cancel_token.is_cancelled() {
            break;
        }

        let filename = file.filename.as_ref();
        let lifecycle = FileRunTracker::new(store, job_id, filename);
        let started_at = unix_now();
        lifecycle
            .begin_first_attempt(WorkUnitKind::FileProcess, started_at, FileStage::Processing)
            .await;

        let result_filename = result_filename_for_command(&job.dispatch.command, filename);

        let output_text = if file.has_chat {
            let read_path = if job.filesystem.paths_mode
                && file.file_index < job.filesystem.source_paths.len()
            {
                job.filesystem.source_paths[file.file_index].clone()
            } else {
                job.filesystem.staging_dir.join("input").join(filename)
            };
            match tokio::fs::read_to_string(&read_path).await {
                Ok(content) => content,
                Err(error) => {
                    let err_msg = format!("Failed to read input for test-echo dispatch: {error}");
                    lifecycle
                        .fail(&err_msg, FailureCategory::InputMissing, unix_now())
                        .await;
                    continue;
                }
            }
        } else {
            "@UTF8\n@Begin\n@End\n".to_string()
        };

        let write_path =
            if job.filesystem.paths_mode && file.file_index < job.filesystem.output_paths.len() {
                apply_result_filename(
                    &job.filesystem.output_paths[file.file_index],
                    &result_filename,
                )
            } else {
                job.filesystem
                    .staging_dir
                    .join("output")
                    .join(&result_filename)
            };

        if let Some(parent) = write_path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }

        if let Err(error) = tokio::fs::write(&write_path, output_text).await {
            let err_msg = format!("Failed to write test-echo output: {error}");
            lifecycle
                .fail(&err_msg, FailureCategory::Validation, unix_now())
                .await;
            continue;
        }

        lifecycle
            .complete_with_result(
                FileName::from(result_filename),
                ContentType::Chat,
                unix_now(),
            )
            .await;
    }
}

/// Record media-prevalidation failures as explicit setup attempts before the
/// job enters any concrete dispatch path.
async fn record_preflight_media_failures(
    store: &JobStore,
    job_id: &JobId,
    file_list: &[crate::store::PendingJobFile],
    media_failures: &HashMap<usize, String>,
) -> HashSet<usize> {
    let now = unix_now();
    let mut failed_indices = HashSet::with_capacity(media_failures.len());

    for (&idx, err_msg) in media_failures {
        failed_indices.insert(idx);
        if let Some(file) = file_list.iter().find(|file| file.file_index == idx) {
            FileRunTracker::new(store, job_id, &file.filename)
                .record_setup_failure(now, err_msg, FailureCategory::Validation, now)
                .await;
        }
    }

    failed_indices
}


#[cfg(test)]
mod tests;
