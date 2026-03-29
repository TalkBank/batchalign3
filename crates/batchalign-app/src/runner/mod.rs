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

use async_trait::async_trait;

use crate::api::{
    ContentType, DisplayPath, EngineVersion, JobId, NumWorkers, ReleasedCommand, RevAiJobId,
    UnixTimestamp,
};
use crate::cache::UtteranceCache;
use crate::commands::{RunnerDispatchKind, command_runner_dispatch_kind};
use crate::config::ServerConfig;
use crate::host_memory::{HostMemoryCoordinator, HostMemoryError, JobExecutionPlan};
use crate::pipeline::PipelineServices;
use crate::revai::{RevAiPreflightPlan, preflight_submit_audio_paths};
use crate::scheduling::{FailureCategory, WorkUnitKind};
use crate::worker::InferTask;
use crate::worker::pool::WorkerPool;
use crate::worker::target::task_name as infer_task_name;
use tracing::{error, info, warn};

use crate::api::JobStatus;
use crate::state::resolve_worker_capability_snapshot;
use crate::store::{JobStore, LeaseRenewalOutcome, RunnerJobSnapshot, unix_now};

pub(crate) use dispatch::{
    BatchedInferDispatchPlan, BenchmarkDispatchPlan, BenchmarkDispatchRuntime, FaDispatchPlan,
    FaDispatchRuntime, MediaAnalysisDispatchPlan, MediaAnalysisDispatchRuntime,
    TranscribeDispatchPlan, TranscribeDispatchRuntime, dispatch_batched_infer,
    dispatch_benchmark_infer, dispatch_fa_infer, dispatch_media_analysis_v2,
    dispatch_transcribe_infer,
};
use policy::{command_requires_infer, infer_task_for_command, result_filename_for_command};
use util::{
    FileRunTracker, FileStage, RunnerEventSink, StoreRunnerEventSink, apply_result_filename,
    collect_preflight_audio_paths, compute_job_workers, force_terminal_file_states,
    preflight_validate_media, should_preflight,
};

/// Shared dependencies needed to build per-job runner tasks.
#[derive(Clone)]
pub(crate) struct RunnerExecutionContext {
    pool: Arc<WorkerPool>,
    cache: Arc<UtteranceCache>,
    infer_tasks: Vec<InferTask>,
    engine_versions: BTreeMap<String, String>,
    test_echo_mode: bool,
}

impl RunnerExecutionContext {
    pub(crate) fn new(
        pool: Arc<WorkerPool>,
        cache: Arc<UtteranceCache>,
        infer_tasks: Vec<InferTask>,
        engine_versions: BTreeMap<String, String>,
        test_echo_mode: bool,
    ) -> Self {
        Self {
            pool,
            cache,
            infer_tasks,
            engine_versions,
            test_echo_mode,
        }
    }
}

/// Shared execution engine built on one resolved runtime context.
#[derive(Clone)]
pub(crate) struct ExecutionEngine {
    context: RunnerExecutionContext,
}

impl ExecutionEngine {
    /// Build one shared execution engine over a resolved runtime context.
    pub(crate) fn new(context: RunnerExecutionContext) -> Self {
        Self { context }
    }

    async fn dispatch_job(
        &self,
        request: JobDispatchRequest,
        host: &DispatchHostContext,
    ) -> Result<(), crate::error::ServerError> {
        dispatch_job_with_execution_context(request, host, &self.context).await
    }
}

/// Read-only host/runtime context consulted during shared command dispatch.
///
/// This keeps performance policy and media-resolution config explicit host
/// concerns without threading the full mutable `JobStore` through command code.
#[derive(Clone)]
pub(crate) struct DispatchHostContext {
    store: Arc<JobStore>,
    config: Arc<ServerConfig>,
    sink: Arc<dyn RunnerEventSink>,
}

impl DispatchHostContext {
    fn from_store(store: Arc<JobStore>) -> Self {
        Self {
            store: store.clone(),
            config: Arc::new(store.config().clone()),
            sink: StoreRunnerEventSink::new(store),
        }
    }

    pub(crate) fn config(&self) -> &ServerConfig {
        self.config.as_ref()
    }

    pub(crate) fn sink(&self) -> &Arc<dyn RunnerEventSink> {
        &self.sink
    }

    pub(crate) fn media_mapping_root(
        &self,
        key: &str,
    ) -> Option<&batchalign_types::paths::ServerPath> {
        self.config
            .media_mappings
            .get(&batchalign_types::paths::MediaMappingKey::new(key))
    }

    pub(crate) fn media_roots(&self) -> &[batchalign_types::paths::ServerPath] {
        &self.config.media_roots
    }

    pub(crate) fn trace_store(&self) -> &crate::trace_store::TraceStore {
        self.store.trace_store()
    }
}

/// Shared server-owned host dependencies needed to build per-job runner tasks.
#[derive(Clone)]
pub(crate) struct ServerExecutionHost {
    store: Arc<JobStore>,
    engine: ExecutionEngine,
    orchestrator: Arc<dyn QueuedJobOrchestrator>,
}

impl ServerExecutionHost {
    /// Build the server-owned host bundle around one execution engine.
    pub(crate) fn new(
        store: Arc<JobStore>,
        engine: ExecutionEngine,
        orchestrator: Arc<dyn QueuedJobOrchestrator>,
    ) -> Self {
        Self {
            store,
            engine,
            orchestrator,
        }
    }
}

/// Shared direct-execution host dependencies needed to run one inline job.
#[derive(Clone)]
pub(crate) struct DirectExecutionHost {
    store: Arc<JobStore>,
    engine: ExecutionEngine,
}

impl DirectExecutionHost {
    /// Build one direct-execution host bundle around one execution engine.
    pub(crate) fn new(store: Arc<JobStore>, engine: ExecutionEngine) -> Self {
        Self { store, engine }
    }
}

/// Execution-phase request handed from the host-owned runner wrapper into the
/// shared dispatch kernel.
struct JobDispatchRequest {
    job: Arc<RunnerJobSnapshot>,
    file_list: Vec<crate::store::PendingJobFile>,
    num_workers: NumWorkers,
    rev_job_ids: Arc<HashMap<PathBuf, RevAiJobId>>,
}

/// Host-memory reservation failures separated from the rest of job execution.
enum ExecutionReservationError {
    Capacity {
        requested_workers: NumWorkers,
        error: HostMemoryError,
    },
    Fatal(crate::error::ServerError),
}

/// Host-owned orchestration decision after a memory-gate rejection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum MemoryGateRejectionDisposition {
    /// The host re-queued the job for a later eligibility deadline.
    Requeued { retry_at: UnixTimestamp },
    /// The host wants the runner to fail the job instead of re-queueing it.
    FailJob,
}

/// Result of one host-owned job execution attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum HostedJobRunOutcome {
    /// The job finished its current lifecycle attempt.
    Completed,
    /// The host deferred the job for a later eligibility deadline.
    Requeued { retry_at: UnixTimestamp },
}

/// Host-owned orchestration seam for queued job execution policy.
///
/// This keeps the shared runner from knowing how a specific server backend
/// persists re-queues, computes backoff, or wakes its scheduler when a job
/// cannot currently run.
#[async_trait]
pub(crate) trait QueuedJobOrchestrator: Send + Sync {
    /// Handle a host-memory rejection for one queued job.
    async fn handle_memory_gate_rejection(
        &self,
        sink: &Arc<dyn RunnerEventSink>,
        job_id: &JobId,
        requested_workers: NumWorkers,
        error: &HostMemoryError,
    ) -> Result<MemoryGateRejectionDisposition, crate::error::ServerError>;
}

enum MemoryGateFailurePolicy {
    Queued {
        orchestrator: Arc<dyn QueuedJobOrchestrator>,
    },
    FailJob,
}

/// Build the full future that owns one background job lifecycle.
pub(crate) async fn job_task(job_id: JobId, host: ServerExecutionHost) {
    let store_for_release = host.store.clone();
    let lease_store = host.store.clone();
    let lease_job_id = job_id.clone();
    let correlation_id = host
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
    if let Err(e) = run_server_job_attempt(&job_id, &host).await {
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
pub(crate) async fn run_server_job_attempt(
    job_id: &JobId,
    host: &ServerExecutionHost,
) -> Result<HostedJobRunOutcome, crate::error::ServerError> {
    run_hosted_job(
        job_id,
        &host.store,
        &host.engine,
        MemoryGateFailurePolicy::Queued {
            orchestrator: host.orchestrator.clone(),
        },
    )
    .await
}

/// Run one direct inline job through the shared execution engine.
pub(crate) async fn run_direct_job(
    job_id: &JobId,
    host: &DirectExecutionHost,
) -> Result<(), crate::error::ServerError> {
    match run_hosted_job(
        job_id,
        &host.store,
        &host.engine,
        MemoryGateFailurePolicy::FailJob,
    )
    .await?
    {
        HostedJobRunOutcome::Completed => Ok(()),
        HostedJobRunOutcome::Requeued { retry_at } => Err(crate::error::ServerError::Persistence(
            format!("direct execution unexpectedly requeued job {job_id} until {retry_at}"),
        )),
    }
}

async fn run_hosted_job(
    job_id: &JobId,
    store: &Arc<JobStore>,
    engine: &ExecutionEngine,
    memory_gate_policy: MemoryGateFailurePolicy,
) -> Result<HostedJobRunOutcome, crate::error::ServerError> {
    let pool = &engine.context.pool;
    let host_context = DispatchHostContext::from_store(store.clone());
    let sink = host_context.sink().clone();

    let Some(job) = store.runner_snapshot(job_id).await else {
        return Ok(HostedJobRunOutcome::Completed);
    };
    let job = Arc::new(job);
    let correlation_id = job.identity.correlation_id.clone();

    if job.cancel_token.is_cancelled() {
        return Ok(HostedJobRunOutcome::Completed);
    }

    // Acquire semaphore (blocks if too many concurrent jobs)
    let _permit = store.acquire_job_slot().await.expect("semaphore closed");

    // Check cancellation after acquiring permit
    if job.cancel_token.is_cancelled() {
        return Ok(HostedJobRunOutcome::Completed);
    }

    // Collect file list to process
    let mut file_list = job.pending_files.clone();
    let command = job.dispatch.command;
    let execution_plan = match reserve_job_execution(&host_context, job_id, &job).await {
        Ok(plan) => plan,
        Err(ExecutionReservationError::Capacity {
            requested_workers,
            error,
        }) => match &memory_gate_policy {
            MemoryGateFailurePolicy::Queued { orchestrator } => {
                let disposition = orchestrator
                    .handle_memory_gate_rejection(&sink, job_id, requested_workers, &error)
                    .await?;
                match disposition {
                    MemoryGateRejectionDisposition::Requeued { retry_at } => {
                        warn!(
                            job_id = %job_id,
                            correlation_id = %correlation_id,
                            requested_workers = requested_workers.0,
                            error = %error,
                            retry_at = %retry_at,
                            "Re-queueing job after host-memory capacity rejection"
                        );
                        return Ok(HostedJobRunOutcome::Requeued { retry_at });
                    }
                    MemoryGateRejectionDisposition::FailJob => {
                        let message = error.to_string();
                        sink.bump_memory_gate_aborts().await;
                        sink.fail_job(job_id, &message, unix_now()).await;
                        return Err(crate::error::ServerError::MemoryPressure(message));
                    }
                }
            }
            MemoryGateFailurePolicy::FailJob => {
                let message = error.to_string();
                sink.bump_memory_gate_aborts().await;
                sink.fail_job(job_id, &message, unix_now()).await;
                return Err(crate::error::ServerError::MemoryPressure(message));
            }
        },
        Err(ExecutionReservationError::Fatal(error)) => {
            let message = error.to_string();
            sink.fail_job(job_id, &message, unix_now()).await;
            return Err(error);
        }
    };
    let num_workers = execution_plan.granted_workers;
    let _job_memory_lease = execution_plan.lease;

    // Mark as running only after job execution memory has been reserved.
    sink.mark_job_running(job_id).await;

    // Record on job and DB
    sink.record_job_worker_count(job_id, num_workers.0).await;

    info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        num_files = file_list.len(),
        requested_workers = execution_plan.requested_workers.0,
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
            record_preflight_media_failures(sink.as_ref(), job_id, &file_list, &media_failures)
                .await;
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
    let job_engine_overrides = job.dispatch.options.common().engine_overrides_json();

    if *num_workers > 1 {
        let pre_scale_lang = job.dispatch.lang.to_worker_language();
        pool.pre_scale_with_overrides(
            command,
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
        if should_preflight(command, Some(&job.dispatch.options)) {
            let audio_paths = collect_preflight_audio_paths(command, &job, &file_list).await;

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

    if let Err(error) = engine
        .dispatch_job(
            JobDispatchRequest {
                job: job.clone(),
                file_list,
                num_workers,
                rev_job_ids,
            },
            &host_context,
        )
        .await
    {
        let message = error.to_string();
        sink.fail_job(job_id, &message, unix_now()).await;
        return Err(error);
    }

    // Force unfinished files to terminal status
    let forced_errors = force_terminal_file_states(sink.as_ref(), job_id).await;

    // Set final job status
    let Some(completion) = store.completion_snapshot(job_id).await else {
        return Ok(HostedJobRunOutcome::Completed);
    };

    let completed_at = unix_now();
    let final_status = if completion.cancelled {
        JobStatus::Cancelled
    } else if forced_errors > 0 || completion.all_failed {
        JobStatus::Failed
    } else {
        JobStatus::Completed
    };

    sink.finalize_job(job_id, final_status, completed_at).await;

    info!(
        job_id = %job_id,
        correlation_id = %correlation_id,
        status = %final_status,
        "Job finished"
    );

    Ok(HostedJobRunOutcome::Completed)
}

async fn reserve_job_execution(
    host: &DispatchHostContext,
    job_id: &JobId,
    job: &RunnerJobSnapshot,
) -> Result<JobExecutionPlan, ExecutionReservationError> {
    let command = job.dispatch.command;
    let requested_workers = compute_job_workers(command, job.pending_files.len(), host.config());
    let coordinator = HostMemoryCoordinator::from_server_config(host.config());
    let job_label = format!(
        "job-execution:{}:{}:{}",
        job_id,
        command,
        job.dispatch.lang.to_worker_language()
    );
    let timeout = Duration::from_secs(host.config().memory_gate_timeout_s);
    let poll_interval = Duration::from_secs(host.config().memory_gate_poll_s.max(1));
    let plan = tokio::task::spawn_blocking(move || {
        coordinator.wait_for_job_execution_plan(
            command,
            requested_workers,
            &job_label,
            timeout,
            poll_interval,
        )
    })
    .await
    .map_err(|error| {
        ExecutionReservationError::Fatal(crate::error::ServerError::Persistence(format!(
            "host-memory planner task failed for job {job_id}: {error}"
        )))
    })?;

    match plan {
        Ok(plan) => Ok(plan),
        Err(
            error @ (HostMemoryError::CapacityRejected { .. } | HostMemoryError::TimedOut { .. }),
        ) => Err(ExecutionReservationError::Capacity {
            requested_workers,
            error,
        }),
        Err(error) => Err(ExecutionReservationError::Fatal(
            crate::error::ServerError::Persistence(format!(
                "host-memory coordinator failed for job {job_id}: {error}"
            )),
        )),
    }
}

async fn dispatch_job_with_execution_context(
    request: JobDispatchRequest,
    host: &DispatchHostContext,
    execution: &RunnerExecutionContext,
) -> Result<(), crate::error::ServerError> {
    let sink = host.sink().clone();
    let JobDispatchRequest {
        job,
        file_list,
        num_workers,
        rev_job_ids,
    } = request;
    let job_id = &job.identity.job_id;
    let correlation_id = job.identity.correlation_id.clone();
    let command = job.dispatch.command;
    let pool = &execution.pool;
    let cache = &execution.cache;
    let startup_infer_tasks = &execution.infer_tasks;
    let startup_engine_versions = &execution.engine_versions;
    let test_echo_mode = execution.test_echo_mode;
    let job_engine_overrides = job.dispatch.options.common().engine_overrides_json();

    // Choose between infer path or per-file dispatch.
    let capability_snapshot = match resolve_runtime_capability_snapshot(
        pool,
        startup_infer_tasks,
        startup_engine_versions,
        test_echo_mode,
        command,
        job.dispatch.lang.to_worker_language(),
        &job_engine_overrides,
    )
    .await
    {
        Ok(snapshot) => snapshot,
        Err(err_msg) => {
            warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
            sink.fail_job(job_id, &err_msg, unix_now()).await;
            return Ok(());
        }
    };
    let infer_tasks = &capability_snapshot.infer_tasks;
    let engine_versions = &capability_snapshot.engine_versions;

    let all_chat = file_list.iter().all(|file| file.has_chat);
    let infer_task = infer_task_for_command(command);
    let infer_supported = infer_task.is_some_and(|task| infer_tasks.contains(&task));
    let use_infer = all_chat && infer_supported;

    if command_requires_infer(command) && !use_infer {
        let required_task = infer_task.map(infer_task_name).unwrap_or("unknown");
        let err_msg = format!(
            "Rust-first dispatch requires infer task '{}' for '{}' (all_chat={}). \
             Worker advertises infer_tasks: {:?}",
            required_task, command, all_chat, infer_tasks
        );
        warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
        let failed_at = unix_now();
        sink.fail_job(job_id, &err_msg, failed_at).await;
        return Ok(());
    }

    // Special case: transcribe/transcribe_s with server-side ASR orchestration.
    // These commands take audio input (not CHAT), so they do not go through the
    // standard `use_infer` path which requires all_chat=true.
    let runner_dispatch_kind = command_runner_dispatch_kind(command);
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
        dispatch_test_echo_files(&job, sink.as_ref(), &file_list).await;
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

        dispatch_transcribe_command(
            &job,
            host,
            pool,
            cache,
            &engine_version,
            &rev_job_ids,
            num_workers,
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

        dispatch_benchmark_command(
            &job,
            host,
            pool,
            cache,
            &engine_version,
            &rev_job_ids,
            num_workers,
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

        dispatch_media_analysis_command(&job, host, pool, num_workers).await;
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

        match runner_dispatch_kind {
            Some(RunnerDispatchKind::ForcedAlignment) => {
                dispatch_forced_alignment_command(
                    &job,
                    host,
                    pool,
                    cache,
                    &engine_version,
                    &rev_job_ids,
                    num_workers,
                )
                .await;
            }
            Some(RunnerDispatchKind::BatchedTextInfer) => {
                dispatch_batched_text_command(&job, host, pool, cache, &engine_version).await;
            }
            other => {
                tracing::error!(
                    job_id = %job_id,
                    correlation_id = %correlation_id,
                    command = %command,
                    runner_dispatch_kind = ?other,
                    "Infer path selected for unsupported command dispatch kind"
                );
                return Ok(());
            }
        }
    } else {
        let err_msg = format!(
            "No released dispatch path remains for command '{}' (all_chat={}, infer_task={:?}, infer_supported={}). Legacy process-path fallback is retired.",
            command, all_chat, infer_task, infer_supported
        );
        warn!(job_id = %job_id, correlation_id = %correlation_id, "{}", err_msg);
        sink.fail_job(job_id, &err_msg, unix_now()).await;
        return Ok(());
    }

    Ok(())
}

fn warn_invalid_dispatch_plan(job: &RunnerJobSnapshot) {
    warn!(
        job_id = %job.identity.job_id,
        correlation_id = %job.identity.correlation_id,
        command = %job.dispatch.command,
        "Command plan could not be built from job options"
    );
}

fn debug_dumper_for_job(job: &RunnerJobSnapshot) -> crate::runner::debug_dumper::DebugDumper {
    crate::runner::debug_dumper::DebugDumper::new(
        job.dispatch
            .options
            .common()
            .debug_dir
            .as_deref()
            .map(std::path::Path::new),
    )
}

async fn dispatch_batched_text_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    engine_version: &EngineVersion,
) {
    let plan = BatchedInferDispatchPlan::from_job(job, host.config());
    let dumper = debug_dumper_for_job(job);
    dispatch_batched_infer(
        job,
        host,
        PipelineServices::with_debug(pool, cache, engine_version, &dumper),
        plan,
    )
    .await;
}

async fn dispatch_forced_alignment_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    engine_version: &EngineVersion,
    rev_job_ids: &Arc<HashMap<PathBuf, RevAiJobId>>,
    num_workers: NumWorkers,
) {
    let Some(plan) = FaDispatchPlan::from_job(job, host.config()) else {
        warn_invalid_dispatch_plan(job);
        return;
    };

    dispatch_fa_infer(
        job,
        host,
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
}

async fn dispatch_transcribe_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    engine_version: &EngineVersion,
    rev_job_ids: &Arc<HashMap<PathBuf, RevAiJobId>>,
    num_workers: NumWorkers,
) {
    let Some(plan) = TranscribeDispatchPlan::from_job(job, host.config()) else {
        warn_invalid_dispatch_plan(job);
        return;
    };

    dispatch_transcribe_infer(
        job,
        host,
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
}

async fn dispatch_benchmark_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    cache: &Arc<UtteranceCache>,
    engine_version: &EngineVersion,
    rev_job_ids: &Arc<HashMap<PathBuf, RevAiJobId>>,
    num_workers: NumWorkers,
) {
    let Some(plan) = BenchmarkDispatchPlan::from_job(job, host.config()) else {
        warn_invalid_dispatch_plan(job);
        return;
    };

    dispatch_benchmark_infer(
        job,
        host,
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
}

async fn dispatch_media_analysis_command(
    job: &RunnerJobSnapshot,
    host: &DispatchHostContext,
    pool: &Arc<WorkerPool>,
    num_workers: NumWorkers,
) {
    let Some(plan) = MediaAnalysisDispatchPlan::from_job(job, host.config()) else {
        warn_invalid_dispatch_plan(job);
        return;
    };

    dispatch_media_analysis_v2(
        job,
        host,
        MediaAnalysisDispatchRuntime {
            pool: pool.clone(),
            num_workers,
        },
        plan,
    )
    .await;
}

async fn resolve_runtime_capability_snapshot(
    pool: &WorkerPool,
    startup_infer_tasks: &[InferTask],
    startup_engine_versions: &BTreeMap<String, String>,
    test_echo_mode: bool,
    command: ReleasedCommand,
    lang: impl Into<crate::api::WorkerLanguage>,
    engine_overrides: &str,
) -> Result<crate::state::WorkerCapabilitySnapshot, String> {
    if !test_echo_mode
        && pool.detected_capabilities().is_none()
        && infer_task_for_command(command).is_some()
    {
        pool.ensure_command_capabilities_with_overrides(command, lang, engine_overrides)
            .await
            .map_err(|error| {
                format!(
                    "Failed to bootstrap live worker capabilities for '{}': {}",
                    command, error
                )
            })?;
    }

    resolve_worker_capability_snapshot(
        &[],
        startup_infer_tasks,
        startup_engine_versions,
        test_echo_mode,
        pool.detected_capabilities(),
    )
    .map_err(|error| error.to_string())
}

async fn dispatch_test_echo_files(
    job: &RunnerJobSnapshot,
    sink: &dyn util::RunnerEventSink,
    file_list: &[crate::store::PendingJobFile],
) {
    let job_id = &job.identity.job_id;

    for file in file_list {
        if job.cancel_token.is_cancelled() {
            break;
        }

        let filename = file.filename.as_ref();
        let lifecycle = FileRunTracker::new(sink, job_id, filename);
        let started_at = unix_now();
        lifecycle
            .begin_first_attempt(WorkUnitKind::FileProcess, started_at, FileStage::Processing)
            .await;

        let result_filename = result_filename_for_command(job.dispatch.command, filename);

        let output_text = if file.has_chat {
            let read_path: std::path::PathBuf = if job.filesystem.paths_mode
                && file.file_index < job.filesystem.source_paths.len()
            {
                job.filesystem.source_paths[file.file_index]
                    .assume_shared_filesystem()
                    .as_path()
                    .to_owned()
            } else {
                job.filesystem.staging_dir.join("input").join(filename)
                    .as_path()
                    .to_owned()
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
                    job.filesystem.output_paths[file.file_index]
                        .assume_shared_filesystem()
                        .as_path(),
                    &result_filename,
                )
            } else {
                job.filesystem
                    .staging_dir
                    .join("output")
                    .join(&result_filename)
                    .as_path()
                    .to_owned()
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
                DisplayPath::from(result_filename),
                ContentType::Chat,
                unix_now(),
            )
            .await;
    }
}

/// Record media-prevalidation failures as explicit setup attempts before the
/// job enters any concrete dispatch path.
async fn record_preflight_media_failures(
    sink: &dyn util::RunnerEventSink,
    job_id: &JobId,
    file_list: &[crate::store::PendingJobFile],
    media_failures: &HashMap<usize, String>,
) -> HashSet<usize> {
    let now = unix_now();
    let mut failed_indices = HashSet::with_capacity(media_failures.len());

    for (&idx, err_msg) in media_failures {
        failed_indices.insert(idx);
        if let Some(file) = file_list.iter().find(|file| file.file_index == idx) {
            FileRunTracker::new(sink, job_id, &file.filename)
                .record_setup_failure(now, err_msg, FailureCategory::Validation, now)
                .await;
        }
    }

    failed_indices
}

#[cfg(test)]
mod tests;
