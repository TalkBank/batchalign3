//! Server control-plane backend seam.
//!
//! This boundary keeps route handlers and other app-facing code from depending
//! directly on a specific orchestration backend. The [`ServerBackend`] trait
//! is the single interface that route handlers and lifecycle code consume.
//!
//! Two implementations exist:
//! - [`TemporalServerBackend`](crate::temporal_backend::TemporalServerBackend) — production
//!   backend backed by Temporal workflows and activities.
//! - [`TestServerBackend`] — lightweight in-process backend for integration tests
//!   that spawns inline runner tasks without external dependencies.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::api::{
    JobControlPlaneBackendKind, JobControlPlaneInfo, JobId, JobInfo, JobListItem, JobStatus,
    NodeId, NumWorkers, UnixTimestamp,
};
use crate::db::JobDB;
use crate::error::ServerError;
use crate::host_memory::HostMemoryError;
use crate::runner::util::RunnerEventSink;
use crate::runner::{
    ExecutionEngine, MemoryGateRejectionDisposition, QueuedJobOrchestrator, ServerExecutionHost,
    job_task,
};
use crate::runtime_supervisor::{RuntimeSupervisor, ShutdownError, ShutdownSummary};
use crate::scheduling::{DurationMs, RetryPolicy};
use crate::store::{Job, JobDetail, JobStore, unix_now};
use crate::types::traces::JobTraces;
use crate::ws::{BROADCAST_CAPACITY, WsEvent};

/// Store-backed health and queue-state snapshot for the server control plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerControlPlaneSnapshot {
    /// Node identifier used for queue-lease ownership.
    pub node_id: NodeId,
    /// Number of workers currently available to accept work.
    pub workers_available: i64,
    /// Number of jobs currently considered active by the control plane.
    pub active_jobs: i64,
    /// Number of unexpected worker crashes seen by this server.
    pub worker_crashes: i64,
    /// Number of work-unit attempts started.
    pub attempts_started: i64,
    /// Number of work-unit attempts retried.
    pub attempts_retried: i64,
    /// Number of deferred work units waiting for later eligibility.
    pub deferred_work_units: i64,
    /// Number of files forced into a terminal error state by the runner.
    pub forced_terminal_errors: i64,
    /// Number of jobs aborted by the host-memory gate.
    pub memory_gate_aborts: i64,
}

/// App-facing backend for queued-job orchestration and persisted job state.
#[async_trait]
pub trait ServerBackend: Send + Sync {
    /// Persist a newly submitted job and wake the dispatcher if needed.
    async fn submit_job(&self, job: Job) -> Result<(), ServerError>;

    /// Return the current list view for all known jobs.
    async fn list_jobs(&self) -> Vec<JobListItem>;

    /// Return one current job snapshot if it exists.
    async fn get_job(&self, job_id: &JobId) -> Option<JobInfo>;

    /// Return the detail projection used by results/traces routes.
    async fn get_job_detail(&self, job_id: &JobId) -> Option<JobDetail>;

    /// Return the current lifecycle status of one job.
    async fn job_status(&self, job_id: &JobId) -> Option<JobStatus>;

    /// Return whether the given job is still running.
    async fn is_job_running(&self, job_id: &JobId) -> Option<bool>;

    /// Request cancellation of one queued or running job.
    async fn cancel_job(&self, job_id: &JobId) -> Result<(), ServerError>;

    /// Permanently delete one terminal job.
    async fn delete_job(&self, job_id: &JobId) -> Result<(), ServerError>;

    /// Re-queue a restartable job and wake the dispatcher.
    async fn restart_job(&self, job_id: &JobId) -> Result<JobInfo, ServerError>;

    /// Cancel all jobs owned by this server instance.
    async fn cancel_all(&self) -> usize;

    /// Return the store-backed control-plane health snapshot.
    async fn control_plane_snapshot(&self) -> ServerControlPlaneSnapshot;

    /// Return algorithm traces for one job if they were collected.
    async fn get_job_traces(&self, job_id: &JobId) -> Option<Arc<JobTraces>>;

    /// Subscribe to control-plane broadcast events.
    fn subscribe_events(&self) -> broadcast::Receiver<WsEvent>;

    /// Stop background control-plane tasks tracked by the runtime supervisor.
    async fn shutdown_runtime(&self, timeout: Duration) -> Result<ShutdownSummary, ShutdownError>;
}

/// Summary returned when bootstrapping one concrete server backend.
///
/// This keeps the server factory from needing direct access to backend-specific
/// queue/runtime/broadcast internals just to report basic startup information.
pub(crate) struct ServerBackendBootstrap {
    /// Route-facing backend over the fully started control plane.
    pub backend: Arc<dyn ServerBackend>,
    /// Number of persisted jobs loaded from the database at startup.
    pub loaded_jobs: usize,
    /// Number of queued jobs eligible for dispatch resume.
    pub queued_jobs: usize,
}

// ---------------------------------------------------------------------------
// TestServerBackend — lightweight in-process backend for integration tests
// ---------------------------------------------------------------------------

/// Queued-job orchestrator for the test backend.
///
/// Handles memory gate rejections by re-queuing with a backoff policy,
/// mirroring the behavior of the production Temporal orchestrator.
struct TestJobOrchestrator {
    memory_gate_retry_policy: RetryPolicy,
}

impl TestJobOrchestrator {
    fn new() -> Self {
        Self {
            memory_gate_retry_policy: RetryPolicy {
                max_attempts: 1,
                initial_backoff_ms: DurationMs(30_000),
                max_backoff_ms: DurationMs(120_000),
                backoff_multiplier: 2,
            },
        }
    }
}

#[async_trait]
impl QueuedJobOrchestrator for TestJobOrchestrator {
    async fn handle_memory_gate_rejection(
        &self,
        sink: &Arc<dyn RunnerEventSink>,
        job_id: &JobId,
        _requested_workers: NumWorkers,
        _error: &HostMemoryError,
    ) -> Result<MemoryGateRejectionDisposition, ServerError> {
        let retry_at = UnixTimestamp(
            unix_now().0 + (self.memory_gate_retry_policy.backoff_for_retry(1).0 as f64 / 1000.0),
        );
        sink.requeue_job_after_memory_gate(job_id, retry_at).await;
        sink.bump_deferred_work_units().await;
        Ok(MemoryGateRejectionDisposition::Requeued { retry_at })
    }
}

/// Lightweight in-process server backend for integration tests.
///
/// Spawns inline runner tasks via `tokio::spawn` without requiring any
/// external dependencies (no Temporal, no queue dispatcher). Each submitted
/// job is persisted to the store and immediately dispatched.
pub(crate) struct TestServerBackend {
    store: Arc<JobStore>,
    host: ServerExecutionHost,
    runtime: RuntimeSupervisor,
    ws_tx: broadcast::Sender<WsEvent>,
}

fn test_control_plane_info() -> JobControlPlaneInfo {
    JobControlPlaneInfo {
        backend: JobControlPlaneBackendKind::Test,
        temporal: None,
    }
}

#[async_trait]
impl ServerBackend for TestServerBackend {
    async fn submit_job(&self, job: Job) -> Result<(), ServerError> {
        let job_id = job.identity.job_id.clone();
        self.store.submit(job).await?;
        let host = self.host.clone();
        self.runtime.spawn_job(job_task(job_id, host));
        Ok(())
    }

    async fn list_jobs(&self) -> Vec<JobListItem> {
        self.store
            .list_all()
            .await
            .into_iter()
            .map(|job| job.with_control_plane(test_control_plane_info()))
            .collect()
    }

    async fn get_job(&self, job_id: &JobId) -> Option<JobInfo> {
        self.store
            .get(job_id)
            .await
            .map(|job| job.with_control_plane(test_control_plane_info()))
    }

    async fn get_job_detail(&self, job_id: &JobId) -> Option<JobDetail> {
        self.store.get_job_detail(job_id).await
    }

    async fn job_status(&self, job_id: &JobId) -> Option<JobStatus> {
        self.store.job_status(job_id).await
    }

    async fn is_job_running(&self, job_id: &JobId) -> Option<bool> {
        self.store.is_running(job_id).await
    }

    async fn cancel_job(&self, job_id: &JobId) -> Result<(), ServerError> {
        self.store.cancel(job_id).await
    }

    async fn delete_job(&self, job_id: &JobId) -> Result<(), ServerError> {
        self.store.delete(job_id).await
    }

    async fn restart_job(&self, job_id: &JobId) -> Result<JobInfo, ServerError> {
        let info = self.store.restart(job_id).await?;
        let host = self.host.clone();
        let restart_job_id = job_id.clone();
        self.runtime.spawn_job(job_task(restart_job_id, host));
        Ok(info.with_control_plane(test_control_plane_info()))
    }

    async fn cancel_all(&self) -> usize {
        self.store.cancel_all().await
    }

    async fn control_plane_snapshot(&self) -> ServerControlPlaneSnapshot {
        store_backed_control_plane_snapshot(self.store.as_ref()).await
    }

    async fn get_job_traces(&self, job_id: &JobId) -> Option<Arc<JobTraces>> {
        self.store.trace_store().get(job_id).await
    }

    fn subscribe_events(&self) -> broadcast::Receiver<WsEvent> {
        self.ws_tx.subscribe()
    }

    async fn shutdown_runtime(&self, timeout: Duration) -> Result<ShutdownSummary, ShutdownError> {
        self.runtime.shutdown(timeout).await
    }
}

/// Build and start a lightweight in-process server control plane for tests.
///
/// Unlike the Temporal backend, this spawns inline runner tasks directly via
/// `tokio::spawn`. No external services are required.
pub(crate) async fn bootstrap_test_server_backend(
    config: crate::config::ServerConfig,
    db: Arc<JobDB>,
    engine: ExecutionEngine,
) -> Result<ServerBackendBootstrap, ServerError> {
    let (ws_tx, _) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::new(config, Some(db), ws_tx.clone()));
    let loaded_jobs = store.load_from_db().await?;
    let queued_jobs = store.queued_job_ids().await.len();

    let orchestrator = Arc::new(TestJobOrchestrator::new());
    let runtime = RuntimeSupervisor::new();
    let host = ServerExecutionHost::new(store.clone(), engine, orchestrator);

    let backend: Arc<dyn ServerBackend> = Arc::new(TestServerBackend {
        store,
        host,
        runtime,
        ws_tx,
    });

    Ok(ServerBackendBootstrap {
        backend,
        loaded_jobs,
        queued_jobs,
    })
}

/// Store-backed control-plane snapshot shared by all non-embedded backends.
pub(crate) async fn store_backed_control_plane_snapshot(
    store: &JobStore,
) -> ServerControlPlaneSnapshot {
    let (
        worker_crashes,
        attempts_started,
        attempts_retried,
        deferred_work_units,
        forced_terminal_errors,
        memory_gate_aborts,
    ) = store.operational_counters().await;
    let workers_available = store.workers_available().await;
    let active_jobs = store.active_jobs().await;
    ServerControlPlaneSnapshot {
        node_id: store.node_id().clone(),
        workers_available,
        active_jobs,
        worker_crashes,
        attempts_started,
        attempts_retried,
        deferred_work_units,
        forced_terminal_errors,
        memory_gate_aborts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use crate::api::{
        CorrelationId, DisplayPath, LanguageCode3, LanguageSpec, NumSpeakers, ReleasedCommand,
    };
    use crate::cache::UtteranceCache;
    use crate::db::JobDB;
    use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
    use crate::runner::{ExecutionEngine, RunnerExecutionContext};
    use crate::store::{
        FileStatus, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext, unix_now,
    };
    use crate::worker::pool::{PoolConfig, WorkerPool};
    use crate::ws::BROADCAST_CAPACITY;

    fn sample_job(job_id: &str) -> Job {
        let filename = "sample.cha";
        let mut file_statuses = HashMap::new();
        file_statuses.insert(
            filename.to_string(),
            FileStatus::new(DisplayPath::from(filename)),
        );

        Job {
            identity: JobIdentity {
                job_id: JobId::from(job_id),
                correlation_id: CorrelationId::from(format!("test-{job_id}")),
            },
            dispatch: JobDispatchConfig {
                command: ReleasedCommand::Morphotag,
                lang: LanguageSpec::Resolved(LanguageCode3::eng()),
                num_speakers: NumSpeakers(1),
                options: CommandOptions::Morphotag(MorphotagOptions {
                    common: CommonOptions::default(),
                    retokenize: false,
                    skipmultilang: false,
                    merge_abbrev: false.into(),
                }),
                runtime_state: Default::default(),
                debug_traces: false,
            },
            source: JobSourceContext {
                submitted_by: "127.0.0.1".into(),
                submitted_by_name: String::new(),
                source_dir: Default::default(),
            },
            filesystem: JobFilesystemConfig {
                filenames: vec![DisplayPath::from(filename)],
                has_chat: vec![true],
                staging_dir: Default::default(),
                paths_mode: false,
                source_paths: Vec::new(),
                output_paths: Vec::new(),
                before_paths: Vec::new(),
                media_mapping: Default::default(),
                media_subdir: Default::default(),
                source_dir: Default::default(),
            },
            execution: JobExecutionState {
                status: JobStatus::Queued,
                file_statuses,
                results: Vec::new(),
                error: None,
                completed_files: 0,
                batch_progress: None,
            },
            schedule: JobScheduleState {
                submitted_at: unix_now(),
                completed_at: None,
                next_eligible_at: None,
                num_workers: None,
                lease: JobLeaseState {
                    leased_by_node: None,
                    expires_at: None,
                    heartbeat_at: None,
                },
            },
            runtime: JobRuntimeControl {
                cancel_token: CancellationToken::new(),
                runner_active: false,
            },
        }
    }

    #[tokio::test]
    async fn test_backend_submit_dispatches_job() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let layout = crate::config::RuntimeLayout::from_state_dir(tempdir.path().join("state"));
        std::fs::create_dir_all(layout.state_dir()).expect("create state dir");

        let db = Arc::new(
            crate::db::JobDB::open_with_layout(&layout, Some(layout.state_dir()))
                .await
                .expect("open job db"),
        );
        let pool = Arc::new(WorkerPool::new(PoolConfig {
            test_echo: true,
            ..Default::default()
        }));
        pool.start_background_tasks();
        let cache = Arc::new(
            UtteranceCache::sqlite(Some(tempdir.path().join("cache")))
                .await
                .expect("open cache"),
        );
        let engine = ExecutionEngine::new(RunnerExecutionContext::new(
            pool,
            cache,
            Vec::new(),
            BTreeMap::new(),
            true,
        ));

        let bootstrap =
            bootstrap_test_server_backend(crate::config::ServerConfig::default(), db, engine)
                .await
                .expect("bootstrap test backend");

        bootstrap
            .backend
            .submit_job(sample_job("job-test-dispatch"))
            .await
            .expect("test backend should submit the job");

        // Verify the job exists in the store.
        assert!(
            bootstrap
                .backend
                .get_job(&JobId::from("job-test-dispatch"))
                .await
                .is_some(),
            "test backend should expose submitted job state"
        );

        bootstrap
            .backend
            .shutdown_runtime(Duration::from_secs(5))
            .await
            .expect("shutdown runtime");
    }
}
