//! Server control-plane backend seam.
//!
//! This boundary keeps route handlers and other app-facing code from depending
//! directly on the embedded `JobStore + QueueBackend + RuntimeSupervisor`
//! bundle. The current implementation is still the in-process control plane,
//! but the rest of the app can now talk to that machinery through one named
//! backend instead of poking separate store/queue/runtime handles.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::api::{
    JobControlPlaneInfo, JobId, JobInfo, JobListItem, JobStatus, NodeId, NumWorkers, UnixTimestamp,
};
use crate::config::ServerConfig;
use crate::db::JobDB;
use crate::error::ServerError;
use crate::host_memory::HostMemoryError;
use crate::queue::{LocalQueueBackend, QueueBackend, QueueDispatcher};
use crate::runner::util::RunnerEventSink;
use crate::runner::{
    ExecutionEngine, MemoryGateRejectionDisposition, QueuedJobOrchestrator, ServerExecutionHost,
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

/// Embedded server orchestrator that owns the in-process control plane.
///
/// This is the stronger embedded seam under [`ServerBackend`]. It owns the
/// concrete `JobStore + QueueBackend + RuntimeSupervisor + WsEvent` bundle and
/// is the natural home for embedded-only queue/retry/recovery policy as that
/// logic is pulled away from startup glue and route handlers.
struct EmbeddedJobOrchestrator {
    store: Arc<JobStore>,
    queue: Arc<dyn QueueBackend>,
    runtime: RuntimeSupervisor,
    ws_tx: broadcast::Sender<WsEvent>,
    memory_gate_retry_policy: RetryPolicy,
}

impl EmbeddedJobOrchestrator {
    /// Create a new embedded orchestrator over the existing in-process control plane.
    pub fn new(
        store: Arc<JobStore>,
        queue: Arc<dyn QueueBackend>,
        runtime: RuntimeSupervisor,
        ws_tx: broadcast::Sender<WsEvent>,
    ) -> Self {
        Self {
            store,
            queue,
            runtime,
            ws_tx,
            memory_gate_retry_policy: embedded_memory_gate_retry_policy(),
        }
    }

    async fn submit_job(&self, job: Job) -> Result<(), ServerError> {
        self.store.submit(job).await?;
        self.queue.notify();
        Ok(())
    }

    async fn list_jobs(&self) -> Vec<JobListItem> {
        self.store.list_all().await
    }

    async fn get_job(&self, job_id: &JobId) -> Option<JobInfo> {
        self.store.get(job_id).await
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
        self.queue.notify();
        Ok(info)
    }

    async fn cancel_all(&self) -> usize {
        self.store.cancel_all().await
    }

    async fn control_plane_snapshot(&self) -> ServerControlPlaneSnapshot {
        let (
            worker_crashes,
            attempts_started,
            attempts_retried,
            deferred_work_units,
            forced_terminal_errors,
            memory_gate_aborts,
        ) = self.store.operational_counters().await;
        let workers_available = self.store.workers_available().await;
        let active_jobs = self.store.active_jobs().await;

        ServerControlPlaneSnapshot {
            node_id: self.store.node_id().clone(),
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

fn embedded_memory_gate_retry_policy() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 1,
        initial_backoff_ms: DurationMs(30_000),
        max_backoff_ms: DurationMs(120_000),
        backoff_multiplier: 2,
    }
}

#[async_trait]
impl QueuedJobOrchestrator for EmbeddedJobOrchestrator {
    async fn handle_memory_gate_rejection(
        &self,
        sink: &Arc<dyn RunnerEventSink>,
        job_id: &JobId,
        _requested_workers: NumWorkers,
        _error: &HostMemoryError,
    ) -> Result<MemoryGateRejectionDisposition, ServerError> {
        if self.memory_gate_retry_policy.max_attempts == 0 {
            return Ok(MemoryGateRejectionDisposition::FailJob);
        }
        let retry_at = UnixTimestamp(
            unix_now().0 + (self.memory_gate_retry_policy.backoff_for_retry(1).0 as f64 / 1000.0),
        );
        sink.requeue_job_after_memory_gate(job_id, retry_at).await;
        sink.bump_deferred_work_units().await;
        self.queue.notify();
        Ok(MemoryGateRejectionDisposition::Requeued { retry_at })
    }
}

/// Embedded server backend backed by one embedded orchestrator.
pub struct EmbeddedServerBackend {
    orchestrator: Arc<EmbeddedJobOrchestrator>,
}

impl EmbeddedServerBackend {
    /// Create a new embedded server backend over one in-process orchestrator.
    fn new(orchestrator: Arc<EmbeddedJobOrchestrator>) -> Self {
        Self { orchestrator }
    }
}

/// Summary returned when bootstrapping one concrete server backend.
///
/// This keeps the server factory from needing direct access to backend-specific
/// queue/runtime/broadcast internals just to report basic startup information.
pub(crate) struct ServerBackendBootstrap {
    /// Route-facing backend over the fully started embedded control plane.
    pub backend: Arc<dyn ServerBackend>,
    /// Number of persisted jobs loaded from the database at startup.
    pub loaded_jobs: usize,
    /// Number of queued jobs eligible for local dispatcher resume.
    pub queued_jobs: usize,
}

/// Build and start the in-process server control plane.
///
/// This is the embedded/default backend bootstrap path. It owns the concrete
/// `JobStore + LocalQueueBackend + RuntimeSupervisor + QueueDispatcher`
/// assembly and returns only the app-facing [`ServerBackend`] plus the startup
/// counters the server factory needs for logging.
pub(crate) async fn bootstrap_embedded_server_backend(
    config: ServerConfig,
    db: Arc<JobDB>,
    engine: ExecutionEngine,
) -> Result<ServerBackendBootstrap, ServerError> {
    let (ws_tx, _) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::new(config, Some(db), ws_tx.clone()));
    let loaded_jobs = store.load_from_db().await?;
    let queued_jobs = store.queued_job_ids().await.len();

    let queue_notify = Arc::new(tokio::sync::Notify::new());
    let queue: Arc<dyn QueueBackend> =
        Arc::new(LocalQueueBackend::new(store.clone(), queue_notify));
    let runtime = RuntimeSupervisor::new();
    let orchestrator = Arc::new(EmbeddedJobOrchestrator::new(store, queue, runtime, ws_tx));
    let server_host =
        ServerExecutionHost::new(orchestrator.store.clone(), engine, orchestrator.clone());
    let dispatcher = QueueDispatcher::new(
        orchestrator.queue.clone(),
        orchestrator.runtime.clone(),
        server_host,
    );
    orchestrator.runtime.start_queue_task(dispatcher.run());
    orchestrator.queue.notify();
    let backend: Arc<dyn ServerBackend> = Arc::new(EmbeddedServerBackend::new(orchestrator));

    Ok(ServerBackendBootstrap {
        backend,
        loaded_jobs,
        queued_jobs,
    })
}

#[async_trait]
impl ServerBackend for EmbeddedServerBackend {
    async fn submit_job(&self, job: Job) -> Result<(), ServerError> {
        self.orchestrator.submit_job(job).await
    }

    async fn list_jobs(&self) -> Vec<JobListItem> {
        self.orchestrator
            .list_jobs()
            .await
            .into_iter()
            .map(|job| job.with_control_plane(JobControlPlaneInfo::embedded()))
            .collect()
    }

    async fn get_job(&self, job_id: &JobId) -> Option<JobInfo> {
        self.orchestrator
            .get_job(job_id)
            .await
            .map(|job| job.with_control_plane(JobControlPlaneInfo::embedded()))
    }

    async fn get_job_detail(&self, job_id: &JobId) -> Option<JobDetail> {
        self.orchestrator.get_job_detail(job_id).await
    }

    async fn job_status(&self, job_id: &JobId) -> Option<JobStatus> {
        self.orchestrator.job_status(job_id).await
    }

    async fn is_job_running(&self, job_id: &JobId) -> Option<bool> {
        self.orchestrator.is_job_running(job_id).await
    }

    async fn cancel_job(&self, job_id: &JobId) -> Result<(), ServerError> {
        self.orchestrator.cancel_job(job_id).await
    }

    async fn delete_job(&self, job_id: &JobId) -> Result<(), ServerError> {
        self.orchestrator.delete_job(job_id).await
    }

    async fn restart_job(&self, job_id: &JobId) -> Result<JobInfo, ServerError> {
        self.orchestrator
            .restart_job(job_id)
            .await
            .map(|job| job.with_control_plane(JobControlPlaneInfo::embedded()))
    }

    async fn cancel_all(&self) -> usize {
        self.orchestrator.cancel_all().await
    }

    async fn control_plane_snapshot(&self) -> ServerControlPlaneSnapshot {
        self.orchestrator.control_plane_snapshot().await
    }

    async fn get_job_traces(&self, job_id: &JobId) -> Option<Arc<JobTraces>> {
        self.orchestrator.get_job_traces(job_id).await
    }

    fn subscribe_events(&self) -> broadcast::Receiver<WsEvent> {
        self.orchestrator.subscribe_events()
    }

    async fn shutdown_runtime(&self, timeout: Duration) -> Result<ShutdownSummary, ShutdownError> {
        self.orchestrator.shutdown_runtime(timeout).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::BTreeMap;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use tokio_util::sync::CancellationToken;

    use crate::api::{
        CorrelationId, DisplayPath, LanguageCode3, LanguageSpec, NumSpeakers, ReleasedCommand,
    };
    use crate::cache::UtteranceCache;
    use crate::db::JobDB;
    use crate::host_memory::HostMemoryError;
    use crate::options::{CommandOptions, CommonOptions, MorphotagOptions};
    use crate::runner::util::StoreRunnerEventSink;
    use crate::runner::{ExecutionEngine, RunnerExecutionContext};
    use crate::store::{
        FileStatus, JobDispatchConfig, JobExecutionState, JobFilesystemConfig, JobIdentity,
        JobLeaseState, JobRuntimeControl, JobScheduleState, JobSourceContext, unix_now,
    };
    use crate::worker::pool::{PoolConfig, WorkerPool};
    use crate::ws::BROADCAST_CAPACITY;

    #[derive(Default)]
    struct RecordingQueue {
        notified: AtomicUsize,
    }

    impl RecordingQueue {
        fn notify_count(&self) -> usize {
            self.notified.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl QueueBackend for RecordingQueue {
        fn notify(&self) {
            self.notified.fetch_add(1, Ordering::SeqCst);
        }

        async fn claim_ready_jobs(&self) -> crate::queue::QueuePoll {
            crate::queue::QueuePoll::default()
        }

        async fn wait_for_work(&self, _next_wake_at: Option<crate::api::UnixTimestamp>) {}
    }

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
    async fn embedded_backend_submit_notifies_queue() {
        let (ws_tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            ws_tx.clone(),
        ));
        let queue = Arc::new(RecordingQueue::default());
        let orchestrator = Arc::new(EmbeddedJobOrchestrator::new(
            store,
            queue.clone(),
            RuntimeSupervisor::new(),
            ws_tx,
        ));
        let backend = EmbeddedServerBackend::new(orchestrator);

        backend
            .submit_job(sample_job("job-submit-notify"))
            .await
            .expect("embedded backend should submit the job");

        assert_eq!(queue.notify_count(), 1);
    }

    #[tokio::test]
    async fn bootstrap_embedded_backend_hides_control_plane_assembly() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let layout = crate::config::RuntimeLayout::from_state_dir(tempdir.path().join("state"));
        std::fs::create_dir_all(layout.state_dir()).expect("create state dir");

        let db = Arc::new(
            JobDB::open_with_layout(&layout, Some(layout.state_dir()))
                .await
                .expect("open job db"),
        );
        let (ws_tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let seed_store = Arc::new(JobStore::new(
            crate::config::ServerConfig::default(),
            Some(db.clone()),
            ws_tx,
        ));
        seed_store
            .submit(sample_job("job-bootstrap-load"))
            .await
            .expect("seed job should persist");

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
            bootstrap_embedded_server_backend(crate::config::ServerConfig::default(), db, engine)
                .await
                .expect("bootstrap embedded backend");

        assert_eq!(bootstrap.loaded_jobs, 1);
        assert_eq!(bootstrap.queued_jobs, 1);
        assert!(
            bootstrap
                .backend
                .get_job(&JobId::from("job-bootstrap-load"))
                .await
                .is_some(),
            "bootstrapped backend should expose loaded job state"
        );

        bootstrap
            .backend
            .shutdown_runtime(Duration::from_secs(1))
            .await
            .expect("shutdown runtime");
    }

    #[tokio::test]
    async fn embedded_orchestrator_requeues_memory_gate_rejection() {
        let (ws_tx, _rx) = broadcast::channel(BROADCAST_CAPACITY);
        let store = Arc::new(JobStore::new(
            crate::config::ServerConfig::default(),
            None,
            ws_tx.clone(),
        ));
        store
            .submit(sample_job("job-memory-gate"))
            .await
            .expect("submit sample job");
        let queue = Arc::new(RecordingQueue::default());
        let orchestrator = EmbeddedJobOrchestrator::new(
            store.clone(),
            queue.clone(),
            RuntimeSupervisor::new(),
            ws_tx,
        );
        let sink = StoreRunnerEventSink::new(store.clone());

        let disposition = orchestrator
            .handle_memory_gate_rejection(
                &sink,
                &JobId::from("job-memory-gate"),
                crate::api::NumWorkers(2),
                &HostMemoryError::CapacityRejected {
                    label: "job-memory-gate".into(),
                    available_mb: 256,
                    pending_reserved_mb: 512,
                    requested_mb: 1024,
                    reserve_mb: 512,
                    total_mb: 4096,
                },
            )
            .await
            .expect("memory gate rejection should be handled");

        let retry_at = match disposition {
            MemoryGateRejectionDisposition::Requeued { retry_at } => retry_at,
            MemoryGateRejectionDisposition::FailJob => {
                panic!("embedded orchestrator should requeue")
            }
        };
        let info = store
            .get(&JobId::from("job-memory-gate"))
            .await
            .expect("job info");
        assert_eq!(info.status, JobStatus::Queued);
        assert_eq!(info.next_eligible_at, Some(retry_at));
        assert_eq!(queue.notify_count(), 1);

        let (_, _, _, deferred_work_units, _, _) = store.operational_counters().await;
        assert_eq!(deferred_work_units, 1);
    }
}
