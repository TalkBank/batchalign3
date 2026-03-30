//! Temporal-backed server control plane.
//!
//! This is an alternate server backend that keeps the existing Batchalign
//! execution engine and worker pool, but hands queued-job orchestration to
//! Temporal workflows and activities instead of the embedded local queue.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use temporalio_client::errors::{WorkflowInteractionError, WorkflowStartError};
use temporalio_client::{
    Client, ClientOptions, Connection, ConnectionOptions, WorkflowCancelOptions,
    WorkflowDescribeOptions, WorkflowStartOptions, WorkflowTerminateOptions,
};
use temporalio_common::protos::temporal::api::enums::v1::{
    WorkflowExecutionStatus, WorkflowIdConflictPolicy, WorkflowIdReusePolicy,
};
use temporalio_macros::{activities, workflow, workflow_methods};
use temporalio_sdk::activities::{ActivityContext, ActivityError};
use temporalio_sdk::{
    ActivityOptions, Worker, WorkerOptions, WorkflowContext, WorkflowContextView, WorkflowResult,
};
use temporalio_sdk_core::{CoreRuntime, RuntimeOptions, Url};
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::api::{
    JobControlPlaneInfo, JobId, JobInfo, JobListItem, JobStatus, NumWorkers,
    TemporalWorkflowExecutionInfo, UnixTimestamp,
};
use crate::config::ServerConfig;
use crate::db::JobDB;
use crate::error::ServerError;
use crate::host_memory::HostMemoryError;
use crate::runner::util::RunnerEventSink;
use crate::runner::{
    ExecutionEngine, HostedJobRunOutcome, MemoryGateRejectionDisposition, QueuedJobOrchestrator,
    ServerExecutionHost, run_server_job_attempt,
};
use crate::runtime_supervisor::{ShutdownError, ShutdownSummary};
use crate::scheduling::{DurationMs, RetryPolicy};
use crate::server_backend::{
    ServerBackend, ServerBackendBootstrap, ServerControlPlaneSnapshot,
    store_backed_control_plane_snapshot,
};
use crate::store::{Job, JobDetail, JobStore, unix_now};
use crate::types::traces::JobTraces;
use crate::ws::{BROADCAST_CAPACITY, WsEvent};

/// Temporal workflow input for one Batchalign job.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TemporalJobWorkflowInput {
    job_id: String,
    activity_timeout_s: u64,
    heartbeat_timeout_s: u64,
}

/// Temporal activity input for one Batchalign job attempt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TemporalJobActivityInput {
    job_id: String,
}

/// Serializable outcome returned from the Temporal activity back into the workflow.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum TemporalJobActivityOutcome {
    /// The shared Batchalign engine finished the job attempt successfully.
    Completed,
    /// The shared engine asked the workflow to sleep before trying again.
    Requeued { retry_after_ms: u64 },
}

#[workflow]
struct BatchalignJobWorkflow {
    job_id: String,
    activity_timeout_s: u64,
    heartbeat_timeout_s: u64,
}

#[workflow_methods]
impl BatchalignJobWorkflow {
    #[init]
    fn new(_ctx: &WorkflowContextView, input: TemporalJobWorkflowInput) -> Self {
        Self {
            job_id: input.job_id,
            activity_timeout_s: input.activity_timeout_s,
            heartbeat_timeout_s: input.heartbeat_timeout_s,
        }
    }

    #[run]
    async fn run(ctx: &mut WorkflowContext<Self>) -> WorkflowResult<()> {
        loop {
            let input = TemporalJobActivityInput {
                job_id: ctx.state(|state| state.job_id.clone()),
            };
            let activity_timeout_s = ctx.state(|state| state.activity_timeout_s.max(1));
            let heartbeat_timeout_s = ctx.state(|state| state.heartbeat_timeout_s.max(1));
            let outcome = ctx
                .start_activity(
                    BatchalignTemporalActivities::run_job_attempt,
                    input,
                    ActivityOptions {
                        start_to_close_timeout: Some(Duration::from_secs(activity_timeout_s)),
                        heartbeat_timeout: Some(Duration::from_secs(heartbeat_timeout_s)),
                        ..Default::default()
                    },
                )
                .await?;
            match outcome {
                TemporalJobActivityOutcome::Completed => return Ok(()),
                TemporalJobActivityOutcome::Requeued { retry_after_ms } => {
                    ctx.timer(Duration::from_millis(retry_after_ms.max(1)))
                        .await;
                }
            }
        }
    }
}

/// Activity bridge from Temporal into the shared Batchalign execution engine.
#[derive(Clone)]
pub struct BatchalignTemporalActivities {
    store: Arc<JobStore>,
    host: ServerExecutionHost,
}

#[activities]
impl BatchalignTemporalActivities {
    /// Execute one job attempt as a Temporal activity.
    ///
    /// Bridges from Temporal into the shared Batchalign execution engine.
    /// Heartbeats periodically so Temporal can detect stuck activities.
    /// Forwards cancel signals from Temporal to the job store.
    #[activity]
    pub async fn run_job_attempt(
        self: Arc<Self>,
        ctx: ActivityContext,
        input: TemporalJobActivityInput,
    ) -> Result<TemporalJobActivityOutcome, ActivityError> {
        let job_id = JobId::from(input.job_id);
        let run_job_id = job_id.clone();
        let host = self.host.clone();
        let store = self.store.clone();
        let run_task =
            tokio::spawn(async move { run_server_job_attempt(&run_job_id, &host).await });

        let heartbeat_every = ctx
            .info()
            .heartbeat_timeout
            .map(|timeout| std::cmp::max(Duration::from_secs(1), timeout / 2))
            .unwrap_or_else(|| Duration::from_secs(5));
        let mut heartbeat = tokio::time::interval(heartbeat_every);
        let mut forwarded_cancel = false;
        tokio::pin!(run_task);

        loop {
            tokio::select! {
                _ = heartbeat.tick() => {
                    ctx.record_heartbeat(Vec::new());
                    if ctx.is_cancelled() && !forwarded_cancel {
                        forwarded_cancel = true;
                        let _ = store.cancel(&job_id).await;
                    }
                }
                result = &mut run_task => {
                    let outcome = result.map_err(|error| {
                        ActivityError::NonRetryable(Box::new(ServerError::Persistence(
                            format!("Temporal job activity join failed for {job_id}: {error}"),
                        )))
                    })?;
                    return match outcome {
                        Ok(HostedJobRunOutcome::Completed) => Ok(TemporalJobActivityOutcome::Completed),
                        Ok(HostedJobRunOutcome::Requeued { retry_at }) => Ok(
                            TemporalJobActivityOutcome::Requeued {
                                retry_after_ms: retry_after_ms_from_retry_at(retry_at),
                            },
                        ),
                        Err(error) => Err(ActivityError::NonRetryable(Box::new(error))),
                    };
                }
            }
        }
    }
}

/// Temporal-specific queued-job orchestrator used by the shared runner.
struct TemporalJobOrchestrator {
    memory_gate_retry_policy: RetryPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemporalWorkflowStartMode {
    ResumeOrUseExisting,
    ReplaceExisting,
}

impl TemporalJobOrchestrator {
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
impl QueuedJobOrchestrator for TemporalJobOrchestrator {
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

/// Real Temporal-backed server backend over one JobStore projection.
pub struct TemporalServerBackend {
    store: Arc<JobStore>,
    client: Client,
    worker_runtime: TemporalWorkerRuntime,
    ws_tx: broadcast::Sender<WsEvent>,
    config: ServerConfig,
}

impl TemporalServerBackend {
    fn new(
        store: Arc<JobStore>,
        client: Client,
        worker_runtime: TemporalWorkerRuntime,
        ws_tx: broadcast::Sender<WsEvent>,
        config: ServerConfig,
    ) -> Self {
        Self {
            store,
            client,
            worker_runtime,
            ws_tx,
            config,
        }
    }

    async fn ensure_workflow_for_job(
        &self,
        job_id: &JobId,
        start_mode: TemporalWorkflowStartMode,
    ) -> Result<(), ServerError> {
        let input = TemporalJobWorkflowInput {
            job_id: job_id.to_string(),
            activity_timeout_s: self.config.temporal_activity_timeout_s,
            heartbeat_timeout_s: self.config.temporal_heartbeat_s,
        };
        let options = match start_mode {
            TemporalWorkflowStartMode::ResumeOrUseExisting => WorkflowStartOptions::new(
                self.config.temporal_task_queue.clone(),
                job_id.to_string(),
            )
            .id_reuse_policy(WorkflowIdReusePolicy::AllowDuplicate)
            .id_conflict_policy(WorkflowIdConflictPolicy::UseExisting)
            .build(),
            TemporalWorkflowStartMode::ReplaceExisting => WorkflowStartOptions::new(
                self.config.temporal_task_queue.clone(),
                job_id.to_string(),
            )
            .id_reuse_policy(WorkflowIdReusePolicy::AllowDuplicate)
            .id_conflict_policy(WorkflowIdConflictPolicy::TerminateExisting)
            .build(),
        };
        match self
            .client
            .start_workflow(BatchalignJobWorkflow::run, input, options)
            .await
        {
            Ok(_) => Ok(()),
            Err(WorkflowStartError::AlreadyStarted { .. }) => Ok(()),
            Err(error) => Err(ServerError::Persistence(format!(
                "failed to start Temporal workflow for job {job_id}: {error}"
            ))),
        }
    }

    async fn bootstrap_active_workflows(&self) -> Result<(), ServerError> {
        for job in self
            .store
            .list_all()
            .await
            .into_iter()
            .filter(|job| job.status.is_active())
        {
            self.ensure_workflow_for_job(
                &job.job_id,
                TemporalWorkflowStartMode::ResumeOrUseExisting,
            )
            .await?;
        }
        Ok(())
    }

    async fn cancel_temporal_workflow(&self, job_id: &JobId) -> Result<(), ServerError> {
        let handle = self
            .client
            .get_workflow_handle::<BatchalignJobWorkflow>(job_id.to_string());
        match handle
            .cancel(
                WorkflowCancelOptions::builder()
                    .reason(format!("batchalign job {job_id} cancelled"))
                    .build(),
            )
            .await
        {
            Ok(()) => Ok(()),
            Err(WorkflowInteractionError::NotFound(_)) => Ok(()),
            Err(error) => Err(ServerError::Persistence(format!(
                "failed to cancel Temporal workflow for job {job_id}: {error}"
            ))),
        }
    }

    async fn terminate_temporal_workflow(
        &self,
        job_id: &JobId,
        reason: &str,
    ) -> Result<(), ServerError> {
        let handle = self
            .client
            .get_workflow_handle::<BatchalignJobWorkflow>(job_id.to_string());
        match handle
            .terminate(
                WorkflowTerminateOptions::builder()
                    .reason(reason.to_string())
                    .build(),
            )
            .await
        {
            Ok(()) => Ok(()),
            Err(WorkflowInteractionError::NotFound(_)) => Ok(()),
            Err(error) => Err(ServerError::Persistence(format!(
                "failed to terminate Temporal workflow for job {job_id}: {error}"
            ))),
        }
    }

    async fn describe_temporal_workflow(
        &self,
        job_id: &JobId,
    ) -> Result<temporalio_client::WorkflowExecutionDescription, WorkflowInteractionError> {
        self.client
            .get_workflow_handle::<BatchalignJobWorkflow>(job_id.to_string())
            .describe(WorkflowDescribeOptions::default())
            .await
    }

    async fn temporal_workflow_execution_info(
        &self,
        job_id: &JobId,
    ) -> TemporalWorkflowExecutionInfo {
        match self.describe_temporal_workflow(job_id).await {
            Ok(description) => {
                let info = description.raw_description.workflow_execution_info;
                let workflow_id = info
                    .as_ref()
                    .and_then(|info| info.execution.as_ref())
                    .and_then(|execution| non_empty_string(&execution.workflow_id))
                    .unwrap_or_else(|| job_id.to_string());
                TemporalWorkflowExecutionInfo {
                    workflow_id,
                    run_id: info
                        .as_ref()
                        .and_then(|info| info.execution.as_ref())
                        .and_then(|execution| non_empty_string(&execution.run_id)),
                    status: info
                        .as_ref()
                        .map(|info| temporal_workflow_status_name(info.status()).to_string()),
                    task_queue: info
                        .as_ref()
                        .and_then(|info| non_empty_string(&info.task_queue)),
                    history_length: info.as_ref().map(|info| info.history_length),
                    describe_error: None,
                }
            }
            Err(WorkflowInteractionError::NotFound(_)) => TemporalWorkflowExecutionInfo {
                workflow_id: job_id.to_string(),
                run_id: None,
                status: Some("not-found".into()),
                task_queue: None,
                history_length: None,
                describe_error: Some("Workflow not found".into()),
            },
            Err(error) => TemporalWorkflowExecutionInfo {
                workflow_id: job_id.to_string(),
                run_id: None,
                status: None,
                task_queue: None,
                history_length: None,
                describe_error: Some(error.to_string()),
            },
        }
    }

    async fn enrich_job_info(&self, job: JobInfo) -> JobInfo {
        let job_id = job.job_id.clone();
        job.with_control_plane(JobControlPlaneInfo::temporal_with_execution(
            self.temporal_workflow_execution_info(&job_id).await,
        ))
    }
}

/// Owned host thread for the in-process Temporal worker.
struct TemporalWorkerRuntime {
    shutdown_tx: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    join_handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl TemporalWorkerRuntime {
    fn start(
        config: &ServerConfig,
        client: Client,
        activities: BatchalignTemporalActivities,
    ) -> Result<Self, ServerError> {
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let temporal_config = config.clone();
        let join_handle = std::thread::Builder::new()
            .name("batchalign3-temporal-worker".into())
            .spawn(move || {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("Temporal worker thread should build tokio runtime");
                let local = tokio::task::LocalSet::new();
                runtime.block_on(local.run_until(async move {
                    let core_runtime = CoreRuntime::new_assume_tokio(
                        RuntimeOptions::builder()
                            .heartbeat_interval(Some(Duration::from_secs(
                                temporal_config.temporal_heartbeat_s,
                            )))
                            .build()
                            .expect("Temporal worker runtime options should validate"),
                    )
                    .expect("Temporal worker core runtime should initialize");
                    let worker_options =
                        WorkerOptions::new(temporal_config.temporal_task_queue.clone())
                            .register_activities(activities)
                            .register_workflow::<BatchalignJobWorkflow>()
                            .build();
                    let mut worker = Worker::new(&core_runtime, client, worker_options)
                        .expect("Temporal worker should initialize");
                    let shutdown = worker.shutdown_handle();
                    tokio::spawn(async move {
                        let _ = shutdown_rx.await;
                        shutdown();
                    });
                    if let Err(error) = worker.run().await {
                        warn!(error = %error, "Temporal worker stopped with error");
                    }
                }));
            })
            .map_err(|error| {
                ServerError::Persistence(format!("failed to spawn Temporal worker thread: {error}"))
            })?;
        Ok(Self {
            shutdown_tx: Mutex::new(Some(shutdown_tx)),
            join_handle: Mutex::new(Some(join_handle)),
        })
    }

    async fn shutdown(&self, timeout: Duration) -> Result<ShutdownSummary, ShutdownError> {
        if let Some(sender) = self
            .shutdown_tx
            .lock()
            .expect("Temporal worker shutdown lock poisoned")
            .take()
        {
            let _ = sender.send(());
        }
        let Some(join_handle) = self
            .join_handle
            .lock()
            .expect("Temporal worker join lock poisoned")
            .take()
        else {
            return Ok(ShutdownSummary {
                timed_out: false,
                remaining_jobs: 0,
            });
        };

        match tokio::time::timeout(
            timeout,
            tokio::task::spawn_blocking(move || {
                let _ = join_handle.join();
            }),
        )
        .await
        {
            Ok(join_result) => {
                let _ = join_result;
                Ok(ShutdownSummary {
                    timed_out: false,
                    remaining_jobs: 0,
                })
            }
            Err(_) => Ok(ShutdownSummary {
                timed_out: true,
                remaining_jobs: 1,
            }),
        }
    }
}

#[async_trait]
impl ServerBackend for TemporalServerBackend {
    async fn submit_job(&self, job: Job) -> Result<(), ServerError> {
        let job_id = job.identity.job_id.clone();
        self.store.submit(job).await?;
        if let Err(error) = self
            .ensure_workflow_for_job(&job_id, TemporalWorkflowStartMode::ResumeOrUseExisting)
            .await
        {
            self.store
                .fail_job(&job_id, &error.to_string(), unix_now())
                .await;
            return Err(error);
        }
        Ok(())
    }

    async fn list_jobs(&self) -> Vec<JobListItem> {
        self.store
            .list_all()
            .await
            .into_iter()
            .map(|job| job.with_control_plane(JobControlPlaneInfo::temporal()))
            .collect()
    }

    async fn get_job(&self, job_id: &JobId) -> Option<JobInfo> {
        match self.store.get(job_id).await {
            Some(job) => Some(self.enrich_job_info(job).await),
            None => None,
        }
    }

    async fn get_job_detail(&self, job_id: &JobId) -> Option<JobDetail> {
        self.store.get_job_detail(job_id).await
    }

    async fn job_status(&self, job_id: &JobId) -> Option<JobStatus> {
        self.store.job_status(job_id).await
    }

    async fn is_job_running(&self, job_id: &JobId) -> Option<bool> {
        // Check the store first. If the store says the job is in a terminal
        // state (completed, cancelled, failed), return false immediately
        // without querying Temporal — the store is the source of truth for
        // job lifecycle.
        let store_status = self.store.job_status(job_id).await?;
        if store_status.is_terminal() {
            return Some(false);
        }
        if store_status == JobStatus::Running {
            return Some(true);
        }
        // Job is queued — check Temporal to see if a workflow is active.
        match self.describe_temporal_workflow(job_id).await {
            Ok(description) => Some(
                description
                    .raw_description
                    .workflow_execution_info
                    .as_ref()
                    .map(|info| temporal_workflow_status_is_active(info.status()))
                    .unwrap_or(false),
            ),
            Err(WorkflowInteractionError::NotFound(_)) => Some(false),
            Err(error) => {
                warn!(
                    job_id = %job_id,
                    error = %error,
                    "Temporal describe failed during running-state check; treating job as active"
                );
                Some(true)
            }
        }
    }

    async fn cancel_job(&self, job_id: &JobId) -> Result<(), ServerError> {
        self.store.cancel(job_id).await?;
        self.cancel_temporal_workflow(job_id).await
    }

    async fn delete_job(&self, job_id: &JobId) -> Result<(), ServerError> {
        self.terminate_temporal_workflow(job_id, &format!("batchalign job {job_id} deleted"))
            .await?;
        self.store.delete(job_id).await
    }

    async fn restart_job(&self, job_id: &JobId) -> Result<JobInfo, ServerError> {
        let info = self.store.restart(job_id).await?;
        if let Err(error) = self
            .ensure_workflow_for_job(job_id, TemporalWorkflowStartMode::ReplaceExisting)
            .await
        {
            self.store
                .fail_job(job_id, &error.to_string(), unix_now())
                .await;
            return Err(error);
        }
        Ok(self.enrich_job_info(info).await)
    }

    async fn cancel_all(&self) -> usize {
        let active_job_ids: Vec<JobId> = self
            .store
            .list_all()
            .await
            .into_iter()
            .filter(|job| job.status.can_cancel())
            .map(|job| job.job_id)
            .collect();
        let mut cancelled = 0usize;
        for job_id in active_job_ids {
            if self.cancel_job(&job_id).await.is_ok() {
                cancelled += 1;
            }
        }
        cancelled
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
        self.worker_runtime.shutdown(timeout).await
    }
}

/// Build and start the Temporal-backed server control plane.
pub(crate) async fn bootstrap_temporal_server_backend(
    config: ServerConfig,
    db: Arc<JobDB>,
    engine: ExecutionEngine,
) -> Result<ServerBackendBootstrap, ServerError> {
    let (ws_tx, _) = broadcast::channel(BROADCAST_CAPACITY);
    let store = Arc::new(JobStore::new(config.clone(), Some(db), ws_tx.clone()));
    let loaded_jobs = store.load_from_db().await?;
    let queued_jobs = store
        .list_all()
        .await
        .into_iter()
        .filter(|job| job.status == JobStatus::Queued)
        .count();

    let connection = Connection::connect(
        ConnectionOptions::new(Url::parse(&config.temporal_server_url).map_err(|error| {
            ServerError::Validation(format!(
                "invalid temporal_server_url '{}': {error}",
                config.temporal_server_url
            ))
        })?)
        .identity("batchalign3-server-temporal".to_string())
        .build(),
    )
    .await
    .map_err(|error| {
        ServerError::Persistence(format!(
            "failed to connect to Temporal server at '{}': {error}. \
         Ensure the Temporal server is running and reachable.",
            config.temporal_server_url
        ))
    })?;
    let client = Client::new(
        connection,
        ClientOptions::new(config.temporal_namespace.clone()).build(),
    )
    .map_err(|error| {
        ServerError::Persistence(format!("failed to build Temporal client: {error}"))
    })?;

    let orchestrator = Arc::new(TemporalJobOrchestrator::new());
    let server_host = ServerExecutionHost::new(store.clone(), engine, orchestrator);
    let activities = BatchalignTemporalActivities {
        store: store.clone(),
        host: server_host,
    };
    let worker_runtime = TemporalWorkerRuntime::start(&config, client.clone(), activities)?;

    let backend_impl = Arc::new(TemporalServerBackend::new(
        store,
        client,
        worker_runtime,
        ws_tx,
        config,
    ));
    backend_impl.bootstrap_active_workflows().await?;
    let backend: Arc<dyn ServerBackend> = backend_impl;

    info!(loaded_jobs, queued_jobs, "Temporal backend bootstrapped");

    Ok(ServerBackendBootstrap {
        backend,
        loaded_jobs,
        queued_jobs,
    })
}

fn retry_after_ms_from_retry_at(retry_at: UnixTimestamp) -> u64 {
    let now = unix_now().0;
    if retry_at.0 <= now {
        0
    } else {
        ((retry_at.0 - now) * 1000.0).ceil() as u64
    }
}

fn non_empty_string(value: &str) -> Option<String> {
    (!value.trim().is_empty()).then(|| value.to_string())
}

fn temporal_workflow_status_name(status: WorkflowExecutionStatus) -> &'static str {
    match status {
        WorkflowExecutionStatus::Running => "running",
        WorkflowExecutionStatus::Completed => "completed",
        WorkflowExecutionStatus::Failed => "failed",
        WorkflowExecutionStatus::Canceled => "cancelled",
        WorkflowExecutionStatus::Terminated => "terminated",
        WorkflowExecutionStatus::ContinuedAsNew => "continued-as-new",
        WorkflowExecutionStatus::TimedOut => "timed-out",
        WorkflowExecutionStatus::Paused => "paused",
        WorkflowExecutionStatus::Unspecified => "unspecified",
    }
}

fn temporal_workflow_status_is_active(status: WorkflowExecutionStatus) -> bool {
    matches!(
        status,
        WorkflowExecutionStatus::Running | WorkflowExecutionStatus::Paused
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_after_ms_clamps_past_deadlines() {
        assert_eq!(
            retry_after_ms_from_retry_at(UnixTimestamp(unix_now().0 - 10.0)),
            0
        );
    }

    #[test]
    fn temporal_status_name_normalizes_proto_values() {
        assert_eq!(
            temporal_workflow_status_name(WorkflowExecutionStatus::Canceled),
            "cancelled"
        );
        assert_eq!(
            temporal_workflow_status_name(WorkflowExecutionStatus::ContinuedAsNew),
            "continued-as-new"
        );
    }

    #[test]
    fn temporal_active_status_matches_running_and_paused_only() {
        assert!(temporal_workflow_status_is_active(
            WorkflowExecutionStatus::Running
        ));
        assert!(temporal_workflow_status_is_active(
            WorkflowExecutionStatus::Paused
        ));
        assert!(!temporal_workflow_status_is_active(
            WorkflowExecutionStatus::Completed
        ));
    }

    #[test]
    fn non_empty_string_rejects_whitespace() {
        assert_eq!(non_empty_string(""), None);
        assert_eq!(non_empty_string("   "), None);
        assert_eq!(non_empty_string("run-123"), Some("run-123".into()));
    }
}
