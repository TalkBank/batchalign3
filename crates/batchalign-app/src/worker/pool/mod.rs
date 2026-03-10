//! `WorkerPool` — manages multiple Python worker processes.
//!
//! Workers are keyed by `(bootstrap target, lang, engine overrides)`. The
//! released target space is infer-task-only (for example `infer:asr` or
//! `infer:morphosyntax`), even when the Rust control plane is scheduling
//! higher-level commands such as `transcribe` or `compare`.
//!
//! Each key maps to a `WorkerGroup` containing up to `max_workers_per_key`
//! workers, spawned lazily on demand. Background tasks handle health checking
//! and idle timeouts.
//!
//! ## Concurrency model
//!
//! Workers are *owned values* in a `VecDeque`, not wrapped in `Arc<Mutex>`.
//! Availability is tracked by a `tokio::sync::Semaphore` (one permit per idle
//! worker). Callers *check out* a worker via `checkout()`, which acquires a
//! semaphore permit (async wait if all busy) then pops from the idle queue.
//! The returned `CheckedOutWorker` is an RAII guard that returns the worker
//! to the pool on drop.
//!
//! This eliminates the previous `Arc<tokio::sync::Mutex<WorkerHandle>>` pattern
//! where a tokio mutex was held for 10–300 seconds during dispatch.

mod checkout;
mod lifecycle;

pub use checkout::CheckedOutWorker;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crate::api::{CommandName, LanguageCode3, NumSpeakers};
use crate::types::worker_v2::{
    AsrBackendV2, ExecuteRequestV2, ExecuteResponseV2, FaBackendV2, InferenceTaskV2, TaskRequestV2,
};
use crate::worker::{
    BatchInferRequest, BatchInferResponse, InferTask, WorkerCapabilities, WorkerTarget,
};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle, WorkerRuntimeConfig};
use crate::worker::python::resolve_python_executable;

/// Key for looking up workers: (bootstrap target, lang, engine overrides).
type WorkerKey = (WorkerTarget, LanguageCode3, String);

/// Lifecycle state of background model warmup.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    utoipa::ToSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum WarmupStatus {
    /// No warmup has been requested yet (initial state).
    #[default]
    NotStarted,
    /// Warmup is running — workers are being spawned in the background.
    InProgress,
    /// All requested warmup spawns have finished (or none were requested).
    Complete,
}

impl WarmupStatus {
    const NOT_STARTED: u8 = 0;
    const IN_PROGRESS: u8 = 1;
    const COMPLETE: u8 = 2;

    fn from_u8(v: u8) -> Self {
        match v {
            Self::IN_PROGRESS => Self::InProgress,
            Self::COMPLETE => Self::Complete,
            _ => Self::NotStarted,
        }
    }

    fn as_u8(self) -> u8 {
        match self {
            Self::NotStarted => Self::NOT_STARTED,
            Self::InProgress => Self::IN_PROGRESS,
            Self::Complete => Self::COMPLETE,
        }
    }
}

/// Default maximum workers per `(target, lang, engine_overrides)` key.
const DEFAULT_MAX_WORKERS_PER_KEY: usize = 8;

/// Configuration for the worker pool.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Path to the Python executable.
    pub python_path: String,
    /// Seconds between health checks.
    pub health_check_interval_s: u64,
    /// Seconds of inactivity before a worker is shut down.
    pub idle_timeout_s: u64,
    /// Maximum seconds to wait for a worker to become ready.
    pub ready_timeout_s: u64,
    /// Use test-echo mode for all workers (no ML models).
    pub test_echo: bool,
    /// Maximum workers per `(target, lang)` key. Default: 8.
    /// The pool is the capacity ceiling; the runner controls per-job
    /// concurrency via a semaphore.
    pub max_workers_per_key: usize,
    /// Verbosity level forwarded to Python workers (0=warn, 1=info, 2=debug).
    pub verbose: u8,
    /// Engine overrides as a JSON string, passed to every spawned worker via
    /// `--engine-overrides`. Empty string means no overrides.
    pub engine_overrides: String,
    /// Runtime-owned worker launch inputs (device policy, injected creds).
    pub runtime: WorkerRuntimeConfig,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            python_path: resolve_python_executable(),
            health_check_interval_s: 30,
            idle_timeout_s: 600, // 10 minutes
            ready_timeout_s: 120,
            test_echo: false,
            max_workers_per_key: DEFAULT_MAX_WORKERS_PER_KEY,
            verbose: 0,
            engine_overrides: String::new(),
            runtime: WorkerRuntimeConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerGroup — per (target, lang) key
// ---------------------------------------------------------------------------

/// A group of workers for a single `(target, lang)` key.
///
/// Each group independently tracks its own pool of workers. Workers are
/// spawned lazily on first demand and capped at `max_workers_per_key`.
/// The group uses a split concurrency model: a semaphore for async
/// waiting and a mutex for the actual worker queue, so the mutex is
/// never held across an `.await` point.
struct WorkerGroup {
    /// Owned worker handles that are currently idle (not checked out).
    ///
    /// Protected by a `std::sync::Mutex` (not `tokio::sync::Mutex`)
    /// because it is held only for the duration of a `push_back` or
    /// `pop_front` -- microseconds, never across an `.await`. This avoids
    /// the overhead of a tokio-aware mutex and is safe because the
    /// critical section cannot yield.
    idle: std::sync::Mutex<VecDeque<WorkerHandle>>,

    /// Semaphore with one permit per idle worker.
    ///
    /// `checkout()` acquires a permit (blocking asynchronously if all
    /// workers are busy), then pops from `idle`. When a `CheckedOutWorker`
    /// is dropped, it pushes the worker back into `idle` and adds a
    /// permit, waking the next waiter. Permits are managed manually
    /// (`.forget()` after acquire, `.add_permits(1)` on return) rather
    /// than via RAII `SemaphorePermit` guards.
    available: Semaphore,

    /// Total number of live workers in this group: idle + checked-out.
    ///
    /// `AtomicUsize` so that `worker_count()` and spawn-cap checks can
    /// read it without acquiring any mutex. Incremented in
    /// `try_claim_spawn_slot()` (via `compare_exchange`) before the
    /// worker is spawned, and decremented when a worker is removed
    /// (idle timeout, health failure, or `CheckedOutWorker::take()`).
    total: AtomicUsize,

    /// Serialize worker bootstrap for one key.
    ///
    /// This prevents a burst of concurrent requests from launching multiple
    /// heavy Python workers for the same `(target, lang, engine_overrides)`
    /// bucket at once, which smooths model-loading spikes without changing the
    /// eventual steady-state concurrency of the pool.
    bootstrap: AsyncMutex<()>,
}

impl WorkerGroup {
    fn new() -> Self {
        Self {
            idle: std::sync::Mutex::new(VecDeque::new()),
            available: Semaphore::new(0),
            total: AtomicUsize::new(0),
            bootstrap: AsyncMutex::new(()),
        }
    }
}

/// Shared map of worker groups, accessible from both the pool and background tasks.
type GroupsMap = Arc<std::sync::Mutex<HashMap<WorkerKey, Arc<WorkerGroup>>>>;

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/// Manages a pool of Python worker processes.
pub struct WorkerPool {
    config: PoolConfig,
    groups: GroupsMap,
    cancel: CancellationToken,
    /// Background warmup lifecycle state.
    warmup_status: AtomicU8,
}

impl WorkerPool {
    /// Create a new worker pool. Call [`start_background_tasks`](Self::start_background_tasks)
    /// to begin health checking and idle timeout.
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            groups: Arc::new(std::sync::Mutex::new(HashMap::new())),
            cancel: CancellationToken::new(),
            warmup_status: AtomicU8::new(WarmupStatus::NotStarted.as_u8()),
        }
    }

    /// Check out an idle worker or spawn a new one.
    ///
    /// 1. Try to acquire a semaphore permit immediately.
    /// 2. If none available, try to spawn a new worker (if under capacity).
    /// 3. If at capacity, wait for a permit (async suspend).
    /// 4. Pop from the idle queue and wrap in `CheckedOutWorker` (RAII guard).
    async fn checkout(
        &self,
        target: &WorkerTarget,
        lang: &LanguageCode3,
        engine_overrides: &str,
    ) -> Result<CheckedOutWorker, WorkerError> {
        let group = self.get_or_create_group(target, lang, engine_overrides);

        loop {
            // Try to acquire a permit without waiting.
            match group.available.try_acquire() {
                Ok(permit) => {
                    permit.forget(); // We manage permits manually
                    match group.idle.lock().unwrap().pop_front() {
                        Some(handle) => {
                            return Ok(CheckedOutWorker {
                                handle: Some(handle),
                                group: group.clone(),
                            });
                        }
                        None => {
                            // The health-check task drains the idle queue before
                            // re-adding permits for survivors, so there is a brief
                            // window where a permit exists but the queue is empty.
                            // Return the permit and loop: the health check will
                            // re-add permits for healthy workers, or we fall
                            // through to the spawn/async-wait path below.
                            group.available.add_permits(1);
                            continue;
                        }
                    }
                }
                Err(_) => {
                    // No idle workers. Try to spawn one.
                    match self
                        .try_spawn_into_group(&group, target, lang, engine_overrides)
                        .await
                    {
                        Ok(true) => {
                            // Spawned -- loop back to acquire the permit.
                            continue;
                        }
                        Ok(false) => {
                            // At capacity -- fall through to async wait.
                        }
                        Err(e) => return Err(e),
                    }
                }
            }

            // All workers busy and at capacity. Wait for a permit.
            let permit = group
                .available
                .acquire()
                .await
                .map_err(|_| WorkerError::SpawnFailed("worker pool semaphore closed".into()))?;
            permit.forget();

            match group.idle.lock().unwrap().pop_front() {
                Some(handle) => {
                    return Ok(CheckedOutWorker {
                        handle: Some(handle),
                        group: group.clone(),
                    });
                }
                None => {
                    // Shouldn't happen -- re-release and retry.
                    group.available.add_permits(1);
                    continue;
                }
            }
        }
    }

    /// Dispatch a batch inference request to a single worker.
    ///
    /// Checks out an idle worker (or spawns one), sends the batch infer
    /// request, and returns the response. Used for the new infer protocol
    /// where the server owns CHAT parsing/serialization and the worker
    /// provides pure NLP inference.
    pub async fn dispatch_batch_infer(
        &self,
        lang: &LanguageCode3,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        let mut worker = self
            .checkout(
                &WorkerTarget::infer_task(request.task),
                lang,
                &self.config.engine_overrides,
            )
            .await?;
        worker.batch_infer(request).await
    }

    /// Dispatch one typed worker-protocol V2 execute request to an infer-task
    /// worker.
    pub async fn dispatch_execute_v2(
        &self,
        lang: &LanguageCode3,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let (target, worker_lang, engine_overrides) =
            execute_v2_worker_key(lang, request, &self.config.engine_overrides)?;
        let mut worker = self
            .checkout(&target, &worker_lang, &engine_overrides)
            .await?;
        worker.execute_v2(request).await
    }

    /// Detect capabilities by spawning a probe worker.
    ///
    /// Spawns a temporary worker with a representative infer target, queries
    /// capabilities, and returns the full `WorkerCapabilities` struct (commands,
    /// free-threaded flag, infer tasks).
    pub async fn detect_capabilities(&self) -> Result<WorkerCapabilities, WorkerError> {
        let config = WorkerConfig {
            python_path: self.config.python_path.clone(),
            target: WorkerTarget::infer_task(InferTask::Morphosyntax),
            lang: "eng".into(),
            num_speakers: NumSpeakers(1),
            engine_overrides: self.config.engine_overrides.clone(),
            test_echo: self.config.test_echo,
            ready_timeout_s: self.config.ready_timeout_s,
            verbose: self.config.verbose,
            runtime: self.config.runtime.clone(),
        };

        let mut handle = match WorkerHandle::spawn(config).await {
            Ok(h) => h,
            Err(e) => {
                warn!(error = %e, "Failed to spawn probe worker for capabilities detection");
                return Err(e);
            }
        };

        let caps = match handle.capabilities().await {
            Ok(c) => c,
            Err(e) => {
                warn!(error = %e, "Failed to query worker capabilities");
                if let Err(shutdown_err) = handle.shutdown().await {
                    warn!(error = %shutdown_err, "Failed to shut down probe worker");
                }
                return Err(e);
            }
        };

        if let Err(e) = handle.shutdown().await {
            warn!(error = %e, "Failed to shut down probe worker");
        }

        info!(
            capabilities = ?caps.commands,
            infer_tasks = ?caps.infer_tasks,
            "Detected worker capabilities"
        );
        Ok(caps)
    }

    /// Pre-start workers for the given commands (warmup).
    ///
    /// Each command spawns concurrently so independent models load in
    /// parallel rather than sequentially.  The caller is responsible for
    /// setting [`mark_warmup_complete()`] after this returns (or spawning
    /// this method in a background task).
    pub async fn warmup(&self, commands: &[(String, String)]) {
        // Resolve all (target, group) pairs up front so the spawned tasks
        // only need the group Arc and config, not a &self borrow.
        struct WarmupItem {
            target: WorkerTarget,
            lang: LanguageCode3,
            group: Arc<WorkerGroup>,
            engine_overrides: String,
        }

        let items: Vec<WarmupItem> = commands
            .iter()
            .filter_map(|(command, lang)| {
                let command = CommandName::from(command.clone());
                let lang = LanguageCode3::from(lang.clone());
                let target = WorkerTarget::for_command(&command);
                match target {
                    Some(target) => {
                        let group = self.get_or_create_group(
                            &target,
                            &lang,
                            &self.config.engine_overrides,
                        );
                        Some(WarmupItem {
                            target,
                            lang,
                            group,
                            engine_overrides: self.config.engine_overrides.clone(),
                        })
                    }
                    None => {
                        warn!(command = %command, lang = %lang, "Skipping warmup for unknown command target");
                        None
                    }
                }
            })
            .collect();

        let mut set = tokio::task::JoinSet::new();
        for item in items {
            let config = self.config.clone();
            set.spawn(async move {
                let target_label = item.target.label().to_string();
                let wc = WorkerConfig {
                    python_path: config.python_path.clone(),
                    target: item.target,
                    lang: item.lang.clone(),
                    num_speakers: NumSpeakers(1),
                    engine_overrides: item.engine_overrides.clone(),
                    test_echo: config.test_echo,
                    ready_timeout_s: config.ready_timeout_s,
                    verbose: config.verbose,
                    runtime: config.runtime.clone(),
                };

                // Claim a slot atomically.
                let max = config.max_workers_per_key;
                loop {
                    let current = item.group.total.load(Ordering::Relaxed);
                    if current >= max {
                        info!(target = %target_label, lang = %item.lang, "Worker already at capacity");
                        return;
                    }
                    match item.group.total.compare_exchange(
                        current,
                        current + 1,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(_) => continue,
                    }
                }

                // Serialize bootstrap for this key.
                let _guard = item.group.bootstrap.lock().await;

                match WorkerHandle::spawn(wc).await {
                    Ok(handle) => {
                        item.group.idle.lock().unwrap().push_back(handle);
                        item.group.available.add_permits(1);
                        info!(target = %target_label, lang = %item.lang, "Worker warmed up");
                    }
                    Err(e) => {
                        item.group.total.fetch_sub(1, Ordering::Relaxed);
                        error!(target = %target_label, lang = %item.lang, error = %e, "Warmup failed");
                    }
                }
            });
        }

        // Wait for all concurrent warmup spawns.
        while set.join_next().await.is_some() {}
    }

    /// Transition warmup state to `InProgress`.
    pub fn mark_warmup_started(&self) {
        self.warmup_status
            .store(WarmupStatus::InProgress.as_u8(), Ordering::Release);
    }

    /// Transition warmup state to `Complete`.
    pub fn mark_warmup_complete(&self) {
        self.warmup_status
            .store(WarmupStatus::Complete.as_u8(), Ordering::Release);
    }

    /// Current warmup lifecycle state.
    pub fn warmup_status(&self) -> WarmupStatus {
        WarmupStatus::from_u8(self.warmup_status.load(Ordering::Acquire))
    }

    /// Pre-scale workers for a given command/lang up to `target` count.
    ///
    /// Spawns workers eagerly so they're ready before file dispatch begins.
    /// Uses `compare_exchange` on `total` for concurrent-safe slot claiming.
    /// Stops early if a spawn fails.
    pub async fn pre_scale(&self, command: &CommandName, lang: &LanguageCode3, target: usize) {
        let target = target.min(self.config.max_workers_per_key);
        let Some(worker_target) = WorkerTarget::for_command(command) else {
            warn!(command = %command, lang = %lang, "Skipping pre-scale for unknown command target");
            return;
        };
        let group = self.get_or_create_group(&worker_target, lang, &self.config.engine_overrides);

        loop {
            let current = group.total.load(Ordering::Relaxed);
            if current >= target {
                break;
            }

            match self
                .try_spawn_into_group(&group, &worker_target, lang, &self.config.engine_overrides)
                .await
            {
                Ok(true) => {}      // Keep going
                Ok(false) => break, // At capacity
                Err(e) => {
                    warn!(
                        target = %worker_target.label(),
                        lang = %lang,
                        current = group.total.load(Ordering::Relaxed),
                        target = target,
                        error = %e,
                        "Pre-scale spawn failed, stopping early"
                    );
                    break;
                }
            }
        }
    }

    /// Shut down all workers gracefully.
    ///
    /// Idle workers are shut down immediately.  Checked-out workers (currently
    /// processing a request) are logged as warnings -- they'll be killed when
    /// the `CheckedOutWorker` RAII guard drops.
    pub async fn shutdown(&self) {
        self.cancel.cancel();

        let all_groups: Vec<(WorkerKey, Arc<WorkerGroup>)> = {
            let mut groups = self.groups.lock().unwrap();
            groups.drain().collect()
        };

        for (key, group) in all_groups {
            let workers: Vec<WorkerHandle> = { group.idle.lock().unwrap().drain(..).collect() };
            let idle_count = workers.len();
            let total = group.total.load(Ordering::Relaxed);
            let checked_out = total.saturating_sub(idle_count);

            if checked_out > 0 {
                warn!(
                    target = %key.0.label(),
                    lang = %key.1,
                    engine_overrides = %key.2,
                    checked_out,
                    "Workers still checked out during shutdown — \
                     they will be killed when their RAII guard drops"
                );
            }

            // Decrement total for drained workers
            group.total.fetch_sub(idle_count, Ordering::Relaxed);

            for mut handle in workers {
                if let Err(e) = handle.shutdown_in_place().await {
                    warn!(
                        target = %key.0.label(),
                        lang = %key.1,
                        engine_overrides = %key.2,
                        error = %e,
                        "Error shutting down worker"
                    );
                }
            }
        }
    }

    /// Check if there are idle workers for a given `(command, lang)` key.
    ///
    /// Used by the memory gate to skip the system memory check when reusable
    /// workers already exist -- those workers are already loaded, so no new
    /// memory allocation is needed.
    pub fn has_idle_workers(&self, command: &CommandName, lang: &LanguageCode3) -> bool {
        let Some(target) = WorkerTarget::for_command(command) else {
            return false;
        };
        let groups = self.groups.lock().unwrap();
        groups.iter().any(|((group_target, group_lang, _), group)| {
            *group_target == target && group_lang == lang && !group.idle.lock().unwrap().is_empty()
        })
    }

    /// Number of active workers (total across all keys, including checked-out).
    pub fn worker_count(&self) -> usize {
        let groups = self.groups.lock().unwrap();
        groups
            .values()
            .map(|g| g.total.load(Ordering::Relaxed))
            .sum()
    }

    /// Active worker keys: `["infer:asr:eng (2 total, 1 idle)", ...]`.
    pub fn worker_keys(&self) -> Vec<String> {
        let groups = self.groups.lock().unwrap();
        let mut keys: Vec<String> = groups
            .iter()
            .map(|((target, lang, engine_overrides), group)| {
                let total = group.total.load(Ordering::Relaxed);
                let idle = group.idle.lock().unwrap().len();
                let suffix = if engine_overrides.is_empty() {
                    String::new()
                } else {
                    format!(":{}", engine_overrides)
                };
                format!(
                    "{}:{lang}{suffix} ({total} total, {idle} idle)",
                    target.label()
                )
            })
            .collect();
        keys.sort();
        keys
    }

    /// Summary of idle workers: `["infer:fa:eng:pid=1234:transport=stdio", ...]`.
    ///
    /// Only reports idle workers (checked-out workers are invisible).
    /// The total count (including checked-out) is available via `worker_count()`.
    pub fn worker_summary(&self) -> Vec<String> {
        let groups = self.groups.lock().unwrap();
        let mut summary = Vec::new();
        for group in groups.values() {
            let idle = group.idle.lock().unwrap();
            for worker in idle.iter() {
                summary.push(format!(
                    "{}:{}:pid={}:transport={}",
                    worker.target_label(),
                    worker.lang(),
                    worker.pid(),
                    worker.transport()
                ));
            }
        }
        summary.sort();
        summary
    }
}

/// Map one V2 task family onto the existing worker bootstrap target space.
fn infer_task_for_execute_v2(task: InferenceTaskV2) -> Result<InferTask, WorkerError> {
    match task {
        InferenceTaskV2::Morphosyntax => Ok(InferTask::Morphosyntax),
        InferenceTaskV2::Utseg => Ok(InferTask::Utseg),
        InferenceTaskV2::Translate => Ok(InferTask::Translate),
        InferenceTaskV2::Coref => Ok(InferTask::Coref),
        InferenceTaskV2::Asr => Ok(InferTask::Asr),
        InferenceTaskV2::ForcedAlignment => Ok(InferTask::Fa),
        InferenceTaskV2::Speaker => Ok(InferTask::Speaker),
        InferenceTaskV2::Opensmile => Ok(InferTask::Opensmile),
        InferenceTaskV2::Avqi => Ok(InferTask::Avqi),
    }
}

fn execute_v2_worker_key(
    lang: &LanguageCode3,
    request: &ExecuteRequestV2,
    default_engine_overrides: &str,
) -> Result<WorkerKey, WorkerError> {
    let infer_task = infer_task_for_execute_v2(request.task)?;
    let engine_overrides =
        execute_v2_engine_overrides(request).unwrap_or_else(|| default_engine_overrides.to_owned());
    Ok((
        WorkerTarget::infer_task(infer_task),
        lang.clone(),
        engine_overrides,
    ))
}

fn execute_v2_engine_overrides(request: &ExecuteRequestV2) -> Option<String> {
    match &request.payload {
        TaskRequestV2::Asr(request) => asr_backend_override_name(request.backend)
            .map(|backend| format!(r#"{{"asr":"{backend}"}}"#)),
        TaskRequestV2::ForcedAlignment(request) => Some(format!(
            r#"{{"fa":"{}"}}"#,
            fa_backend_override_name(request.backend)
        )),
        _ => None,
    }
}

fn asr_backend_override_name(backend: AsrBackendV2) -> Option<&'static str> {
    match backend {
        AsrBackendV2::LocalWhisper => Some("whisper"),
        AsrBackendV2::HkTencent => Some("tencent"),
        AsrBackendV2::HkAliyun => Some("aliyun"),
        AsrBackendV2::HkFunaudio => Some("funaudio"),
        AsrBackendV2::Revai => None,
    }
}

fn fa_backend_override_name(backend: FaBackendV2) -> &'static str {
    match backend {
        FaBackendV2::Whisper => "whisper",
        FaBackendV2::Wave2vec => "wave2vec",
        FaBackendV2::Wav2vecCanto => "wav2vec_canto",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::worker_v2::{
        AsrInputV2, AsrRequestV2, FaTextModeV2, ForcedAlignmentRequestV2, PreparedAudioInputV2,
        WorkerArtifactIdV2, WorkerRequestIdV2,
    };

    fn request_with_payload(task: InferenceTaskV2, payload: TaskRequestV2) -> ExecuteRequestV2 {
        ExecuteRequestV2 {
            request_id: WorkerRequestIdV2::from("req-1"),
            task,
            payload,
            attachments: Vec::new(),
        }
    }

    /// The V2 execute path should reuse the established infer-task worker
    /// targets instead of inventing a separate bootstrap namespace.
    #[test]
    fn maps_forced_alignment_execute_v2_to_fa_worker_target() {
        assert_eq!(
            infer_task_for_execute_v2(InferenceTaskV2::ForcedAlignment).unwrap(),
            InferTask::Fa
        );
    }

    #[test]
    fn execute_v2_asr_worker_key_uses_request_backend_override() {
        let request = request_with_payload(
            InferenceTaskV2::Asr,
            TaskRequestV2::Asr(AsrRequestV2 {
                lang: "fra".into(),
                backend: AsrBackendV2::LocalWhisper,
                input: AsrInputV2::PreparedAudio(PreparedAudioInputV2 {
                    audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                }),
            }),
        );

        let key = execute_v2_worker_key(
            &LanguageCode3::from("fra"),
            &request,
            r#"{"asr":"tencent"}"#,
        )
        .unwrap();

        assert_eq!(key.0, WorkerTarget::infer_task(InferTask::Asr));
        assert_eq!(key.1, LanguageCode3::from("fra"));
        assert_eq!(key.2, r#"{"asr":"whisper"}"#);
    }

    #[test]
    fn execute_v2_fa_worker_key_uses_request_backend_override() {
        let request = request_with_payload(
            InferenceTaskV2::ForcedAlignment,
            TaskRequestV2::ForcedAlignment(ForcedAlignmentRequestV2 {
                backend: FaBackendV2::Wave2vec,
                payload_ref_id: WorkerArtifactIdV2::from("payload-1"),
                audio_ref_id: WorkerArtifactIdV2::from("audio-1"),
                text_mode: FaTextModeV2::SpaceJoined,
                pauses: false,
            }),
        );

        let key =
            execute_v2_worker_key(&LanguageCode3::from("eng"), &request, r#"{"fa":"whisper"}"#)
                .unwrap();

        assert_eq!(key.0, WorkerTarget::infer_task(InferTask::Fa));
        assert_eq!(key.1, LanguageCode3::from("eng"));
        assert_eq!(key.2, r#"{"fa":"wave2vec"}"#);
    }
}
