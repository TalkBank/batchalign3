//! `WorkerPool` — manages multiple Python worker processes.
//!
//! Workers are keyed by `(worker profile, lang, engine overrides)`. The
//! profile space is infer-task-only (for example `infer:asr` or
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
mod execute_v2;
mod lifecycle;
pub(crate) mod reaper;
pub(crate) mod shared_gpu;
mod status;

pub use checkout::CheckedOutWorker;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

use crate::api::{LanguageCode3, NumSpeakers, ReleasedCommand, WorkerLanguage};
use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::{
    BatchInferRequest, BatchInferResponse, WorkerCapabilities, WorkerPid, WorkerProfile,
};
use tokio::sync::{Mutex as AsyncMutex, Semaphore};
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle, WorkerRuntimeConfig};
use crate::worker::python::resolve_python_executable;
use crate::worker::registry;
use crate::worker::tcp_handle::{TcpWorkerHandle, TcpWorkerInfo};

// ---------------------------------------------------------------------------
// Poison-recovery helper for std::sync::Mutex
// ---------------------------------------------------------------------------

/// Lock a `std::sync::Mutex`, recovering from poison if a previous thread
/// panicked while holding it.
///
/// All `std::sync::Mutex` instances in the worker pool guard `VecDeque` or
/// `HashMap` containers with short (microsecond) critical sections. If a
/// panic occurs during a push/pop, the data structure may have been partially
/// mutated, but it is still structurally valid -- the worst case is a
/// missing or double-counted worker, which the health checker will reconcile.
/// Recovering from poison keeps the server alive instead of cascading the
/// panic into every subsequent request.
pub(super) fn lock_recovered<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|poisoned| {
        warn!("Recovering from poisoned std::sync::Mutex in worker pool");
        poisoned.into_inner()
    })
}

/// Key for looking up workers: (worker profile, lang, engine overrides).
pub(super) type WorkerKey = (WorkerProfile, WorkerLanguage, String);

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

/// Default maximum workers per `(profile, lang, engine_overrides)` key.
///
/// Lowered from 8 to 4 to prevent OOM on 64 GB developer machines where
/// GPU workers consume 13-15 GB each (8 × 15 GB = 120 GB → crash).
/// Override via `max_workers_per_key` in `server.yaml` for production
/// servers with more RAM (e.g., net with 256 GB).
const DEFAULT_MAX_WORKERS_PER_KEY: usize = 4;

/// Absolute ceiling on total workers. Even with unlimited RAM, never spawn
/// more than this many concurrent Python processes.
const ABSOLUTE_MAX_TOTAL_WORKERS: usize = 32;

/// RAM budget per worker for the global cap heuristic (6 GB).
///
/// This is a conservative median across all profiles. GPU workers actually
/// use 4-15 GB, Stanza workers 2-8 GB. Using 6 GB prevents the heuristic
/// from allowing more workers than physical RAM can support.
const RAM_PER_WORKER_BYTES: u64 = 6 * 1024 * 1024 * 1024;

/// Compute a default global worker cap from available system memory.
///
/// Uses `available_memory / 6GB`, clamped to `[2, 32]`. Falls back to 4
/// if sysinfo reports 0 (macOS undercounts).
fn default_max_total_workers() -> usize {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available = sys.available_memory(); // bytes
    if available == 0 {
        return 4; // sysinfo couldn't read memory
    }
    let computed = (available / RAM_PER_WORKER_BYTES) as usize;
    computed.clamp(2, ABSOLUTE_MAX_TOTAL_WORKERS)
}

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
    /// Maximum workers per `(profile, lang)` key. Default: 8.
    /// The pool is the capacity ceiling; the runner controls per-job
    /// concurrency via a semaphore.
    pub max_workers_per_key: usize,
    /// Hard ceiling on total workers across all keys. Prevents OOM when
    /// many different `(profile, lang, engine_overrides)` keys are active
    /// simultaneously (e.g. multi-language test suites, concurrent jobs).
    /// Default: computed from available RAM / 4GB per worker, capped at 32.
    /// 0 = use computed default.
    pub max_total_workers: usize,
    /// Verbosity level forwarded to Python workers (0=warn, 1=info, 2=debug).
    pub verbose: u8,
    /// Engine overrides as a JSON string, passed to every spawned worker via
    /// `--engine-overrides`. Empty string means no overrides.
    pub engine_overrides: String,
    /// Runtime-owned worker launch inputs (device policy, injected creds).
    pub runtime: WorkerRuntimeConfig,
    /// Timeout override for audio-heavy tasks (ASR, FA, speaker).
    /// 0 = use built-in default (1800).
    pub audio_task_timeout_s: u64,
    /// Timeout override for lightweight analysis tasks (OpenSMILE, AVQI).
    /// 0 = use built-in default (120).
    pub analysis_task_timeout_s: u64,
    /// Path to the worker registry file. Empty = default
    /// (`~/.batchalign3/workers.json`).
    pub worker_registry_path: String,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            python_path: resolve_python_executable(),
            health_check_interval_s: 30,
            idle_timeout_s: 600, // 10 minutes
            ready_timeout_s: 300,
            test_echo: false,
            max_workers_per_key: DEFAULT_MAX_WORKERS_PER_KEY,
            max_total_workers: 0, // 0 = use computed default
            verbose: 0,
            engine_overrides: String::new(),
            runtime: WorkerRuntimeConfig::default(),
            audio_task_timeout_s: 0,
            analysis_task_timeout_s: 0,
            worker_registry_path: String::new(),
        }
    }
}

impl PoolConfig {
    /// Resolved global worker cap: uses `max_total_workers` if nonzero,
    /// otherwise computes from available system memory.
    pub fn effective_max_total_workers(&self) -> usize {
        if self.max_total_workers > 0 {
            self.max_total_workers
        } else {
            default_max_total_workers()
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerGroup — per (profile, lang) key
// ---------------------------------------------------------------------------

/// A group of workers for a single `(profile, lang)` key.
///
/// Each group independently tracks its own pool of workers. Workers are
/// spawned lazily on first demand and capped at `max_workers_per_key`.
/// The group uses a split concurrency model: a semaphore for async
/// waiting and a mutex for the actual worker queue, so the mutex is
/// never held across an `.await` point.
pub(super) struct WorkerGroup {
    /// Owned worker handles that are currently idle (not checked out).
    ///
    /// Protected by a `std::sync::Mutex` (not `tokio::sync::Mutex`)
    /// because it is held only for the duration of a `push_back` or
    /// `pop_front` -- microseconds, never across an `.await`. This avoids
    /// the overhead of a tokio-aware mutex and is safe because the
    /// critical section cannot yield.
    pub(super) idle: std::sync::Mutex<VecDeque<WorkerHandle>>,

    /// Semaphore with one permit per idle worker.
    ///
    /// `checkout()` acquires a permit (blocking asynchronously if all
    /// workers are busy), then pops from `idle`. When a `CheckedOutWorker`
    /// is dropped, it pushes the worker back into `idle` and adds a
    /// permit, waking the next waiter. Permits are managed manually
    /// (`.forget()` after acquire, `.add_permits(1)` on return) rather
    /// than via RAII `SemaphorePermit` guards.
    available: Semaphore,

    /// TCP worker handles discovered from the registry. These are
    /// persistent daemons that survive server restarts.
    pub(super) tcp_workers: std::sync::Mutex<VecDeque<TcpWorkerHandle>>,

    /// Semaphore with one permit per idle TCP worker.
    tcp_available: Semaphore,

    /// Total number of live workers in this group: idle + checked-out
    /// (both stdio and TCP).
    ///
    /// `AtomicUsize` so that `worker_count()` and spawn-cap checks can
    /// read it without acquiring any mutex. Incremented in
    /// `try_claim_spawn_slot()` (via `compare_exchange`) before the
    /// worker is spawned, and decremented when a worker is removed
    /// (idle timeout, health failure, or `CheckedOutWorker::take()`).
    pub(super) total: AtomicUsize,

    /// Serialize worker bootstrap for one key.
    ///
    /// This prevents a burst of concurrent requests from launching multiple
    /// heavy Python workers for the same `(profile, lang, engine_overrides)`
    /// bucket at once, which smooths model-loading spikes without changing the
    /// eventual steady-state concurrency of the pool.
    bootstrap: AsyncMutex<()>,
}

impl WorkerGroup {
    fn new() -> Self {
        Self {
            idle: std::sync::Mutex::new(VecDeque::new()),
            available: Semaphore::new(0),
            tcp_workers: std::sync::Mutex::new(VecDeque::new()),
            tcp_available: Semaphore::new(0),
            total: AtomicUsize::new(0),
            bootstrap: AsyncMutex::new(()),
        }
    }
}

/// Shared map of worker groups, accessible from both the pool and background tasks.
pub(super) type GroupsMap = Arc<std::sync::Mutex<HashMap<WorkerKey, Arc<WorkerGroup>>>>;

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/// Key for shared GPU workers: (lang, engine_overrides).
pub(super) type GpuWorkerKey = (WorkerLanguage, String);

/// Manages a pool of Python worker processes.
pub struct WorkerPool {
    config: PoolConfig,
    /// Sequential worker groups (Stanza, IO profiles).
    groups: GroupsMap,
    /// Shared GPU workers for concurrent V2 dispatch (GPU profile, stdio).
    gpu_workers: Arc<tokio::sync::Mutex<HashMap<GpuWorkerKey, Arc<shared_gpu::SharedGpuWorker>>>>,
    /// Shared GPU workers discovered from registry (TCP transport).
    gpu_tcp_workers:
        Arc<tokio::sync::Mutex<HashMap<GpuWorkerKey, Arc<shared_gpu::SharedGpuTcpWorker>>>>,
    cancel: CancellationToken,
    /// Background warmup lifecycle state.
    warmup_status: AtomicU8,
    /// Lazily detected worker capabilities (populated on first worker spawn).
    lazy_capabilities: std::sync::OnceLock<WorkerCapabilities>,
}

impl WorkerPool {
    /// Create a new worker pool. Call [`start_background_tasks`](Self::start_background_tasks)
    /// to begin health checking and idle timeout.
    ///
    /// On creation, reaps any orphaned worker processes left by crashed or
    /// killed servers (Layer 3 of the OOM defense).
    pub fn new(config: PoolConfig) -> Self {
        let effective_cap = config.effective_max_total_workers();
        info!(
            max_total_workers = effective_cap,
            max_workers_per_key = config.max_workers_per_key,
            "Worker pool created"
        );

        // Layer 3: reap orphans from any previous server that crashed.
        let reaped = reaper::reap_orphaned_workers();
        if reaped > 0 {
            info!(reaped, "Cleaned up orphaned workers from previous server");
        }

        Self {
            config,
            groups: Arc::new(std::sync::Mutex::new(HashMap::new())),
            gpu_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            gpu_tcp_workers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            cancel: CancellationToken::new(),
            warmup_status: AtomicU8::new(WarmupStatus::NotStarted.as_u8()),
            lazy_capabilities: std::sync::OnceLock::new(),
        }
    }

    /// Discover pre-started TCP workers from the registry file.
    ///
    /// Reads `workers.json`, health-checks each entry, and integrates healthy
    /// workers into the pool. GPU workers become `SharedGpuWorker` entries;
    /// non-GPU workers are not integrated into the sequential pool (they use
    /// TCP handles directly via the dispatch path).
    ///
    /// Returns the number of workers discovered and integrated.
    pub async fn discover_from_registry(&self) -> usize {
        let registry_path = if self.config.worker_registry_path.is_empty() {
            registry::default_registry_path()
        } else {
            std::path::PathBuf::from(&self.config.worker_registry_path)
        };

        let discovered = registry::discover_workers(
            &registry_path,
            self.config.audio_task_timeout_s,
            self.config.analysis_task_timeout_s,
        )
        .await;

        if discovered.is_empty() {
            return 0;
        }

        let count = discovered.len();
        info!(count, "Discovered worker(s) from registry");

        // Integrate GPU workers into the shared GPU worker map.
        // Non-GPU TCP workers are tracked in a separate TCP worker map.
        for worker in &discovered {
            if worker.profile.is_concurrent() {
                let info = TcpWorkerInfo {
                    host: worker.entry.host.clone(),
                    port: worker.entry.port,
                    profile: worker.profile,
                    lang: worker.lang.clone(),
                    engine_overrides: worker.entry.engine_overrides.clone(),
                    pid: WorkerPid(worker.entry.pid),
                    audio_task_timeout_s: self.config.audio_task_timeout_s,
                    analysis_task_timeout_s: self.config.analysis_task_timeout_s,
                };

                match shared_gpu::SharedGpuTcpWorker::connect(info).await {
                    Ok(shared) => {
                        let key = (worker.lang.clone(), worker.entry.engine_overrides.clone());
                        self.gpu_tcp_workers
                            .lock()
                            .await
                            .entry(key)
                            .or_insert_with(|| Arc::new(shared));
                        info!(
                            profile = %worker.entry.profile,
                            lang = %worker.entry.lang,
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            "Integrated GPU TCP worker"
                        );
                    }
                    Err(e) => {
                        warn!(
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            error = %e,
                            "Failed to integrate GPU TCP worker"
                        );
                    }
                }
            } else {
                // For non-GPU TCP workers, add them to the sequential pool.
                let info = TcpWorkerInfo {
                    host: worker.entry.host.clone(),
                    port: worker.entry.port,
                    profile: worker.profile,
                    lang: worker.lang.clone(),
                    engine_overrides: worker.entry.engine_overrides.clone(),
                    pid: WorkerPid(worker.entry.pid),
                    audio_task_timeout_s: self.config.audio_task_timeout_s,
                    analysis_task_timeout_s: self.config.analysis_task_timeout_s,
                };

                match TcpWorkerHandle::connect(info).await {
                    Ok(handle) => {
                        let key = (
                            worker.profile,
                            worker.lang.clone(),
                            worker.entry.engine_overrides.clone(),
                        );
                        let group = self.get_or_create_group(
                            &worker.profile,
                            &worker.lang,
                            &worker.entry.engine_overrides,
                        );
                        lock_recovered(&group.tcp_workers).push_back(handle);
                        group.tcp_available.add_permits(1);
                        group.total.fetch_add(1, Ordering::Relaxed);
                        info!(
                            profile = %worker.entry.profile,
                            lang = %worker.entry.lang,
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            "Integrated non-GPU TCP worker into pool (key={:?})",
                            key
                        );
                    }
                    Err(e) => {
                        warn!(
                            host = %worker.entry.host,
                            port = worker.entry.port,
                            error = %e,
                            "Failed to integrate non-GPU TCP worker"
                        );
                    }
                }
            }
        }

        // Probe capabilities from the first discovered TCP worker so that
        // `infer_tasks` is populated even when no stdio workers are spawned.
        // Without this, servers that only have pre-started TCP daemons start
        // with `infer_tasks = Vec::new()` and reject every job.
        if self.lazy_capabilities.get().is_none() && !discovered.is_empty() {
            let first = &discovered[0];
            let probe_info = TcpWorkerInfo {
                host: first.entry.host.clone(),
                port: first.entry.port,
                profile: first.profile,
                lang: first.lang.clone(),
                engine_overrides: first.entry.engine_overrides.clone(),
                pid: WorkerPid(first.entry.pid),
                audio_task_timeout_s: self.config.audio_task_timeout_s,
                analysis_task_timeout_s: self.config.analysis_task_timeout_s,
            };
            match TcpWorkerHandle::connect(probe_info).await {
                Ok(mut probe_handle) => match probe_handle.capabilities().await {
                    Ok(caps) => {
                        info!(
                            infer_tasks = ?caps.infer_tasks,
                            engine_versions = ?caps.engine_versions,
                            "Detected worker capabilities from TCP registry worker"
                        );
                        let _ = self.lazy_capabilities.set(caps);
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to probe capabilities from TCP registry worker");
                    }
                },
                Err(e) => {
                    warn!(error = %e, "Failed to connect to TCP registry worker for capability probe");
                }
            }
        }

        count
    }

    /// Check out an idle worker or spawn a new one.
    ///
    /// 1. Try to acquire a semaphore permit immediately.
    /// 2. If none available, try to spawn a new worker (if under capacity).
    /// 3. If at capacity, wait for a permit (async suspend).
    /// 4. Pop from the idle queue and wrap in `CheckedOutWorker` (RAII guard).
    async fn checkout(
        &self,
        profile: &WorkerProfile,
        lang: &WorkerLanguage,
        engine_overrides: &str,
    ) -> Result<CheckedOutWorker, WorkerError> {
        let group = self.get_or_create_group(profile, lang, engine_overrides);

        loop {
            // Try to acquire a permit without waiting.
            match group.available.try_acquire() {
                Ok(permit) => {
                    permit.forget(); // We manage permits manually
                    match lock_recovered(&group.idle).pop_front() {
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
                        .try_spawn_into_group(&group, profile, lang, engine_overrides)
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

            match lock_recovered(&group.idle).pop_front() {
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
    /// Tries TCP workers first (from registry), then falls back to stdio
    /// workers. Checks out an idle worker (or spawns one), sends the batch
    /// infer request, and returns the response.
    pub async fn dispatch_batch_infer(
        &self,
        lang: &LanguageCode3,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        let profile = WorkerProfile::for_task(request.task);
        let engine_overrides = &self.config.engine_overrides;
        let worker_lang = WorkerLanguage::from(lang);

        // Try TCP worker first.
        if let Some(mut tcp_handle) =
            self.try_checkout_tcp(&profile, &worker_lang, engine_overrides)
        {
            let result = tcp_handle.batch_infer(request).await;
            self.return_tcp_worker(tcp_handle, &profile, &worker_lang, engine_overrides);
            return result;
        }

        // Fall back to stdio worker.
        let mut worker = self
            .checkout(&profile, &worker_lang, engine_overrides)
            .await?;
        worker.batch_infer(request).await
    }

    /// Try to check out a TCP worker handle (non-blocking).
    fn try_checkout_tcp(
        &self,
        profile: &WorkerProfile,
        lang: &WorkerLanguage,
        engine_overrides: &str,
    ) -> Option<TcpWorkerHandle> {
        let key: WorkerKey = (*profile, lang.clone(), engine_overrides.to_owned());
        let groups = lock_recovered(&self.groups);
        let group = groups.get(&key)?;
        match group.tcp_available.try_acquire() {
            Ok(permit) => {
                permit.forget();
                lock_recovered(&group.tcp_workers).pop_front()
            }
            Err(_) => None,
        }
    }

    /// Return a TCP worker handle to the pool.
    fn return_tcp_worker(
        &self,
        handle: TcpWorkerHandle,
        profile: &WorkerProfile,
        lang: &WorkerLanguage,
        engine_overrides: &str,
    ) {
        let key: WorkerKey = (*profile, lang.clone(), engine_overrides.to_owned());
        let groups = lock_recovered(&self.groups);
        if let Some(group) = groups.get(&key) {
            lock_recovered(&group.tcp_workers).push_back(handle);
            group.tcp_available.add_permits(1);
        }
    }

    /// Dispatch one typed worker-protocol V2 execute request.
    ///
    /// GPU profile tasks are routed to a shared concurrent worker (multiple
    /// requests in flight to one process). Non-GPU tasks try TCP workers first,
    /// then fall back to the traditional exclusive checkout model.
    pub async fn dispatch_execute_v2(
        &self,
        lang: impl Into<WorkerLanguage>,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let lang = lang.into();
        let (profile, worker_lang, engine_overrides) =
            execute_v2_worker_key(lang, request, &self.config.engine_overrides)?;

        if profile.is_concurrent() {
            return self
                .dispatch_gpu_execute_v2(&worker_lang, &engine_overrides, request)
                .await;
        }

        // Try TCP worker first.
        if let Some(mut tcp_handle) =
            self.try_checkout_tcp(&profile, &worker_lang, &engine_overrides)
        {
            let result = tcp_handle.execute_v2(request).await;
            self.return_tcp_worker(tcp_handle, &profile, &worker_lang, &engine_overrides);
            return result;
        }

        // Fall back to stdio worker.
        let mut worker = self
            .checkout(&profile, &worker_lang, &engine_overrides)
            .await?;
        worker.execute_v2(request).await
    }

    /// Dispatch a V2 execute request to a GPU worker.
    ///
    /// Tries TCP workers first (discovered from registry), then falls back to
    /// stdio workers. For TCP workers, multiple callers share one worker via
    /// concurrent dispatch. For stdio workers, uses the existing
    /// [`SharedGpuWorker`](shared_gpu::SharedGpuWorker) pattern.
    async fn dispatch_gpu_execute_v2(
        &self,
        lang: &WorkerLanguage,
        engine_overrides: &str,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        // Try TCP worker first (discovered from registry).
        let tcp_key = (lang.clone(), engine_overrides.to_owned());
        {
            let tcp_workers = self.gpu_tcp_workers.lock().await;
            if let Some(tcp_worker) = tcp_workers.get(&tcp_key) {
                return tcp_worker.execute_v2(request).await;
            }
        }

        // Fall back to stdio worker.
        let gpu_worker = self
            .get_or_create_gpu_worker(lang, engine_overrides)
            .await?;
        gpu_worker.execute_v2(request).await
    }

    /// Get or create a shared GPU worker for the given (lang, engine_overrides).
    /// Get or create a shared GPU worker for the given (lang, engine_overrides).
    ///
    /// Holds the lock across the spawn to prevent the TOCTOU race where
    /// multiple concurrent callers each spawn their own worker process.
    /// The spawn includes waiting for the `{"ready": true}` signal, so
    /// the lock is held for 10-30 seconds on first call. This is acceptable
    /// because GPU worker creation is rare (once per lang+overrides combo),
    /// and the `pre_scale` call in the runner ensures the worker exists
    /// before file dispatch begins.
    async fn get_or_create_gpu_worker(
        &self,
        lang: &WorkerLanguage,
        engine_overrides: &str,
    ) -> Result<Arc<shared_gpu::SharedGpuWorker>, WorkerError> {
        let key = (lang.clone(), engine_overrides.to_owned());

        let mut gpu_workers = self.gpu_workers.lock().await;

        // Fast path: worker already exists.
        if let Some(worker) = gpu_workers.get(&key) {
            return Ok(worker.clone());
        }

        // Slow path: spawn while holding the lock to prevent duplicate spawns.
        let config = WorkerConfig {
            python_path: self.config.python_path.clone(),
            profile: WorkerProfile::Gpu,
            lang: lang.clone(),
            num_speakers: NumSpeakers(1),
            engine_overrides: engine_overrides.to_owned(),
            test_echo: self.config.test_echo,
            ready_timeout_s: self.config.ready_timeout_s,
            verbose: self.config.verbose,
            runtime: self.config.runtime.clone(),
            audio_task_timeout_s: self.config.audio_task_timeout_s,
            analysis_task_timeout_s: self.config.analysis_task_timeout_s,
            test_delay_ms: 0,
        };

        let handle = WorkerHandle::spawn(config).await?;
        info!(
            lang = %lang,
            pid = %handle.pid(),
            "GPU worker spawned (concurrent mode)"
        );
        let shared = Arc::new(shared_gpu::SharedGpuWorker::from_handle(handle).await);

        gpu_workers.insert(key, shared.clone());
        Ok(shared)
    }

    /// Query capabilities from an already-spawned worker and cache the result.
    ///
    /// Called once after the first worker spawn. The `OnceLock` ensures this
    /// only runs once even under concurrent job dispatch.
    pub(crate) async fn detect_capabilities_from_worker(
        &self,
        handle: &mut WorkerHandle,
    ) -> Result<(), WorkerError> {
        if self.lazy_capabilities.get().is_some() {
            return Ok(()); // Already detected
        }

        let caps = handle.capabilities().await?;
        info!(
            infer_tasks = ?caps.infer_tasks,
            engine_versions = ?caps.engine_versions,
            "Lazily detected worker capabilities from first spawn"
        );
        let _ = self.lazy_capabilities.set(caps);
        Ok(())
    }

    /// Return lazily detected capabilities, or `None` if no worker has
    /// spawned yet.
    pub fn detected_capabilities(&self) -> Option<&WorkerCapabilities> {
        self.lazy_capabilities.get()
    }

    /// Pre-start workers for the given commands (warmup).
    ///
    /// Spawns **persistent TCP daemon workers** that outlive the server process.
    /// On the next server restart, `discover_from_registry()` finds them already
    /// running — zero cold start. This is the key user-facing improvement:
    /// the first `batchalign3 morphotag` run pays the model-loading cost, but
    /// every subsequent run starts instantly.
    ///
    /// Each command spawns concurrently so independent models load in parallel.
    /// The caller is responsible for setting [`mark_warmup_complete()`] after
    /// this returns (or spawning this method in a background task).
    pub async fn warmup(&self, targets: &[crate::server::WarmupTarget]) {
        use crate::worker::handle::spawn_tcp_daemon;

        struct WarmupItem {
            profile: WorkerProfile,
            lang: WorkerLanguage,
            engine_overrides: String,
        }

        let items: Vec<WarmupItem> = targets
            .iter()
            .filter_map(|target| {
                let profile = WorkerProfile::for_command(target.command);
                match profile {
                    Some(profile) => Some(WarmupItem {
                        profile,
                        lang: target.lang.clone(),
                        engine_overrides: self.config.engine_overrides.clone(),
                    }),
                    None => {
                        warn!(command = %target.command, lang = %target.lang, "Skipping warmup for unknown command profile");
                        None
                    }
                }
            })
            .collect();

        let gpu_tcp_ref = self.gpu_tcp_workers.clone();
        let groups_ref = self.groups.clone();
        let mut set = tokio::task::JoinSet::new();
        for item in items {
            let config = self.config.clone();
            let gpu_tcp_ref = gpu_tcp_ref.clone();
            let groups_ref = groups_ref.clone();
            set.spawn(async move {
                let profile_label = item.profile.label().to_string();
                let wc = WorkerConfig {
                    python_path: config.python_path.clone(),
                    profile: item.profile,
                    lang: item.lang.clone(),
                    num_speakers: NumSpeakers(1),
                    engine_overrides: item.engine_overrides.clone(),
                    test_echo: config.test_echo,
                    ready_timeout_s: config.ready_timeout_s,
                    verbose: config.verbose,
                    runtime: config.runtime.clone(),
                    audio_task_timeout_s: config.audio_task_timeout_s,
                    analysis_task_timeout_s: config.analysis_task_timeout_s,
                    test_delay_ms: 0,
                };

                // Spawn a detached TCP daemon. It registers in workers.json
                // and outlives the server.
                let (pid, port) = match spawn_tcp_daemon(&wc, 0).await {
                    Ok(result) => result,
                    Err(e) => {
                        error!(
                            target = %profile_label,
                            lang = %item.lang,
                            error = %e,
                            "TCP daemon warmup failed"
                        );
                        return;
                    }
                };

                // Connect to the just-spawned daemon.
                let tcp_info = TcpWorkerInfo {
                    host: "127.0.0.1".to_string(),
                    port,
                    profile: item.profile,
                    lang: item.lang.clone(),
                    engine_overrides: item.engine_overrides.clone(),
                    pid: WorkerPid(pid),
                    audio_task_timeout_s: config.audio_task_timeout_s,
                    analysis_task_timeout_s: config.analysis_task_timeout_s,
                };

                if item.profile.is_concurrent() {
                    // GPU warmup — connect as SharedGpuTcpWorker.
                    match shared_gpu::SharedGpuTcpWorker::connect(tcp_info).await {
                        Ok(shared) => {
                            let key = (item.lang.clone(), item.engine_overrides.clone());
                            gpu_tcp_ref
                                .lock()
                                .await
                                .entry(key)
                                .or_insert_with(|| Arc::new(shared));
                            info!(
                                target = %profile_label,
                                lang = %item.lang,
                                pid = pid,
                                port = port,
                                "GPU TCP worker warmed up (persistent daemon)"
                            );
                        }
                        Err(e) => {
                            error!(
                                target = %profile_label,
                                lang = %item.lang,
                                error = %e,
                                "Failed to connect to GPU TCP daemon after spawn"
                            );
                        }
                    }
                } else {
                    // Non-GPU warmup — connect as TcpWorkerHandle.
                    match TcpWorkerHandle::connect(tcp_info).await {
                        Ok(handle) => {
                            let key: WorkerKey = (
                                item.profile,
                                item.lang.clone(),
                                item.engine_overrides.clone(),
                            );
                            let mut groups = lock_recovered(&groups_ref);
                            let group = groups
                                .entry(key)
                                .or_insert_with(|| Arc::new(WorkerGroup::new()))
                                .clone();
                            drop(groups);

                            lock_recovered(&group.tcp_workers).push_back(handle);
                            group.tcp_available.add_permits(1);
                            group.total.fetch_add(1, Ordering::Relaxed);
                            info!(
                                target = %profile_label,
                                lang = %item.lang,
                                pid = pid,
                                port = port,
                                "TCP worker warmed up (persistent daemon)"
                            );
                        }
                        Err(e) => {
                            error!(
                                target = %profile_label,
                                lang = %item.lang,
                                error = %e,
                                "Failed to connect to TCP daemon after spawn"
                            );
                        }
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
    /// Delegates to [`pre_scale_with_overrides`] using the pool's default
    /// engine overrides.
    pub async fn pre_scale(
        &self,
        command: ReleasedCommand,
        lang: impl Into<WorkerLanguage>,
        target: usize,
    ) {
        self.pre_scale_with_overrides(command, lang, target, &self.config.engine_overrides)
            .await;
    }

    /// Pre-scale workers with explicit engine overrides.
    ///
    /// Spawns workers eagerly so they're ready before file dispatch begins.
    /// The `engine_overrides` must match the overrides that dispatch will use
    /// (typically from the job's `CommonOptions`), otherwise the pre-scaled
    /// worker will have a different key than what dispatch looks up.
    ///
    /// **TCP worker shortcut:** If a TCP worker is already discovered from the
    /// registry for this profile/lang, pre-scale is a no-op — the worker is
    /// already running and ready. This eliminates the TOCTOU race, ready
    /// timeout, and cold-start delay that motivated pre-scale in the first
    /// place.
    ///
    /// For GPU-profile commands, pre-creates the `SharedGpuWorker` so all
    /// concurrent file dispatches hit the fast path (no spawn race).
    /// For non-GPU commands, uses `compare_exchange` on `total` for
    /// concurrent-safe slot claiming.
    pub async fn pre_scale_with_overrides(
        &self,
        command: ReleasedCommand,
        lang: impl Into<WorkerLanguage>,
        target: usize,
        engine_overrides: &str,
    ) {
        let lang = lang.into();
        let target = target.min(self.config.max_workers_per_key);
        let Some(profile) = WorkerProfile::for_command(command) else {
            warn!(command = %command, lang = %lang, "Skipping pre-scale for unknown command profile");
            return;
        };

        // TCP worker shortcut: if a TCP worker already exists for this
        // profile/lang, skip spawning — the worker is already running.
        if profile.is_concurrent() {
            let tcp_key = (lang.clone(), engine_overrides.to_owned());
            if self.gpu_tcp_workers.lock().await.contains_key(&tcp_key) {
                info!(
                    command = %command,
                    lang = %lang,
                    "GPU TCP worker already discovered, skipping pre-scale"
                );
                return;
            }
        } else {
            let key: WorkerKey = (profile, lang.clone(), engine_overrides.to_owned());
            let has_tcp = {
                let groups = lock_recovered(&self.groups);
                groups
                    .get(&key)
                    .is_some_and(|g| !lock_recovered(&g.tcp_workers).is_empty())
            };
            if has_tcp {
                info!(
                    command = %command,
                    lang = %lang,
                    profile = %profile.label(),
                    "TCP worker already discovered, skipping pre-scale"
                );
                return;
            }
        }

        // GPU workers use the shared concurrent worker map. Pre-creating the
        // worker here ensures it's ready before file dispatch begins, avoiding
        // the TOCTOU race in `get_or_create_gpu_worker` where multiple tasks
        // would each try to spawn their own worker process.
        if profile.is_concurrent() {
            match self.get_or_create_gpu_worker(&lang, engine_overrides).await {
                Ok(_) => {
                    info!(
                        command = %command,
                        lang = %lang,
                        engine_overrides = %engine_overrides,
                        "GPU worker pre-scaled (ready for concurrent dispatch)"
                    );
                }
                Err(e) => {
                    warn!(
                        command = %command,
                        lang = %lang,
                        error = %e,
                        "GPU worker pre-scale failed"
                    );
                }
            }
            return;
        }

        let group = self.get_or_create_group(&profile, &lang, engine_overrides);

        loop {
            let current = group.total.load(Ordering::Relaxed);
            if current >= target {
                break;
            }

            match self
                .try_spawn_into_group(&group, &profile, &lang, &self.config.engine_overrides)
                .await
            {
                Ok(true) => {}      // Keep going
                Ok(false) => break, // At capacity
                Err(e) => {
                    warn!(
                        target = %profile.label(),
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
    /// the `CheckedOutWorker` RAII guard drops. Shared GPU workers are shut
    /// down via their concurrent shutdown path. TCP workers are disconnected
    /// but not killed (they are managed by the OS service manager).
    pub async fn shutdown(&self) {
        self.cancel.cancel();

        // Shut down shared GPU workers (stdio).
        {
            let mut gpu_workers = self.gpu_workers.lock().await;
            for ((lang, overrides), worker) in gpu_workers.drain() {
                info!(
                    lang = %lang,
                    engine_overrides = %overrides,
                    pid = %worker.pid(),
                    "Shutting down GPU worker"
                );
                worker.shutdown().await;
            }
        }

        // Disconnect shared TCP GPU workers (does not kill the daemon).
        {
            let mut tcp_gpu_workers = self.gpu_tcp_workers.lock().await;
            for ((lang, overrides), worker) in tcp_gpu_workers.drain() {
                info!(
                    lang = %lang,
                    engine_overrides = %overrides,
                    pid = %worker.pid(),
                    "Disconnecting TCP GPU worker"
                );
                worker.shutdown().await;
            }
        }

        let all_groups: Vec<(WorkerKey, Arc<WorkerGroup>)> = {
            let mut groups = lock_recovered(&self.groups);
            groups.drain().collect()
        };

        for (key, group) in all_groups {
            let workers: Vec<WorkerHandle> = { lock_recovered(&group.idle).drain(..).collect() };
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
}

/// Layer 2: kill idle workers when the pool is dropped without calling
/// `shutdown()`. This catches test code and panic unwinds where the pool
/// goes out of scope without graceful shutdown.
///
/// GPU workers behind `tokio::sync::Mutex` cannot be locked outside a
/// runtime, but their shared-worker owners are dropped when Arc refcounts
/// hit zero. The stdio variant's `Drop` impl kills the worker process; the
/// TCP variant only disconnects from the daemon it does not own.
impl Drop for WorkerPool {
    fn drop(&mut self) {
        self.cancel.cancel();

        // Drain all groups and kill workers synchronously.
        // This works even outside a tokio context.
        if let Ok(mut groups) = self.groups.lock() {
            for (_, group) in groups.drain() {
                if let Ok(mut idle) = group.idle.lock() {
                    for handle in idle.drain(..) {
                        // WorkerHandle::Drop sends SIGTERM+SIGKILL.
                        drop(handle);
                    }
                }
            }
        }
    }
}

impl WorkerPool {
    // Status query methods (has_idle_workers, worker_count, worker_keys,
    // worker_summary) live in status.rs for browsability.
}

/// Map one V2 task family onto the existing worker bootstrap profile space.
// V2 execute helpers live in execute_v2.rs for browsability.
use execute_v2::execute_v2_worker_key;
