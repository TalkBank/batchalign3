//! App factory and server lifecycle (create, serve, shutdown).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use tokio::sync::broadcast;
use tracing::{info, warn};

use crate::cache::UtteranceCache;
use crate::config::{RuntimeLayout, ServerConfig};
use crate::db::JobDB;
use crate::error;
use crate::media::MediaResolver;
use crate::queue::{LocalQueueBackend, QueueBackend, QueueDispatcher};
use crate::runner::RunnerContext;
use crate::runtime_supervisor::RuntimeSupervisor;
use crate::state::{
    AppBuildInfo, AppControlPlane, AppEnvironment, AppPaths, AppState, WorkerSubsystem,
    validate_infer_capability_gate,
};
use crate::store::JobStore;
use crate::worker::InferTask;
use crate::worker::pool::{PoolConfig, WorkerPool};
use crate::ws::BROADCAST_CAPACITY;

// ---------------------------------------------------------------------------
// App factory
// ---------------------------------------------------------------------------

/// Prepared worker subsystem that can be reused across multiple app instances.
///
/// Tests use this seam to amortize capability probing and model warmup while
/// still creating a fresh [`AppState`] and runtime-owned filesystem layout for
/// each isolated server session.
#[derive(Clone)]
pub struct PreparedWorkers {
    pool: Arc<WorkerPool>,
    capabilities: Vec<String>,
    infer_tasks: Vec<InferTask>,
    engine_versions: BTreeMap<String, String>,
    test_echo_mode: bool,
}

impl PreparedWorkers {
    /// Return the released command surface discovered during worker probing.
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// Return the infer-task set reported by the prepared worker subsystem.
    pub fn infer_tasks(&self) -> &[InferTask] {
        &self.infer_tasks
    }
}

/// Create the application: open DB, recover state, build router.
///
/// Returns `(Router, Arc<AppState>)` — the caller binds the router to a
/// TCP listener.
///
/// `db_dir` overrides the SQLite database directory (defaults to the runtime
/// state root, typically `~/.batchalign3/`). Useful for tests that need an
/// isolated DB.
pub async fn create_app(
    config: ServerConfig,
    pool_config: PoolConfig,
    jobs_dir: Option<String>,
    db_dir: Option<std::path::PathBuf>,
    build_hash: Option<String>,
) -> Result<(Router, Arc<AppState>), error::ServerError> {
    let layout = RuntimeLayout::from_env();
    create_app_with_runtime(config, pool_config, layout, jobs_dir, db_dir, build_hash).await
}

/// Create the application using an explicit runtime layout for state-owned
/// filesystem roots.
///
/// Warmup runs in the background so the HTTP port binds immediately.
pub async fn create_app_with_runtime(
    config: ServerConfig,
    pool_config: PoolConfig,
    layout: RuntimeLayout,
    jobs_dir: Option<String>,
    db_dir: Option<std::path::PathBuf>,
    build_hash: Option<String>,
) -> Result<(Router, Arc<AppState>), error::ServerError> {
    let workers = prepare_workers_background(&config, pool_config).await?;
    create_app_with_prepared_workers(config, layout, jobs_dir, db_dir, None, build_hash, workers)
        .await
}

/// Probe, validate, and optionally warm a worker pool for reuse.
///
/// The returned [`PreparedWorkers`] value owns a live [`WorkerPool`] plus the
/// capability metadata derived from it. Callers can share that value across
/// multiple app instances to keep expensive model loads hot while still
/// rebuilding the server control plane and runtime-owned temp directories.
///
/// Warmup runs synchronously (all commands spawn concurrently within the call).
/// For non-blocking startup, use [`prepare_workers_background`] which returns
/// immediately after capability probing and spawns warmup in a background task.
pub async fn prepare_workers(
    config: &ServerConfig,
    pool_config: PoolConfig,
) -> Result<PreparedWorkers, error::ServerError> {
    let (prepared, pairs) = probe_workers(config, pool_config).await?;

    if !pairs.is_empty() {
        prepared.pool.warmup(&pairs).await;
    }
    prepared.pool.mark_warmup_complete();

    Ok(prepared)
}

/// Like [`prepare_workers`] but warmup runs as a background `tokio::spawn`
/// task.  The HTTP server can bind its port immediately while models load.
///
/// The returned [`PreparedWorkers`] is ready for use — jobs that arrive
/// before warmup finishes will block on checkout until their required worker
/// spawns, which is correct (no duplicate spawns).
pub async fn prepare_workers_background(
    config: &ServerConfig,
    pool_config: PoolConfig,
) -> Result<PreparedWorkers, error::ServerError> {
    let (prepared, pairs) = probe_workers(config, pool_config).await?;

    if !pairs.is_empty() {
        prepared.pool.mark_warmup_started();
        let warmup_pool = prepared.pool.clone();
        tokio::spawn(async move {
            warmup_pool.warmup(&pairs).await;
            warmup_pool.mark_warmup_complete();
            info!("Background warmup complete");
        });
    } else {
        prepared.pool.mark_warmup_complete();
    }

    Ok(prepared)
}

/// Probe worker capabilities and compute the warmup command list, but do not
/// start warmup yet.  Shared implementation for both synchronous and
/// background warmup entry points.
async fn probe_workers(
    config: &ServerConfig,
    pool_config: PoolConfig,
) -> Result<(PreparedWorkers, Vec<(String, String)>), error::ServerError> {
    let test_echo_mode = pool_config.test_echo;
    let pool = Arc::new(WorkerPool::new(pool_config));
    pool.start_background_tasks();

    // Discover pre-started TCP workers from the registry file.
    // This happens before capability probing and warmup so discovered
    // workers are available immediately — no spawn delay.
    let discovered = pool.discover_from_registry().await;
    if discovered > 0 {
        info!(discovered, "Pre-started TCP workers integrated into pool");
    }

    let worker_caps = pool
        .detect_capabilities()
        .await
        .map_err(error::ServerError::Worker)?;
    let infer_tasks = worker_caps.infer_tasks;
    let engine_versions = worker_caps.engine_versions;
    let capabilities =
        validate_infer_capability_gate(&infer_tasks, &engine_versions, test_echo_mode)?;

    if !infer_tasks.is_empty() {
        info!(infer_tasks = ?infer_tasks, engine_versions = ?engine_versions, "Worker infer capabilities");
    }

    let warmup_cmds = config.resolved_warmup_commands();
    let pairs: Vec<(String, String)> = warmup_cmds
        .iter()
        .filter(|cmd| capabilities.contains(cmd))
        .map(|cmd| (cmd.clone(), config.default_lang.to_string()))
        .collect();

    if pairs.is_empty() {
        info!("Worker warmup disabled or no warmable capabilities");
    } else {
        info!(commands = ?pairs, "Warmup commands resolved");
    }

    Ok((
        PreparedWorkers {
            pool,
            capabilities,
            infer_tasks,
            engine_versions,
            test_echo_mode,
        },
        pairs,
    ))
}

/// Create the application with an already-prepared worker subsystem.
///
/// This keeps the expensive worker pool hot across repeated app lifecycles
/// while giving each app instance a fresh store, runtime supervisor, database,
/// and filesystem layout. `cache_dir` lets tests pin the utterance cache under
/// that owned runtime root instead of falling back to the ambient platform
/// cache directory.
pub async fn create_app_with_prepared_workers(
    config: ServerConfig,
    layout: RuntimeLayout,
    jobs_dir: Option<String>,
    db_dir: Option<std::path::PathBuf>,
    cache_dir: Option<std::path::PathBuf>,
    build_hash: Option<String>,
    workers: PreparedWorkers,
) -> Result<(Router, Arc<AppState>), error::ServerError> {
    let jobs_dir = jobs_dir.unwrap_or_else(|| layout.jobs_dir().to_string_lossy().into_owned());
    let _ = tokio::fs::create_dir_all(&jobs_dir).await;

    // Open database (includes schema migration)
    let db = JobDB::open_with_layout(&layout, db_dir.as_deref()).await?;

    // Recovery: mark interrupted, prune expired
    let interrupted = db.recover_interrupted().await?;
    if !interrupted.is_empty() {
        info!(count = interrupted.len(), "Recovered interrupted jobs");
    }
    let expired_dirs = db.prune_expired(config.job_ttl_days).await?;
    for d in &expired_dirs {
        let _ = tokio::fs::remove_dir_all(d).await;
    }

    // Create broadcast channel for WS
    let (ws_tx, _) = broadcast::channel(BROADCAST_CAPACITY);

    // Create job store and load from DB
    let db = Arc::new(db);
    let store = Arc::new(JobStore::new(config.clone(), Some(db), ws_tx.clone()));
    let loaded = store.load_from_db().await?;

    if loaded > 0 {
        info!(loaded = loaded, "Jobs loaded from DB");
    }

    let capabilities = workers.capabilities.clone();
    let infer_tasks = workers.infer_tasks.clone();
    let engine_versions = workers.engine_versions.clone();
    let test_echo_mode = workers.test_echo_mode;
    let pool = workers.pool.clone();

    // Initialize utterance cache (SQLite, shared with Python workers)
    // Must be before auto-resume so spawn_job can access it.
    let cache = Arc::new(
        UtteranceCache::tiered(cache_dir, None)
            .await
            .map_err(|e| error::ServerError::Validation(format!("cache init failed: {e}")))?,
    );

    let queued = store.queued_job_ids().await;
    if !queued.is_empty() {
        info!(
            count = queued.len(),
            "Queued jobs will be resumed by local dispatcher"
        );
    }

    let bug_reports_dir = layout.bug_reports_dir().to_string_lossy().into_owned();
    let dashboard_dir = crate::routes::dashboard::find_dashboard_dir_for(
        &layout,
        std::env::var("BATCHALIGN_DASHBOARD_DIR").ok().as_deref(),
    );

    let queue_notify = Arc::new(tokio::sync::Notify::new());
    let queue: Arc<dyn QueueBackend> =
        Arc::new(LocalQueueBackend::new(store.clone(), queue_notify));
    let runtime = RuntimeSupervisor::new();
    let runner_context = RunnerContext::new(
        store.clone(),
        pool.clone(),
        cache.clone(),
        infer_tasks.clone(),
        engine_versions.clone(),
        test_echo_mode,
        queue.clone(),
    );
    let dispatcher = QueueDispatcher::new(queue.clone(), runtime.clone(), runner_context);

    let state = Arc::new(AppState {
        control: AppControlPlane {
            store,
            queue: queue.clone(),
            runtime,
            ws_tx,
        },
        workers: WorkerSubsystem {
            pool,
            capabilities,
            infer_tasks,
        },
        environment: AppEnvironment {
            config,
            media: MediaResolver::new(),
            paths: AppPaths {
                jobs_dir,
                bug_reports_dir,
                dashboard_dir,
            },
        },
        build: AppBuildInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            build_hash: build_hash.unwrap_or_default(),
        },
    });

    state.control.runtime.start_queue_task(dispatcher.run());
    state.control.queue.notify();

    let router = crate::routes::router(state.clone());
    Ok((router, state))
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

/// Start serving on the configured host:port with graceful shutdown.
///
/// Listens for SIGINT and SIGTERM. On signal:
/// 1. Stops accepting new connections
/// 2. Waits for in-flight requests to complete
/// 3. Shuts down the worker pool (SIGTERM → wait → SIGKILL)
pub async fn serve(
    config: ServerConfig,
    pool_config: PoolConfig,
    build_hash: Option<String>,
) -> Result<(), error::ServerError> {
    let layout = RuntimeLayout::from_env();
    serve_with_runtime(config, pool_config, layout, build_hash).await
}

/// Start serving with an explicit runtime layout for state-owned paths.
pub async fn serve_with_runtime(
    config: ServerConfig,
    pool_config: PoolConfig,
    layout: RuntimeLayout,
    build_hash: Option<String>,
) -> Result<(), error::ServerError> {
    let host = config.host.clone();
    let port = config.port;

    let (router, state) =
        create_app_with_runtime(config, pool_config, layout, None, None, build_hash).await?;

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(error::ServerError::Io)?;
    info!(addr = %addr, "Server listening");

    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .map_err(error::ServerError::Io)?;

    info!("Server stopped, shutting down gracefully");

    // 1. Cancel all active jobs
    let cancelled = state.control.store.cancel_all().await;
    if cancelled > 0 {
        info!(cancelled, "Cancelled active jobs");
    }

    // 1b. Stop the queue dispatcher and await tracked job tasks.
    let shutdown_summary = state
        .control
        .runtime
        .shutdown(tokio::time::Duration::from_secs(15))
        .await;
    if shutdown_summary.timed_out {
        warn!(
            remaining = shutdown_summary.remaining_jobs,
            "Some job tasks did not finish in time"
        );
    } else if shutdown_summary.remaining_jobs > 0 {
        info!(
            remaining = shutdown_summary.remaining_jobs,
            "Job runtime shut down with remaining tracked jobs"
        );
    }

    // 3. Shut down the worker pool (gracefully shuts down all workers)
    state.workers.pool.shutdown().await;
    info!("Shutdown complete");

    Ok(())
}

/// Wait for a shutdown signal (SIGINT or SIGTERM).
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install CTRL+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => info!("Received SIGINT, shutting down"),
        () = terminate => info!("Received SIGTERM, shutting down"),
    }
}
