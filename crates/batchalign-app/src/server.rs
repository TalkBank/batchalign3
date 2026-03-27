//! App factory and server lifecycle (create, serve, shutdown).

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use tracing::{info, warn};

use crate::cache::UtteranceCache;
use crate::commands::{released_command_definition, released_command_definitions};
use crate::config::{RuntimeLayout, ServerConfig};
use crate::db::JobDB;
use crate::error;
use crate::host_policy::HostExecutionPolicy;
use crate::media::MediaResolver;
use crate::runner::{ExecutionEngine, RunnerExecutionContext};
use crate::server_backend::{ServerBackendBootstrap, bootstrap_embedded_server_backend};
use crate::state::{
    AppBuildInfo, AppControlPlane, AppEnvironment, AppPaths, AppState, WorkerCapabilitySnapshot,
    WorkerSubsystem, resolve_worker_capability_snapshot, validate_infer_capability_gate,
};
use crate::temporal_backend::bootstrap_temporal_server_backend;
use crate::worker::InferTask;
use crate::worker::pool::{PoolConfig, WorkerPool};

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

/// One host-neutral execution runtime resolved from prepared workers.
pub(crate) struct ResolvedExecutionRuntime {
    pub capability_snapshot: WorkerCapabilitySnapshot,
    pub engine: ExecutionEngine,
}

/// A command + language pair to pre-warm at server startup.
///
/// Replaces the anonymous `(String, String)` tuples that previously flowed
/// from config resolution through `probe_workers` into `WorkerPool::warmup`.
///
/// Both fields are already validated at construction time so downstream
/// consumers do not need to re-parse or handle invalid values.
#[derive(Debug, Clone)]
pub struct WarmupTarget {
    /// Released command to warm (validated from config at construction).
    pub command: crate::api::ReleasedCommand,
    /// Language to warm the command for (validated from config at construction).
    pub lang: crate::api::WorkerLanguage,
}

impl PreparedWorkers {
    /// Resolve the latest capability snapshot, preferring live detected worker
    /// data over the startup placeholder snapshot when available.
    pub(crate) fn capability_snapshot(
        &self,
    ) -> Result<crate::state::WorkerCapabilitySnapshot, error::ServerError> {
        resolve_worker_capability_snapshot(
            &self.capabilities,
            &self.infer_tasks,
            &self.engine_versions,
            self.test_echo_mode,
            self.pool.detected_capabilities(),
        )
    }

    /// Return the released command surface discovered during worker probing.
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }

    /// Return the infer-task set reported by the prepared worker subsystem.
    pub fn infer_tasks(&self) -> &[InferTask] {
        &self.infer_tasks
    }

    /// Return the latest infer-task view, preferring live worker detection
    /// over the startup placeholder snapshot when available.
    pub fn current_infer_tasks(&self) -> Result<Vec<InferTask>, error::ServerError> {
        Ok(self.capability_snapshot()?.infer_tasks)
    }

    /// Build one host-neutral execution runtime over this prepared worker set.
    pub(crate) fn resolve_execution_runtime(
        &self,
        cache: Arc<UtteranceCache>,
    ) -> Result<ResolvedExecutionRuntime, error::ServerError> {
        let capability_snapshot = self.capability_snapshot()?;
        let engine = ExecutionEngine::new(RunnerExecutionContext::new(
            self.pool.clone(),
            cache,
            capability_snapshot.infer_tasks.clone(),
            capability_snapshot.engine_versions.clone(),
            self.test_echo_mode,
        ));
        Ok(ResolvedExecutionRuntime {
            capability_snapshot,
            engine,
        })
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
    let (prepared, targets) = probe_workers(config, pool_config, true).await?;

    if !targets.is_empty() {
        prepared.pool.warmup(&targets).await;
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
    let (prepared, targets) = probe_workers(config, pool_config, true).await?;

    if !targets.is_empty() {
        prepared.pool.mark_warmup_started();
        let warmup_pool = prepared.pool.clone();
        tokio::spawn(async move {
            warmup_pool.warmup(&targets).await;
            warmup_pool.mark_warmup_complete();
            info!("Background warmup complete");
        });
    } else {
        prepared.pool.mark_warmup_complete();
    }

    Ok(prepared)
}

/// Probe and validate one worker pool for direct inline execution.
///
/// Unlike server preparation, this path intentionally skips registry discovery
/// and host-wide warmup so direct mode does not adopt detached daemon behavior.
pub async fn prepare_direct_workers(
    config: &ServerConfig,
    pool_config: PoolConfig,
) -> Result<PreparedWorkers, error::ServerError> {
    let (prepared, _targets) = probe_workers(config, pool_config, false).await?;
    prepared.pool.mark_warmup_complete();
    Ok(prepared)
}

/// Build the worker pool with optimistic capabilities (no Python probe).
///
/// Capabilities are detected lazily on the first real worker spawn, not at
/// server startup. This eliminates the 10-30 second startup delay and 2-3 GB
/// peak memory spike from the probe worker on small machines.
///
/// For test-echo mode, capabilities are synthesized from `cmd2task()`.
async fn probe_workers(
    config: &ServerConfig,
    pool_config: PoolConfig,
    discover_registry_workers: bool,
) -> Result<(PreparedWorkers, Vec<WarmupTarget>), error::ServerError> {
    let test_echo_mode = pool_config.test_echo;
    let host_policy = HostExecutionPolicy::from_server_config(config);
    let pool = Arc::new(WorkerPool::new(pool_config));
    pool.start_background_tasks();

    if discover_registry_workers {
        // Discover pre-started TCP workers from the registry file.
        let discovered = pool.discover_from_registry().await;
        if discovered > 0 {
            info!(discovered, "Pre-started TCP workers integrated into pool");
        }
    }

    // Optimistic capabilities: accept all released commands.
    // Real capabilities are detected lazily on first worker spawn.
    let (capabilities, infer_tasks, engine_versions) = if test_echo_mode {
        let all_tasks: Vec<InferTask> = released_command_definitions()
            .iter()
            .map(|definition| definition.descriptor.infer_task)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let caps = validate_infer_capability_gate(&all_tasks, &BTreeMap::new(), true)?;
        (caps, all_tasks, BTreeMap::new())
    } else if let Some(detected) = pool.detected_capabilities() {
        // TCP registry workers were discovered and capabilities probed.
        let caps = validate_infer_capability_gate(
            &detected.infer_tasks,
            &detected.engine_versions,
            false,
        )?;
        info!(
            capabilities = ?caps,
            infer_tasks = ?detected.infer_tasks,
            "Using capabilities detected from TCP registry workers"
        );
        (
            caps,
            detected.infer_tasks.clone(),
            detected.engine_versions.clone(),
        )
    } else {
        let caps = optimistic_capabilities();
        info!(
            capabilities = ?caps,
            "Using optimistic capabilities (lazy detection on first worker spawn)"
        );
        (caps, Vec::new(), BTreeMap::new())
    };

    // No warmup targets — workers spawn on demand.
    let warmup_cmds = config.resolved_warmup_commands();
    let targets = if warmup_cmds.is_empty() {
        // Skip warmup in production — workers spawn lazily.
        // Test-echo mode can still warmup if configured.
        Vec::new()
    } else {
        let default_lang = crate::api::WorkerLanguage::from(config.default_lang.clone());
        warmup_cmds
            .iter()
            .filter(|cmd| capabilities.contains(cmd))
            .filter_map(|cmd| {
                crate::api::ReleasedCommand::try_from(cmd.as_str())
                    .ok()
                    .and_then(|command| {
                        let definition = released_command_definition(command);
                        host_policy
                            .allows_command_warmup(definition.warmup_policy(), test_echo_mode)
                            .then(|| WarmupTarget {
                                command,
                                lang: default_lang.clone(),
                            })
                    })
            })
            .collect()
    };

    if targets.is_empty() {
        info!("Worker warmup disabled (lazy start)");
    } else {
        info!(commands = ?targets, "Warmup commands resolved");
    }

    Ok((
        PreparedWorkers {
            pool,
            capabilities,
            infer_tasks,
            engine_versions,
            test_echo_mode,
        },
        targets,
    ))
}

/// All released commands — used as the optimistic capability set before
/// the first real worker spawn confirms what's actually installed.
fn optimistic_capabilities() -> Vec<String> {
    released_command_definitions()
        .iter()
        .map(|definition| definition.descriptor.command.to_string())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
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

    // Initialize utterance cache (SQLite, shared with Python workers)
    // Must be before auto-resume so spawn_job can access it.
    let cache = Arc::new(
        UtteranceCache::tiered(cache_dir, None)
            .await
            .map_err(|e| error::ServerError::Validation(format!("cache init failed: {e}")))?,
    );
    let db = Arc::new(db);
    let execution_runtime = workers.resolve_execution_runtime(cache.clone())?;
    let backend_bootstrap: ServerBackendBootstrap = match config.backend {
        crate::config::ServerBackendKind::Embedded => {
            bootstrap_embedded_server_backend(config.clone(), db, execution_runtime.engine).await?
        }
        crate::config::ServerBackendKind::Temporal => {
            bootstrap_temporal_server_backend(config.clone(), db, execution_runtime.engine).await?
        }
    };
    if backend_bootstrap.loaded_jobs > 0 {
        info!(
            loaded = backend_bootstrap.loaded_jobs,
            "Jobs loaded from DB"
        );
    }
    let capability_snapshot = execution_runtime.capability_snapshot;
    let capabilities = capability_snapshot.capabilities;
    let infer_tasks = capability_snapshot.infer_tasks;
    let pool = workers.pool.clone();

    if backend_bootstrap.queued_jobs > 0 {
        info!(
            count = backend_bootstrap.queued_jobs,
            "Queued jobs will be resumed by the configured backend"
        );
    }

    let bug_reports_dir = layout.bug_reports_dir().to_string_lossy().into_owned();
    let dashboard_dir = crate::routes::dashboard::find_dashboard_dir_for(
        &layout,
        std::env::var("BATCHALIGN_DASHBOARD_DIR").ok().as_deref(),
    );

    let state = Arc::new(AppState {
        control: AppControlPlane {
            backend: backend_bootstrap.backend,
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
///
/// Writes a PID file on startup so that `daemon.rs::detect_manual_server()`
/// can discover us, and removes it on exit (including after signal-driven
/// shutdown). This is the single lifecycle owner for foreground servers.
pub async fn serve_with_runtime(
    config: ServerConfig,
    pool_config: PoolConfig,
    layout: RuntimeLayout,
    build_hash: Option<String>,
) -> Result<(), error::ServerError> {
    let host = config.host.clone();
    let port = config.port;

    let (router, state) =
        create_app_with_runtime(config, pool_config, layout.clone(), None, None, build_hash)
            .await?;

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(error::ServerError::Io)?;
    info!(addr = %addr, "Server listening");

    // Write PID file so daemon.rs can discover this foreground server.
    // Best-effort: if the write fails (e.g. read-only filesystem), log
    // and continue -- the server still works, just won't be auto-discovered.
    let pid_path = layout.server_pid_path();
    if let Err(error) = write_pid_file(&pid_path) {
        warn!(path = %pid_path.display(), error = %error,
            "Failed to write server PID file; daemon auto-discovery may not work");
    }

    axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await
    .map_err(error::ServerError::Io)?;

    info!("Server stopped, shutting down gracefully");

    // 1. Cancel all active jobs
    let cancelled = state.control.backend.cancel_all().await;
    if cancelled > 0 {
        info!(cancelled, "Cancelled active jobs");
    }

    // 1b. Stop the queue dispatcher and await tracked job tasks.
    let shutdown_summary = state
        .control
        .backend
        .shutdown_runtime(tokio::time::Duration::from_secs(15))
        .await;
    match shutdown_summary {
        Ok(shutdown_summary) => {
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
        }
        Err(error) => {
            warn!(error = %error, "Runtime supervisor failed to report shutdown status");
        }
    }

    // 3. Shut down the worker pool (gracefully shuts down all workers)
    state.workers.pool.shutdown().await;

    // 4. Remove PID file so stale detection works on next startup.
    remove_pid_file(&pid_path);
    info!("Shutdown complete");

    Ok(())
}

/// Wait for a shutdown signal (SIGINT or SIGTERM).
///
/// If signal handlers fail to install (rare, but possible in constrained
/// environments like containers without a proper init), the server logs the
/// failure and falls through to a pending future that never resolves --
/// meaning the server stays up until the process is killed externally. This
/// is safer than panicking the server on startup.
async fn shutdown_signal() {
    let ctrl_c = async {
        match tokio::signal::ctrl_c().await {
            Ok(()) => {}
            Err(error) => {
                warn!(error = %error, "Failed to install CTRL+C handler; \
                    server will not respond to SIGINT");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => {
                warn!(error = %error, "Failed to install SIGTERM handler; \
                    server will not respond to SIGTERM");
                std::future::pending::<()>().await;
            }
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => info!("Received SIGINT, shutting down"),
        () = terminate => info!("Received SIGTERM, shutting down"),
    }
}

// ---------------------------------------------------------------------------
// PID file helpers
// ---------------------------------------------------------------------------

/// Write the current process PID to a file (atomic via temp + rename).
fn write_pid_file(path: &std::path::Path) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("pid.tmp");
    std::fs::write(&tmp, std::process::id().to_string())?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Remove a PID file. Best-effort: missing file is not an error.
fn remove_pid_file(path: &std::path::Path) {
    match std::fs::remove_file(path) {
        Ok(()) => info!(path = %path.display(), "Removed server PID file"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            warn!(path = %path.display(), error = %error,
                "Failed to remove server PID file");
        }
    }
}
