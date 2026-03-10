//! Shared live-server fixture support for real-model integration tests.
//!
//! The fixture owns one prepared worker pool on a dedicated background Tokio
//! runtime and creates a fresh server runtime (HTTP listener, store, SQLite DB,
//! jobs dir, cache dir, runtime supervisor) for each acquired session. This
//! keeps expensive model loads warm across tests while preventing control-plane
//! state from bleeding between sessions.

#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, mpsc};
use std::thread;
use std::time::Duration;

use batchalign_app::api::{
    FilePayload, FileResult, HealthResponse, JobInfo, JobListItem, JobResultResponse, JobStatus,
    JobSubmission, NumSpeakers,
};
use batchalign_app::config::{MemoryMb, RuntimeLayout, ServerConfig};
use batchalign_app::options::CommandOptions;
use batchalign_app::worker::InferTask;
use batchalign_app::worker::pool::PoolConfig;
use batchalign_app::{
    AppState, PreparedWorkers, create_app_with_prepared_workers, prepare_workers,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Cached worker backend that survives across isolated server sessions.
struct LiveFixtureBackend {
    prepared_workers: PreparedWorkers,
    session_config: ServerConfig,
}

impl LiveFixtureBackend {
    /// Build the shared worker backend used by live-model tests.
    async fn initialize() -> Result<Self, String> {
        let python_path = resolve_python()
            .ok_or_else(|| "Python 3 with batchalign is not available".to_string())?;
        let session_config = live_fixture_server_config();
        let prepared_workers =
            prepare_workers(&session_config, live_fixture_pool_config(&python_path))
                .await
                .map_err(|error| format!("could not prepare live workers: {error}"))?;
        Ok(Self {
            prepared_workers,
            session_config,
        })
    }
}

/// Shared worker-backend state for the fixture thread.
enum BackendState {
    /// No backend has been prepared yet.
    Uninitialized,
    /// Prepared workers are ready for reuse.
    Ready(Box<LiveFixtureBackend>),
    /// Backend initialization failed and later callers should skip quickly.
    Unavailable(String),
}

/// One active server session running on the dedicated fixture runtime.
struct ActiveSession {
    state: Arc<AppState>,
    runtime_root: tempfile::TempDir,
    server_task: tokio::task::JoinHandle<()>,
}

/// Immutable session metadata returned to test code.
#[derive(Clone)]
struct SessionSnapshot {
    base_url: String,
    state_dir: PathBuf,
    infer_tasks: Vec<InferTask>,
}

/// Commands sent from tests into the dedicated fixture thread.
enum FixtureCommand {
    /// Start one isolated server session backed by the shared prepared workers.
    Acquire {
        /// Synchronous reply channel for session metadata or skip reasons.
        reply: mpsc::Sender<Result<SessionSnapshot, String>>,
    },
    /// Tear down the currently active isolated session.
    Release {
        /// Synchronous ack channel completed after teardown finishes.
        reply: mpsc::Sender<()>,
    },
}

/// Bridge that lets test threads talk to the fixture thread.
struct FixtureBridge {
    commands: mpsc::Sender<FixtureCommand>,
    session_slots: Arc<Semaphore>,
}

/// Global bridge for the dedicated live-fixture runtime thread.
static LIVE_FIXTURE: LazyLock<Arc<FixtureBridge>> = LazyLock::new(start_fixture_thread);

/// Handle to one isolated live-server session backed by shared warmed workers.
pub struct LiveServerSession {
    base_url: String,
    client: reqwest::Client,
    state_dir: PathBuf,
    infer_tasks: Vec<InferTask>,
    slot: Option<OwnedSemaphorePermit>,
    bridge: Arc<FixtureBridge>,
}

impl LiveServerSession {
    /// Acquire an isolated live-model server session.
    ///
    /// The worker backend is prepared once on a dedicated background runtime.
    /// Each call then creates a fresh runtime layout rooted in a new temp dir so
    /// jobs, SQLite state, cache state, and the runtime supervisor do not bleed
    /// into the next session.
    pub async fn acquire() -> Option<Self> {
        let bridge = LIVE_FIXTURE.clone();
        let slot = bridge
            .session_slots
            .clone()
            .acquire_owned()
            .await
            .expect("live fixture semaphore should stay open");
        let bridge_for_request = bridge.clone();
        let snapshot =
            tokio::task::spawn_blocking(move || request_session_snapshot(&bridge_for_request))
                .await
                .expect("live fixture acquire task should not panic");
        let snapshot = match snapshot {
            Ok(snapshot) => snapshot,
            Err(message) => {
                eprintln!("SKIP: {message}");
                return None;
            }
        };

        Some(Self {
            base_url: snapshot.base_url,
            client: reqwest::Client::new(),
            state_dir: snapshot.state_dir,
            infer_tasks: snapshot.infer_tasks,
            slot: Some(slot),
            bridge,
        })
    }

    /// HTTP base URL for the isolated server instance.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Shared HTTP client for the session.
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Runtime-owned state directory for this isolated session.
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// Return true when the prepared worker backend advertises one infer task.
    pub fn has_infer_task(&self, task: InferTask) -> bool {
        self.infer_tasks.contains(&task)
    }

    /// Read the session's health snapshot.
    pub async fn health(&self) -> HealthResponse {
        self.client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .expect("GET /health")
            .json()
            .await
            .expect("parse health")
    }

    /// List jobs currently visible to this isolated server session.
    pub async fn list_jobs(&self) -> Vec<JobListItem> {
        self.client
            .get(format!("{}/jobs", self.base_url))
            .send()
            .await
            .expect("GET /jobs")
            .json()
            .await
            .expect("parse jobs")
    }

    /// Shut down the isolated session deterministically.
    pub async fn close(mut self) {
        if let Some((bridge, slot)) = self.begin_release() {
            tokio::task::spawn_blocking(move || {
                let _ = release_active_session(&bridge);
                drop(slot);
            })
            .await
            .expect("live fixture release task should not panic");
        }
    }

    /// Take the release inputs so `close()` and `Drop` share one path.
    fn begin_release(&mut self) -> Option<(Arc<FixtureBridge>, OwnedSemaphorePermit)> {
        Some((self.bridge.clone(), self.slot.take()?))
    }
}

impl Drop for LiveServerSession {
    fn drop(&mut self) {
        let Some((bridge, slot)) = self.begin_release() else {
            return;
        };

        thread::spawn(move || {
            let _ = release_active_session(&bridge);
            drop(slot);
        });
    }
}

/// Poll one submitted job until it reaches a terminal state.
pub async fn poll_job_done(client: &reqwest::Client, base_url: &str, job_id: &str) -> JobInfo {
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(300);

    loop {
        let resp = client
            .get(format!("{base_url}/jobs/{job_id}"))
            .send()
            .await
            .expect("GET /jobs/{job_id}");
        let info: JobInfo = resp.json().await.expect("parse job");

        if matches!(
            info.status,
            JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled
        ) {
            return info;
        }

        assert!(
            tokio::time::Instant::now() < deadline,
            "Job {job_id} did not finish within 5 min (status: {:?})",
            info.status
        );

        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }
}

/// Acquire one isolated live-server session and skip cleanly when a task is unavailable.
pub async fn require_live_server(task: InferTask, skip_message: &str) -> Option<LiveServerSession> {
    let server = LiveServerSession::acquire().await?;
    if !server.has_infer_task(task) {
        eprintln!("SKIP: {skip_message}");
        return None;
    }
    Some(server)
}

/// Submit one content-mode job to a live server and return the completed results.
pub async fn submit_and_complete(
    client: &reqwest::Client,
    base_url: &str,
    command: &str,
    lang: &str,
    files: Vec<FilePayload>,
    options: CommandOptions,
) -> (JobInfo, Vec<FileResult>) {
    let submission = JobSubmission {
        command: command.into(),
        lang: lang.into(),
        num_speakers: NumSpeakers(1),
        files,
        media_files: vec![],
        media_mapping: String::new(),
        media_subdir: String::new(),
        source_dir: String::new(),
        options,
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };

    let resp = client
        .post(format!("{base_url}/jobs"))
        .json(&submission)
        .send()
        .await
        .expect("POST /jobs");
    assert_eq!(resp.status(), 200, "Job submission should succeed");
    let info: JobInfo = resp.json().await.expect("parse initial JobInfo");

    let final_info = poll_job_done(client, base_url, &info.job_id).await;

    let results: JobResultResponse = client
        .get(format!("{base_url}/jobs/{}/results", info.job_id))
        .send()
        .await
        .expect("GET results")
        .json()
        .await
        .expect("parse results");

    (final_info, results.files)
}

/// Assert a live-server job completed cleanly without per-file failures.
pub fn assert_completed_without_errors(label: &str, info: &JobInfo, results: &[FileResult]) {
    assert_eq!(
        info.status,
        JobStatus::Completed,
        "{label} should complete successfully; results={results:#?}"
    );
    assert!(
        results.iter().all(|result| result.error.is_none()),
        "{label} should not report per-file errors; results={results:#?}"
    );
}

/// Resolve the Python path for tests. Prefers the project venv over `python3`.
pub fn resolve_python_for_module(module: &str) -> Option<String> {
    if let Ok(dir) = std::env::current_dir() {
        let mut cursor = dir;
        loop {
            for venv in preferred_venv_pythons(&cursor) {
                if venv.exists() && python_imports_module(&venv, module) {
                    return Some(venv.to_string_lossy().to_string());
                }
            }
            if !cursor.pop() {
                break;
            }
        }
    }

    for candidate in preferred_path_pythons() {
        if python_imports_module(candidate, module) {
            return Some((*candidate).to_string());
        }
    }

    None
}

pub fn resolve_python() -> Option<String> {
    resolve_python_for_module("batchalign.worker")
}

fn python_imports_module(command: impl AsRef<std::ffi::OsStr>, module: &str) -> bool {
    let snippet = format!("import {module}");
    std::process::Command::new(command)
        .args(["-c", &snippet])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

fn preferred_venv_pythons(root: &Path) -> Vec<PathBuf> {
    if cfg!(windows) {
        vec![root.join(".venv").join("Scripts").join("python.exe")]
    } else {
        let bin = root.join(".venv").join("bin");
        vec![
            bin.join("python3.12"),
            bin.join("python3"),
            bin.join("python"),
        ]
    }
}

fn preferred_path_pythons() -> &'static [&'static str] {
    if cfg!(windows) {
        &["python"]
    } else {
        &["python3.12", "python3"]
    }
}

/// Start the dedicated fixture thread and return the test-side bridge.
fn start_fixture_thread() -> Arc<FixtureBridge> {
    let (commands, receiver) = mpsc::channel();
    let bridge = Arc::new(FixtureBridge {
        commands,
        session_slots: Arc::new(Semaphore::new(1)),
    });

    thread::Builder::new()
        .name("batchalign-live-fixture".into())
        .spawn(move || run_fixture_thread(receiver))
        .expect("live fixture thread should spawn");

    bridge
}

/// Run the dedicated fixture runtime thread.
fn run_fixture_thread(receiver: mpsc::Receiver<FixtureCommand>) {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("live fixture runtime should build");
    let mut backend = BackendState::Uninitialized;
    let mut active_session: Option<ActiveSession> = None;

    while let Ok(command) = receiver.recv() {
        match command {
            FixtureCommand::Acquire { reply } => {
                if active_session.is_some() {
                    let _ = reply.send(Err(
                        "live fixture session already active while acquiring a new one".to_string(),
                    ));
                    continue;
                }

                let backend = match ensure_backend(&runtime, &mut backend) {
                    Ok(backend) => backend,
                    Err(message) => {
                        let _ = reply.send(Err(message));
                        continue;
                    }
                };

                match runtime.block_on(start_session(backend)) {
                    Ok((session, snapshot)) => {
                        active_session = Some(session);
                        let _ = reply.send(Ok(snapshot));
                    }
                    Err(message) => {
                        let _ = reply.send(Err(message));
                    }
                }
            }
            FixtureCommand::Release { reply } => {
                if let Some(session) = active_session.take() {
                    runtime.block_on(cleanup_session(session));
                }
                let _ = reply.send(());
            }
        }
    }

    if let Some(session) = active_session.take() {
        runtime.block_on(cleanup_session(session));
    }
}

/// Prepare the backend on first use and cache success or failure.
fn ensure_backend<'a>(
    runtime: &tokio::runtime::Runtime,
    state: &'a mut BackendState,
) -> Result<&'a LiveFixtureBackend, String> {
    if matches!(state, BackendState::Uninitialized) {
        *state = match runtime.block_on(LiveFixtureBackend::initialize()) {
            Ok(backend) => BackendState::Ready(Box::new(backend)),
            Err(message) => BackendState::Unavailable(message),
        };
    }

    match state {
        BackendState::Ready(backend) => Ok(backend),
        BackendState::Unavailable(message) => Err(message.clone()),
        BackendState::Uninitialized => unreachable!("backend should be initialized before use"),
    }
}

/// Start one isolated app/server session on the dedicated fixture runtime.
async fn start_session(
    backend: &LiveFixtureBackend,
) -> Result<(ActiveSession, SessionSnapshot), String> {
    let runtime_root = tempfile::TempDir::new().expect("tempdir");
    let layout = RuntimeLayout::from_state_dir(runtime_root.path().to_path_buf());
    let cache_dir = runtime_root.path().join("cache");
    let (router, state) = create_app_with_prepared_workers(
        backend.session_config.clone(),
        layout,
        None,
        None,
        Some(cache_dir),
        Some("live-fixture-hash".into()),
        backend.prepared_workers.clone(),
    )
    .await
    .map_err(|error| format!("Could not create app with live fixture workers: {error}"))?;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    let base_url = format!("http://127.0.0.1:{port}");
    let server_task = tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .ok();
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let snapshot = SessionSnapshot {
        base_url,
        state_dir: runtime_root.path().to_path_buf(),
        infer_tasks: state.infer_tasks().to_vec(),
    };
    let session = ActiveSession {
        state,
        runtime_root,
        server_task,
    };
    Ok((session, snapshot))
}

/// Tear down the active session on the dedicated fixture runtime.
async fn cleanup_session(session: ActiveSession) {
    let ActiveSession {
        state,
        runtime_root,
        server_task,
    } = session;
    server_task.abort();
    let _ = server_task.await;
    let shutdown = state.shutdown_for_reuse(Duration::from_secs(5)).await;
    if shutdown.timed_out || shutdown.remaining_jobs > 0 {
        eprintln!(
            "WARN: live fixture shutdown left {} tracked jobs (timed_out={})",
            shutdown.remaining_jobs, shutdown.timed_out
        );
    }
    drop(state);
    tokio::task::yield_now().await;
    drop(runtime_root);
}

/// Request one session snapshot from the fixture thread.
fn request_session_snapshot(bridge: &Arc<FixtureBridge>) -> Result<SessionSnapshot, String> {
    let (reply_tx, reply_rx) = mpsc::channel();
    bridge
        .commands
        .send(FixtureCommand::Acquire { reply: reply_tx })
        .map_err(|error| format!("live fixture acquire send failed: {error}"))?;
    reply_rx
        .recv()
        .map_err(|error| format!("live fixture acquire recv failed: {error}"))?
}

/// Release the active session and wait for teardown to finish.
fn release_active_session(bridge: &Arc<FixtureBridge>) -> Result<(), String> {
    let (reply_tx, reply_rx) = mpsc::channel();
    bridge
        .commands
        .send(FixtureCommand::Release { reply: reply_tx })
        .map_err(|error| format!("live fixture release send failed: {error}"))?;
    reply_rx
        .recv()
        .map_err(|error| format!("live fixture release recv failed: {error}"))?;
    Ok(())
}

/// Real-model server config for the live fixture.
fn live_fixture_server_config() -> ServerConfig {
    ServerConfig {
        host: "127.0.0.1".into(),
        port: 0,
        job_ttl_days: 1,
        warmup_commands: vec!["morphotag".into()],
        memory_gate_mb: MemoryMb(0),
        ..Default::default()
    }
}

/// Worker-pool config tuned for long-lived live-model fixture reuse.
fn live_fixture_pool_config(python_path: &str) -> PoolConfig {
    PoolConfig {
        python_path: python_path.into(),
        test_echo: false,
        health_check_interval_s: 3_600,
        idle_timeout_s: 3_600,
        ready_timeout_s: 120,
        max_workers_per_key: 2,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
    }
}
