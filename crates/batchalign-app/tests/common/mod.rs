//! Shared live execution fixture support for real-model integration tests.
//!
//! The fixture owns one prepared worker pool on a dedicated background Tokio
//! runtime. Tests can then acquire either a fresh server session or a fresh
//! direct-execution session over that shared warmed backend. This keeps
//! expensive model loads warm across tests while preventing control-plane state
//! from bleeding between sessions.

#![allow(dead_code)]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, mpsc};
use std::thread;
use std::time::Duration;

use batchalign_app::api::{
    FilePayload, FileResult, HealthResponse, JobInfo, JobListItem, JobResultResponse, JobStatus,
    JobSubmission, LanguageSpec, MemoryMb, NumSpeakers, ReleasedCommand,
};
use batchalign_app::config::{RuntimeLayout, ServerConfig};
use batchalign_app::host_memory::MachineMlTestLock;
use batchalign_app::options::CommandOptions;
use batchalign_app::worker::InferTask;
use batchalign_app::worker::pool::PoolConfig;
use batchalign_app::{
    AppState, DirectHost, PreparedWorkers, create_app_with_prepared_workers, prepare_workers,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// Cached worker backend that survives across isolated server and direct sessions.
struct LiveFixtureBackend {
    _machine_lock: MachineMlTestLock,
    prepared_workers: PreparedWorkers,
    session_config: ServerConfig,
}

impl LiveFixtureBackend {
    /// Build the shared worker backend used by live-model tests.
    async fn initialize() -> Result<Self, String> {
        let python_path = resolve_python()
            .ok_or_else(|| "Python 3 with batchalign is not available".to_string())?;
        let session_config = live_fixture_server_config();
        let machine_lock = MachineMlTestLock::acquire("batchalign-app live fixture")
            .map_err(|error| format!("machine-wide ML test lock unavailable: {error}"))?;
        let prepared_workers =
            prepare_workers(&session_config, live_fixture_pool_config(&python_path))
                .await
                .map_err(|error| format!("could not prepare live workers: {error}"))?;
        Ok(Self {
            _machine_lock: machine_lock,
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

/// Prepared direct-execution metadata returned to test code.
#[derive(Clone)]
struct DirectSnapshot {
    prepared_workers: PreparedWorkers,
    infer_tasks: Vec<InferTask>,
}

/// Commands sent from tests into the dedicated fixture thread.
enum FixtureCommand {
    /// Start one isolated server session backed by the shared prepared workers.
    Acquire {
        /// Synchronous reply channel for session metadata or skip reasons.
        reply: mpsc::Sender<Result<SessionSnapshot, String>>,
    },
    /// Return a clone of the shared warmed worker backend for direct execution.
    AcquireDirect {
        /// Synchronous reply channel for prepared workers or skip reasons.
        reply: mpsc::Sender<Result<DirectSnapshot, String>>,
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
    /// Serialize server and direct sessions over one warmed backend.
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

/// Handle to one isolated direct-execution session backed by shared warmed workers.
pub struct LiveDirectSession {
    host: DirectHost,
    state_dir: PathBuf,
    infer_tasks: Vec<InferTask>,
    _runtime_root: tempfile::TempDir,
    _slot: OwnedSemaphorePermit,
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

impl LiveDirectSession {
    /// Acquire one isolated direct-execution session.
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
            tokio::task::spawn_blocking(move || request_direct_snapshot(&bridge_for_request))
                .await
                .expect("live direct fixture acquire task should not panic");
        let snapshot = match snapshot {
            Ok(snapshot) => snapshot,
            Err(message) => {
                eprintln!("SKIP: {message}");
                return None;
            }
        };

        let runtime_root = tempfile::TempDir::new().expect("tempdir");
        let state_dir = runtime_root.path().to_path_buf();
        let host = DirectHost::new(
            live_fixture_server_config(),
            RuntimeLayout::from_state_dir(state_dir.clone()),
            None,
            Some(state_dir.join("cache")),
            &snapshot.prepared_workers,
        )
        .await
        .expect("create live direct host");

        Some(Self {
            host,
            state_dir,
            infer_tasks: snapshot.infer_tasks,
            _runtime_root: runtime_root,
            _slot: slot,
        })
    }

    /// Runtime-owned state directory for this isolated direct session.
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// Return true when the shared warmed worker backend advertises one infer task.
    pub fn has_infer_task(&self, task: InferTask) -> bool {
        self.infer_tasks.contains(&task)
    }

    /// Run one submission inline and return the final job projections.
    pub async fn run_submission(
        &self,
        submission: JobSubmission,
    ) -> (JobInfo, batchalign_app::store::JobDetail) {
        let outcome = self
            .host
            .run_submission(submission)
            .await
            .expect("run direct submission");
        (outcome.info, outcome.detail)
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

/// Acquire one isolated live direct-execution session and skip cleanly when a task is unavailable.
pub async fn require_live_direct(task: InferTask, skip_message: &str) -> Option<LiveDirectSession> {
    let session = LiveDirectSession::acquire().await?;
    if !session.has_infer_task(task) {
        eprintln!("SKIP: {skip_message}");
        return None;
    }
    Some(session)
}

async fn collect_direct_content_results(
    detail: &batchalign_app::store::JobDetail,
) -> Vec<FileResult> {
    if detail.paths_mode {
        return detail
            .results
            .iter()
            .map(|result| FileResult {
                filename: result.filename.clone(),
                content: String::new(),
                content_type: result.content_type,
                error: result.error.clone(),
            })
            .collect();
    }

    let output_dir = detail.staging_dir.as_path().join("output");
    let mut files = Vec::new();
    for result in &detail.results {
        let content = if result.error.is_none() {
            let path = output_dir.join(&*result.filename);
            tokio::fs::read_to_string(&path)
                .await
                .unwrap_or_else(|error| panic!("read direct result {}: {error}", path.display()))
        } else {
            String::new()
        };
        files.push(FileResult {
            filename: result.filename.clone(),
            content,
            content_type: result.content_type,
            error: result.error.clone(),
        });
    }
    files
}

/// Submit one content-mode job to a live server and return the completed results.
pub async fn submit_and_complete(
    client: &reqwest::Client,
    base_url: &str,
    command: ReleasedCommand,
    lang: &str,
    files: Vec<FilePayload>,
    options: CommandOptions,
) -> (JobInfo, Vec<FileResult>) {
    let submission = JobSubmission {
        command,
        lang: LanguageSpec::try_from(lang)
            .expect("test lang must be a valid ISO 639-3 code or \"auto\""),
        num_speakers: NumSpeakers(1),
        files,
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
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

/// Submit one content-mode job to a live direct session and return the completed results.
pub async fn submit_and_complete_direct(
    session: &LiveDirectSession,
    command: ReleasedCommand,
    lang: &str,
    files: Vec<FilePayload>,
    options: CommandOptions,
) -> (JobInfo, Vec<FileResult>) {
    let submission = JobSubmission {
        command,
        lang: LanguageSpec::try_from(lang)
            .expect("test lang must be a valid ISO 639-3 code or \"auto\""),
        num_speakers: NumSpeakers(1),
        files,
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options,
        paths_mode: false,
        source_paths: vec![],
        output_paths: vec![],
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };

    let (info, detail) = session.run_submission(submission).await;
    let results = collect_direct_content_results(&detail).await;
    (info, results)
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
            FixtureCommand::AcquireDirect { reply } => {
                if active_session.is_some() {
                    let _ = reply.send(Err(
                        "live fixture server session already active while acquiring a direct session"
                            .to_string(),
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

                let snapshot = DirectSnapshot {
                    prepared_workers: backend.prepared_workers.clone(),
                    infer_tasks: backend
                        .prepared_workers
                        .current_infer_tasks()
                        .unwrap_or_else(|_| backend.prepared_workers.infer_tasks().to_vec()),
                };
                let _ = reply.send(Ok(snapshot));
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
    match state.shutdown_for_reuse(Duration::from_secs(5)).await {
        Ok(shutdown) if shutdown.timed_out || shutdown.remaining_jobs > 0 => {
            eprintln!(
                "WARN: live fixture shutdown left {} tracked jobs (timed_out={})",
                shutdown.remaining_jobs, shutdown.timed_out
            );
        }
        Ok(_) => {}
        Err(error) => {
            eprintln!("WARN: live fixture shutdown failed to report runtime status: {error}");
        }
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

/// Request one direct-execution snapshot from the fixture thread.
fn request_direct_snapshot(bridge: &Arc<FixtureBridge>) -> Result<DirectSnapshot, String> {
    let (reply_tx, reply_rx) = mpsc::channel();
    bridge
        .commands
        .send(FixtureCommand::AcquireDirect { reply: reply_tx })
        .map_err(|error| format!("live direct fixture acquire send failed: {error}"))?;
    reply_rx
        .recv()
        .map_err(|error| format!("live direct fixture acquire recv failed: {error}"))?
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

// ---------------------------------------------------------------------------
// Audio fixture and paths-mode helpers
// ---------------------------------------------------------------------------

/// Resolved paths to audio test fixtures copied into a session temp directory.
pub struct AudioFixtures {
    /// Path to the test MP3 audio file.
    pub audio: PathBuf,
    /// Path to the fully-annotated test CHAT file (with %mor/%gra/%wor).
    pub chat: PathBuf,
    /// Path to a stripped CHAT file (main tiers + headers only, no %mor/%gra/%wor).
    pub stripped_chat: PathBuf,
}

/// Locate the committed audio fixtures and copy them into `session_dir`.
///
/// Returns `None` if source fixtures are missing (tests should skip).
pub fn prepare_audio_fixtures(session_dir: &Path) -> Option<AudioFixtures> {
    let repo_root = find_repo_root()?;
    let source_mp3 = repo_root.join("batchalign/tests/support/test.mp3");
    let source_cha = repo_root.join("batchalign/tests/formats/chat/support/test.cha");

    if !source_mp3.exists() || !source_cha.exists() {
        eprintln!(
            "SKIP: audio fixtures not found (expected {}, {})",
            source_mp3.display(),
            source_cha.display()
        );
        return None;
    }

    // Full CHAT (with tiers) — for benchmark gold input.
    let dest_cha = session_dir.join("test.cha");
    std::fs::copy(&source_cha, &dest_cha).expect("copy test.cha");

    // For align: stripped CHAT and audio must be colocated with matching names.
    // @Media says "test, audio" → server looks for test.mp3 next to the CHAT file.
    // Put both in a subdirectory so the stripped file can be named "test.cha".
    let align_dir = session_dir.join("align_input");
    std::fs::create_dir_all(&align_dir).expect("mkdir align_input");
    let dest_mp3 = align_dir.join("test.mp3");
    let dest_stripped = align_dir.join("test.cha");

    std::fs::copy(&source_mp3, &dest_mp3).expect("copy test.mp3");
    let stripped =
        strip_dependent_tiers(&std::fs::read_to_string(&source_cha).expect("read test.cha"));
    std::fs::write(&dest_stripped, &stripped).expect("write stripped test.cha");

    // Also copy audio to session root for transcribe tests (no CHAT needed).
    let transcribe_mp3 = session_dir.join("test.mp3");
    std::fs::copy(&source_mp3, &transcribe_mp3).expect("copy test.mp3 for transcribe");

    Some(AudioFixtures {
        audio: transcribe_mp3,
        chat: dest_cha,
        stripped_chat: dest_stripped,
    })
}

/// Strip %mor, %gra, and %wor dependent tiers from CHAT text.
///
/// Keeps main tiers (*SPEAKER), headers (@), and other dependent tiers intact.
///
/// This uses line-level filtering rather than AST round-tripping because it is
/// a test fixture preparation helper — it creates stripped input for align tests,
/// not a semantic CHAT transformation. The line-prefix check is trivial and
/// correct for well-formed CHAT files from the test fixtures.
pub fn strip_dependent_tiers(chat: &str) -> String {
    let mut result = String::new();
    for line in chat.lines() {
        if line.starts_with("%mor:") || line.starts_with("%gra:") || line.starts_with("%wor:") {
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

/// Walk upward from cwd to find the repo root (directory containing `Cargo.toml`
/// with `[workspace]` or the `batchalign/` directory).
fn find_repo_root() -> Option<PathBuf> {
    let mut cursor = std::env::current_dir().ok()?;
    loop {
        if cursor.join("batchalign").is_dir() && cursor.join("Cargo.toml").is_file() {
            return Some(cursor);
        }
        if !cursor.pop() {
            return None;
        }
    }
}

/// Submit a paths-mode job to a live server and return the completed job info
/// plus the content of the output files.
///
/// `source_paths` and `output_paths` are absolute filesystem paths. The server
/// reads input from `source_paths` and writes results to `output_paths`.
pub async fn submit_paths_and_complete(
    client: &reqwest::Client,
    base_url: &str,
    command: ReleasedCommand,
    lang: &str,
    source_paths: Vec<String>,
    output_paths: Vec<String>,
    options: CommandOptions,
) -> (JobInfo, Vec<String>) {
    assert_eq!(
        source_paths.len(),
        output_paths.len(),
        "source_paths and output_paths must have equal length"
    );

    let submission = JobSubmission {
        command,
        lang: LanguageSpec::try_from(lang)
            .expect("test lang must be a valid ISO 639-3 code or \"auto\""),
        num_speakers: NumSpeakers(1),
        files: vec![],
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options,
        paths_mode: true,
        source_paths: source_paths.iter().map(|s| s.as_str().into()).collect(),
        output_paths: output_paths.iter().map(|s| s.as_str().into()).collect(),
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
    assert_eq!(
        resp.status(),
        200,
        "Paths-mode job submission should succeed"
    );
    let info: JobInfo = resp.json().await.expect("parse initial JobInfo");

    let final_info = poll_job_done(client, base_url, &info.job_id).await;

    if final_info.status != JobStatus::Completed {
        eprintln!(
            "PATHS JOB FAILED: status={:?}, job_id={}",
            final_info.status, final_info.job_id
        );
        // Try to fetch results for error details.
        if let Ok(resp) = client
            .get(format!("{base_url}/jobs/{}/results", final_info.job_id))
            .send()
            .await
            && let Ok(text) = resp.text().await
        {
            eprintln!("  Results response: {}", &text[..text.len().min(500)]);
        }
    }

    // Read output files from disk (only if job completed).
    // Note: the server's apply_result_filename replaces the output path's filename
    // with the input filename's basename. So output_path="/out/foo.cha" with
    // input "test.cha" writes to "/out/test.cha", not "/out/foo.cha".
    // Tests should assert that exact derived path, not silently scavenge a
    // different artifact from the directory.
    let outputs: Vec<String> = if final_info.status == JobStatus::Completed {
        source_paths
            .iter()
            .zip(output_paths.iter())
            .map(|(source_path, output_path)| {
                let expected_path = expected_paths_mode_result_path(source_path, output_path);

                if let Ok(content) = std::fs::read_to_string(&expected_path) {
                    return content;
                }

                let mut nearby_outputs = Vec::new();
                if let Some(dir) = expected_path.parent()
                    && let Ok(entries) = std::fs::read_dir(dir)
                {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().is_some_and(|e| e == "cha" || e == "csv") {
                            nearby_outputs.push(path.display().to_string());
                        }
                    }
                }

                panic!(
                    "Failed to read expected output file {} for source {} and requested output {}; nearby outputs: {:?}",
                    expected_path.display(),
                    source_path,
                    output_path,
                    nearby_outputs
                )
            })
            .collect()
    } else {
        // Job failed — return empty strings so callers can check status.
        vec![String::new(); output_paths.len()]
    };

    (final_info, outputs)
}

/// Submit a paths-mode job to a live direct session and return the completed job
/// info plus the content of the output files.
pub async fn submit_paths_and_complete_direct(
    session: &LiveDirectSession,
    command: ReleasedCommand,
    lang: &str,
    source_paths: Vec<String>,
    output_paths: Vec<String>,
    options: CommandOptions,
) -> (JobInfo, Vec<String>) {
    assert_eq!(
        source_paths.len(),
        output_paths.len(),
        "source_paths and output_paths must have equal length"
    );

    let submission = JobSubmission {
        command,
        lang: LanguageSpec::try_from(lang)
            .expect("test lang must be a valid ISO 639-3 code or \"auto\""),
        num_speakers: NumSpeakers(1),
        files: vec![],
        media_files: vec![],
        media_mapping: Default::default(),
        media_subdir: Default::default(),
        source_dir: Default::default(),
        options,
        paths_mode: true,
        source_paths: source_paths.iter().map(|s| s.as_str().into()).collect(),
        output_paths: output_paths.iter().map(|s| s.as_str().into()).collect(),
        display_names: vec![],
        debug_traces: false,
        before_paths: vec![],
    };

    let (info, detail) = session.run_submission(submission).await;

    if info.status != JobStatus::Completed {
        eprintln!(
            "DIRECT PATHS JOB FAILED: status={:?}, job_id={}",
            info.status, info.job_id
        );
        eprintln!("  File results: {}", detail.results.len());
    }

    let outputs: Vec<String> = if info.status == JobStatus::Completed {
        source_paths
            .iter()
            .zip(output_paths.iter())
            .map(|(source_path, output_path)| {
                let expected_path = expected_paths_mode_result_path(source_path, output_path);

                if let Ok(content) = std::fs::read_to_string(&expected_path) {
                    return content;
                }

                let mut nearby_outputs = Vec::new();
                if let Some(dir) = expected_path.parent()
                    && let Ok(entries) = std::fs::read_dir(dir)
                {
                    for entry in entries.flatten() {
                        let path = entry.path();
                        if path.extension().is_some_and(|e| e == "cha" || e == "csv") {
                            nearby_outputs.push(path.display().to_string());
                        }
                    }
                }

                panic!(
                    "Failed to read expected direct output file {} for source {} and requested output {}; nearby outputs: {:?}",
                    expected_path.display(),
                    source_path,
                    output_path,
                    nearby_outputs
                )
            })
            .collect()
    } else {
        vec![String::new(); output_paths.len()]
    };

    (info, outputs)
}

fn expected_paths_mode_result_path(source_path: &str, output_path: &str) -> PathBuf {
    let source = PathBuf::from(source_path);
    let source_stem = source.file_stem().unwrap_or_else(|| {
        panic!("source path has no filename stem for paths-mode output derivation: {source_path}")
    });
    let source_name = source.file_name().unwrap_or_else(|| {
        panic!("source path has no filename for paths-mode output derivation: {source_path}")
    });

    let requested_output = PathBuf::from(output_path);
    let expected_name = match requested_output.extension() {
        Some(ext) => {
            let mut filename = source_stem.to_os_string();
            filename.push(".");
            filename.push(ext);
            filename
        }
        None => source_name.to_os_string(),
    };

    requested_output
        .parent()
        .map(|dir| dir.join(&expected_name))
        .unwrap_or_else(|| expected_name.into())
}

#[cfg(test)]
mod tests {
    use super::expected_paths_mode_result_path;
    use std::path::PathBuf;

    #[test]
    fn expected_paths_mode_result_path_preserves_requested_extension() {
        let expected = expected_paths_mode_result_path(
            "/tmp/input/eng_acr_first13p5.mp3",
            "/tmp/out/test.cha",
        );
        assert_eq!(expected, PathBuf::from("/tmp/out/eng_acr_first13p5.cha"));
    }

    #[test]
    fn expected_paths_mode_result_path_keeps_source_name_without_output_extension() {
        let expected =
            expected_paths_mode_result_path("/tmp/input/eng_acr_first13p5.mp3", "/tmp/out");
        assert_eq!(expected, PathBuf::from("/tmp/eng_acr_first13p5.mp3"));
    }
}

/// Read the Rev.AI API key from environment variables.
///
/// Checks `REVAI_API_KEY` first, then `BATCHALIGN_REV_API_KEY`.
pub fn require_revai_key() -> Option<String> {
    std::env::var("REVAI_API_KEY")
        .ok()
        .or_else(|| std::env::var("BATCHALIGN_REV_API_KEY").ok())
        .filter(|k| !k.is_empty())
}

// ---------------------------------------------------------------------------
// BA2 parity helpers
// ---------------------------------------------------------------------------

/// Load a CHAT fixture from `batchalign/tests/support/parity/{name}.cha`.
///
/// Returns `None` if the fixture file doesn't exist (test should skip).
pub fn load_parity_fixture(name: &str) -> Option<String> {
    let repo_root = find_repo_root()?;
    let path = repo_root.join(format!("batchalign/tests/support/parity/{name}.cha"));
    if !path.exists() {
        eprintln!("SKIP: parity fixture not found: {}", path.display());
        return None;
    }
    Some(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read parity fixture {}: {e}", path.display())),
    )
}

/// Load a BA2 Jan 9 golden reference output.
///
/// Reads from `batchalign/tests/golden/ba2_reference/{command}/{name}.jan9.cha`.
/// Returns `None` if not yet generated (parity test should still run with
/// structural assertions only).
pub fn load_ba2_golden(command: &str, name: &str) -> Option<String> {
    let repo_root = find_repo_root()?;
    let path = repo_root.join(format!(
        "batchalign/tests/golden/ba2_reference/{command}/{name}.jan9.cha"
    ));
    if !path.exists() {
        eprintln!(
            "NOTE: BA2 golden not found (run scripts/generate_ba2_golden.sh): {}",
            path.display()
        );
        return None;
    }
    Some(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read BA2 golden {}: {e}", path.display())),
    )
}

/// Load a compare fixture pair (`FILE.cha` plus `FILE.gold.cha`).
///
/// Returns `None` if either companion file does not exist.
pub fn load_compare_fixture_pair(name: &str) -> Option<(String, String)> {
    let repo_root = find_repo_root()?;
    let main_path = repo_root.join(format!("batchalign/tests/support/parity/{name}.cha"));
    let gold_path = repo_root.join(format!("batchalign/tests/support/parity/{name}.gold.cha"));
    if !main_path.exists() || !gold_path.exists() {
        eprintln!(
            "SKIP: compare fixture pair not found: {} / {}",
            main_path.display(),
            gold_path.display()
        );
        return None;
    }
    let main = std::fs::read_to_string(&main_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read compare fixture {}: {e}",
            main_path.display()
        )
    });
    let gold = std::fs::read_to_string(&gold_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read compare fixture {}: {e}",
            gold_path.display()
        )
    });
    Some((main, gold))
}

/// Load committed batchalign2-master compare golden outputs.
///
/// Reads from `batchalign/tests/golden/ba2_reference/compare/{name}.master.cha`
/// plus the companion `.master.compare.csv`.
pub fn load_ba2_compare_master_golden(name: &str) -> Option<(String, String)> {
    let repo_root = find_repo_root()?;
    let chat_path = repo_root.join(format!(
        "batchalign/tests/golden/ba2_reference/compare/{name}.master.cha"
    ));
    let csv_path = repo_root.join(format!(
        "batchalign/tests/golden/ba2_reference/compare/{name}.master.compare.csv"
    ));
    if !chat_path.exists() || !csv_path.exists() {
        eprintln!(
            "SKIP: BA2 master compare golden not found: {} / {}",
            chat_path.display(),
            csv_path.display()
        );
        return None;
    }
    let chat = std::fs::read_to_string(&chat_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read BA2 master compare golden {}: {e}",
            chat_path.display()
        )
    });
    let csv = std::fs::read_to_string(&csv_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read BA2 master compare golden {}: {e}",
            csv_path.display()
        )
    });
    Some((chat, csv))
}

/// Compare BA3 output to BA2 golden reference, ignoring metadata differences.
///
/// Filters out lines that are expected to differ between BA2 and BA3
/// (timestamps, PID headers, tool-specific comments) and compares the
/// remaining content line-by-line.
///
/// This deliberately uses line-level text comparison rather than AST diffing
/// because the purpose is to verify textual parity with batchalign2-master
/// output. AST structural equality would miss formatting and ordering
/// differences that matter for CHAT file compatibility.
///
/// Panics with a detailed diff on mismatch.
pub fn assert_ba2_parity(label: &str, ba3_output: &str, ba2_golden: &str) {
    let normalize = |s: &str| -> Vec<String> {
        s.lines()
            .filter(|line| {
                // Skip lines that naturally differ between BA2 and BA3
                !line.starts_with("@PID:")
                    && !line.starts_with("@Date:")
                    && !line.starts_with("@Comment:\t@Languages")
                    && !line.starts_with("@Tape Location:")
                    && !line.starts_with("@New Episode")
                    && !line.starts_with("@Situation:")
                    // Participant/ID ordering may differ — skip for comparison
                    && !line.starts_with("@Participants:")
                    && !line.starts_with("@ID:")
            })
            .map(|line| {
                let mut l = line.trim_end().to_string();
                // Normalize %gra ROOT convention: BA2 uses N|ROOT (self-ref),
                // BA3 uses 0|ROOT (UD standard). Convert BA2 to BA3 convention.
                if l.starts_with("%gra:") {
                    l = normalize_gra_root(&l);
                }
                l
            })
            .collect()
    };

    let ba3_lines = normalize(ba3_output);
    let ba2_lines = normalize(ba2_golden);

    if ba3_lines == ba2_lines {
        return;
    }

    // Build a useful diff report
    let mut report = format!("\n=== BA2 PARITY FAILURE: {label} ===\n\n");

    let max_lines = ba3_lines.len().max(ba2_lines.len());
    let mut diff_count = 0;
    for i in 0..max_lines {
        let ba3_line = ba3_lines.get(i).map(|s| s.as_str()).unwrap_or("<missing>");
        let ba2_line = ba2_lines.get(i).map(|s| s.as_str()).unwrap_or("<missing>");

        if ba3_line != ba2_line {
            diff_count += 1;
            report.push_str(&format!(
                "Line {i}:\n  BA2: {ba2_line}\n  BA3: {ba3_line}\n\n"
            ));
            if diff_count >= 20 {
                report.push_str("  ... (truncated, too many diffs)\n");
                break;
            }
        }
    }

    report.push_str(&format!(
        "Total lines: BA2={}, BA3={}, diffs={diff_count}\n",
        ba2_lines.len(),
        ba3_lines.len(),
    ));

    panic!("{report}");
}

/// Compare two text artifacts exactly after normalizing line endings and
/// trimming line-end whitespace.
pub fn assert_exact_text_parity(label: &str, actual: &str, expected: &str) {
    let normalize =
        |s: &str| -> Vec<String> { s.lines().map(|line| line.trim_end().to_string()).collect() };

    let actual_lines = normalize(actual);
    let expected_lines = normalize(expected);
    if actual_lines == expected_lines {
        return;
    }

    let mut report = format!("\n=== EXACT PARITY FAILURE: {label} ===\n\n");
    let max_lines = actual_lines.len().max(expected_lines.len());
    let mut diff_count = 0;
    for i in 0..max_lines {
        let actual_line = actual_lines
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or("<missing>");
        let expected_line = expected_lines
            .get(i)
            .map(|s| s.as_str())
            .unwrap_or("<missing>");
        if actual_line != expected_line {
            diff_count += 1;
            report.push_str(&format!(
                "Line {i}:\n  expected: {expected_line}\n  actual:   {actual_line}\n\n"
            ));
            if diff_count >= 20 {
                report.push_str("  ... (truncated, too many diffs)\n");
                break;
            }
        }
    }

    report.push_str(&format!(
        "Total lines: expected={}, actual={}, diffs={diff_count}\n",
        expected_lines.len(),
        actual_lines.len(),
    ));
    panic!("{report}");
}

/// Prepare multi-speaker audio fixtures (eng_multi_speaker.mp3).
///
/// Returns `None` if the multi-speaker clip isn't committed.
pub fn prepare_multi_speaker_audio(session_dir: &Path) -> Option<AudioFixtures> {
    let repo_root = find_repo_root()?;
    let source_mp3 = repo_root.join("batchalign/tests/support/eng_multi_speaker.mp3");
    let source_cha = repo_root.join("batchalign/tests/support/parity/eng_multi_speaker.cha");

    if !source_mp3.exists() || !source_cha.exists() {
        eprintln!(
            "SKIP: multi-speaker fixtures not found ({}, {})",
            source_mp3.display(),
            source_cha.display()
        );
        return None;
    }

    let dest_mp3 = session_dir.join("eng_multi_speaker.mp3");
    let dest_cha = session_dir.join("eng_multi_speaker.cha");
    let dest_stripped = session_dir.join("eng_multi_speaker_stripped.cha");

    std::fs::copy(&source_mp3, &dest_mp3).expect("copy eng_multi_speaker.mp3");
    std::fs::copy(&source_cha, &dest_cha).expect("copy eng_multi_speaker.cha");

    let stripped = strip_dependent_tiers(
        &std::fs::read_to_string(&source_cha).expect("read eng_multi_speaker.cha"),
    );
    std::fs::write(&dest_stripped, &stripped).expect("write stripped eng_multi_speaker.cha");

    Some(AudioFixtures {
        audio: dest_mp3,
        chat: dest_cha,
        stripped_chat: dest_stripped,
    })
}

/// Normalize %gra ROOT convention: convert BA2's self-referencing ROOT
/// (e.g., `4|7|ROOT` where 7 is itself) to BA3's UD-standard `4|0|ROOT`.
fn normalize_gra_root(gra_line: &str) -> String {
    // %gra lines contain space-separated items like "1|2|NSUBJ 2|0|ROOT 3|2|PUNCT"
    // Find the ROOT item and set its head to 0.
    let prefix = if gra_line.starts_with("%gra:\t") {
        "%gra:\t"
    } else if gra_line.starts_with("%gra:") {
        "%gra:"
    } else {
        return gra_line.to_string();
    };

    let items: Vec<&str> = gra_line[prefix.len()..].split(' ').collect();
    let normalized: Vec<String> = items
        .iter()
        .map(|item| {
            if item.ends_with("|ROOT") {
                // Replace N|M|ROOT with N|0|ROOT
                let parts: Vec<&str> = item.splitn(3, '|').collect();
                if parts.len() == 3 {
                    format!("{}|0|ROOT", parts[0])
                } else {
                    item.to_string()
                }
            } else {
                item.to_string()
            }
        })
        .collect();

    format!("{prefix}{}", normalized.join(" "))
}

/// Prepare a named audio clip from `batchalign/tests/support/{name}.mp3`.
///
/// Copies the clip to `session_dir` and optionally pairs it with a matching
/// timed CHAT fixture from `batchalign/tests/support/parity/{chat_name}.cha`.
/// Returns `None` if the audio file doesn't exist.
pub fn prepare_named_audio(
    session_dir: &Path,
    audio_name: &str,
    chat_name: Option<&str>,
) -> Option<AudioFixtures> {
    let repo_root = find_repo_root()?;
    let source_mp3 = repo_root.join(format!("batchalign/tests/support/{audio_name}.mp3"));

    if !source_mp3.exists() {
        eprintln!("SKIP: audio fixture not found: {}", source_mp3.display());
        return None;
    }

    // For transcribe: audio in session root (no CHAT needed, server creates CHAT from scratch).
    let transcribe_mp3 = session_dir.join(format!("{audio_name}.mp3"));
    std::fs::copy(&source_mp3, &transcribe_mp3).expect("copy audio for transcribe");

    // For align: CHAT and audio must be colocated with matching names.
    // The CHAT's @Media header references the audio basename — the server resolves
    // media by looking for that basename (+ .mp3/.wav/.mp4) next to the CHAT file.
    // We create a subdirectory so the stripped CHAT can share the audio name.
    let (dest_cha, dest_stripped) = if let Some(cn) = chat_name {
        let source_cha = repo_root.join(format!("batchalign/tests/support/parity/{cn}.cha"));
        if !source_cha.exists() {
            eprintln!("SKIP: CHAT fixture not found: {}", source_cha.display());
            return None;
        }

        // Read the @Media basename from the CHAT file so we can name the
        // colocated audio to match. This uses a trivial line-prefix scan rather
        // than a full AST parse because it is fixture path setup — extracting a
        // single header value from a known-good test file.
        let chat_content = std::fs::read_to_string(&source_cha).expect("read chat");
        let media_basename = chat_content
            .lines()
            .find(|l| l.starts_with("@Media:"))
            .and_then(|l| l.split('\t').nth(1))
            .and_then(|m| m.split(',').next())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| audio_name.to_string());

        // Create align input dir with colocated audio + CHAT.
        let align_dir = session_dir.join(format!("align_{audio_name}"));
        std::fs::create_dir_all(&align_dir).expect("mkdir align dir");

        // Audio named to match @Media reference.
        let align_mp3 = align_dir.join(format!("{media_basename}.mp3"));
        std::fs::copy(&source_mp3, &align_mp3).expect("copy audio for align");

        // Full CHAT (with original tiers).
        let dest = align_dir.join(format!("{media_basename}.cha"));
        std::fs::copy(&source_cha, &dest).expect("copy chat");

        // Stripped CHAT (no %mor/%gra/%wor) — same name, used as align input.
        let stripped = strip_dependent_tiers(&chat_content);
        let dest_s = align_dir.join(format!("{media_basename}_input.cha"));
        std::fs::write(&dest_s, &stripped).expect("write stripped chat");

        // For align, the input CHAT filename must match @Media basename
        // so the server finds the audio. Rename stripped to match.
        let dest_align_input = align_dir.join(format!("{media_basename}.cha"));
        // Overwrite the full CHAT with stripped version for align input.
        std::fs::write(&dest_align_input, &stripped).expect("write align input");

        (dest, dest_align_input)
    } else {
        // No CHAT — transcribe-only (server creates CHAT from audio).
        let dummy = session_dir.join("dummy.cha");
        (dummy.clone(), dummy)
    };

    Some(AudioFixtures {
        audio: transcribe_mp3,
        chat: dest_cha,
        stripped_chat: dest_stripped,
    })
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

/// Worker-pool config tuned for live-model fixture reuse.
///
/// Key memory safety settings:
/// - `idle_timeout_s: 30` — reap workers from completed task types quickly
///   to prevent memory accumulation when tests cycle through ASR→FA→Speaker→OpenSMILE.
///   On a 64GB machine, keeping all task workers resident simultaneously can OOM.
/// - `max_workers_per_key: 1` — one worker per (task, lang) pair. Tests are
///   serialized via semaphore anyway, so >1 just wastes memory.
fn live_fixture_pool_config(python_path: &str) -> PoolConfig {
    PoolConfig {
        python_path: python_path.into(),
        test_echo: false,
        health_check_interval_s: 3_600,
        idle_timeout_s: 30,
        ready_timeout_s: 120,
        max_workers_per_key: 1,
        verbose: 0,
        engine_overrides: String::new(),
        runtime: Default::default(),
        ..Default::default()
    }
}
