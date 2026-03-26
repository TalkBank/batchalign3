//! `WorkerHandle` — manages a single Python worker child process.
//!
//! Split into submodules:
//! - [`config`] — WorkerRuntimeConfig, WorkerConfig
//! - This file — WorkerHandle struct, spawn, IPC, shutdown, tests

pub mod config;

pub use config::{WorkerConfig, WorkerRuntimeConfig};

use std::process::{Command as StdCommand, Stdio};
use std::time::Duration;

use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::target::task_name;
use crate::worker::{
    BatchInferRequest, BatchInferResponse, InferRequest, InferResponse, WorkerCapabilities,
    WorkerHealthResponse, WorkerPid, WorkerProfile, WorkerTarget,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tracing::{debug, info, warn};

use crate::worker::error::WorkerError;
use crate::worker::provider_credentials::HkAsrCredentialSources;

const STARTUP_STDERR_TAIL_CHARS: usize = 2_000;
const MAX_READY_STDOUT_PREAMBLE_LINES: usize = 32;
const MAX_RESPONSE_STDOUT_NOISE_LINES: usize = 8;

/// Ready signal emitted by the Python worker on stdout.
#[derive(Debug, Deserialize)]
struct ReadySignal {
    ready: bool,
    pid: u32,
    transport: Option<String>,
}

/// Internal wire-level request envelope sent to Python.
#[derive(Debug, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum WorkerRequest<'a> {
    Infer { request: &'a InferRequest },
    BatchInfer { request: &'a BatchInferRequest },
    ExecuteV2 { request: &'a ExecuteRequestV2 },
    Health,
    Capabilities,
    Shutdown,
}

/// Internal wire-level response envelope read from Python.
#[derive(Debug, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum WorkerResponse {
    Infer { response: InferResponse },
    BatchInfer { response: BatchInferResponse },
    ExecuteV2 { response: ExecuteResponseV2 },
    Health { response: WorkerHealthResponse },
    Capabilities { response: WorkerCapabilities },
    Shutdown,
    Error { error: String },
}

fn build_worker_command(config: &WorkerConfig) -> StdCommand {
    let mut cmd = StdCommand::new(&config.python_path);
    cmd.arg("-c")
        .arg("import sys; sys.argv = ['batchalign-worker'] + sys.argv[1:]; from batchalign.worker import main; main()")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if config.test_echo {
        cmd.arg("--test-echo");
    }
    match config.bootstrap_target() {
        WorkerTarget::Profile(profile) => {
            cmd.arg("--profile").arg(profile.name());
        }
        WorkerTarget::InferTask(task) => {
            cmd.arg("--task").arg(task_name(task));
        }
    }

    cmd.arg("--lang").arg(config.lang.as_worker_arg());
    cmd.arg("--num-speakers")
        .arg(config.num_speakers.0.to_string());

    if !config.engine_overrides.is_empty() {
        cmd.arg("--engine-overrides").arg(&config.engine_overrides);
    }

    if config.runtime.force_cpu {
        cmd.arg("--force-cpu");
    }

    if config.verbose > 0 {
        cmd.arg("--verbose").arg(config.verbose.to_string());
    }

    if config.profile == WorkerProfile::Gpu {
        cmd.arg("--gpu-thread-pool-size")
            .arg(config.runtime.gpu_thread_pool_size.to_string());
    }

    if config.test_delay_ms > 0 {
        cmd.arg("--test-delay-ms")
            .arg(config.test_delay_ms.to_string());
    }

    if let Some(api_key) = config.runtime.revai_api_key.as_deref() {
        cmd.env("BATCHALIGN_REV_API_KEY", api_key);
    }
    for (key, value) in worker_provider_envs(config, &HkAsrCredentialSources::from_env()) {
        cmd.env(key, value);
    }

    // Each worker becomes its own process group leader so that
    // killpg() in shutdown/Drop kills the worker AND all its children
    // (e.g. Stanza subprocesses).
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;

        unsafe {
            cmd.pre_exec(|| {
                libc::setpgid(0, 0);
                Ok(())
            });
        }
    }

    cmd
}

/// Spawn a **detached** TCP worker daemon.
///
/// Unlike [`WorkerHandle::spawn`] which creates a child process tied directly
/// to the current Rust process, this launches a standalone Python process with
/// `--transport tcp` that:
/// 1. Loads models, binds a TCP port
/// 2. Registers itself in `workers.json`
/// 3. Prints a ready signal to stderr
/// 4. Continues running until explicitly shut down
///
/// If [`WorkerRuntimeConfig::server_instance_id`] is present, the daemon
/// registers itself as **server-owned** and should be retired by that same
/// server instance on shutdown. Otherwise it registers as **external** and may
/// be reused across server restarts.
///
/// Returns `(pid, port)` on success after waiting for the ready signal.
pub async fn spawn_tcp_daemon(config: &WorkerConfig, port: u16) -> Result<(u32, u16), WorkerError> {
    if config.task.is_some() {
        return Err(WorkerError::SpawnFailed(
            "persistent TCP workers only support profile bootstrap targets".into(),
        ));
    }
    // Memory guard — same as WorkerHandle::spawn().
    let _spawn_permit = crate::worker::memory_guard::acquire_spawn_permit(config)
        .await
        .map_err(|e| WorkerError::SpawnFailed(format!("memory guard: {e}")))?;

    let mut cmd = StdCommand::new(&config.python_path);
    cmd.arg("-c")
        .arg("import sys; sys.argv = ['batchalign-worker'] + sys.argv[1:]; from batchalign.worker import main; main()")
        .arg("--transport")
        .arg("tcp")
        .arg("--profile")
        .arg(config.profile.name())
        .arg("--lang")
        .arg(config.lang.as_worker_arg())
        .arg("--num-speakers")
        .arg(config.num_speakers.0.to_string())
        .arg("--host")
        .arg("127.0.0.1");

    if port > 0 {
        cmd.arg("--port").arg(port.to_string());
    }

    if config.test_echo {
        cmd.arg("--test-echo");
    }

    if !config.engine_overrides.is_empty() {
        cmd.arg("--engine-overrides").arg(&config.engine_overrides);
    }

    if config.runtime.force_cpu {
        cmd.arg("--force-cpu");
    }

    if config.verbose > 0 {
        cmd.arg("--verbose").arg(config.verbose.to_string());
    }

    if config.profile == WorkerProfile::Gpu {
        cmd.arg("--gpu-thread-pool-size")
            .arg(config.runtime.gpu_thread_pool_size.to_string());
    }

    if config.test_delay_ms > 0 {
        cmd.arg("--test-delay-ms")
            .arg(config.test_delay_ms.to_string());
    }

    if let Some(api_key) = config.runtime.revai_api_key.as_deref() {
        cmd.env("BATCHALIGN_REV_API_KEY", api_key);
    }
    if let Some(server_instance_id) = config.runtime.server_instance_id.as_deref() {
        cmd.env("BATCHALIGN_SERVER_INSTANCE_ID", server_instance_id);
    }
    if let Some(server_process_id) = config.runtime.server_process_id {
        cmd.env("BATCHALIGN_SERVER_PID", server_process_id.to_string());
    }
    for (key, value) in worker_provider_envs(config, &HkAsrCredentialSources::from_env()) {
        cmd.env(key, value);
    }

    // Detach: stdin from /dev/null, stdout to /dev/null, stderr piped (for ready signal).
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());

    // On Unix, create a new session (setsid) so the worker is fully detached
    // from the server's process group and terminal.
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }

    info!(
        profile = %config.bootstrap_label(),
        lang = %config.lang,
        port = port,
        "Spawning TCP worker daemon"
    );

    let mut child: Command = cmd.into();
    let mut child = child
        .spawn()
        .map_err(|e| WorkerError::SpawnFailed(format!("failed to spawn TCP worker daemon: {e}")))?;

    // Read stderr for the ready signal: {"ready": true, "pid": N, "transport": "tcp", "port": P}
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| WorkerError::SpawnFailed("TCP daemon stderr not captured".into()))?;
    let mut stderr_reader = tokio::io::BufReader::new(stderr);

    let ready = tokio::time::timeout(
        std::time::Duration::from_secs(config.ready_timeout_s),
        read_tcp_ready_signal(&mut stderr_reader),
    )
    .await
    .map_err(|_| WorkerError::ReadyTimeout {
        timeout_s: config.ready_timeout_s,
    })?
    .map_err(|e| WorkerError::ReadyParseFailed(format!("TCP daemon ready failed: {e}")))?;

    // Detach stderr reader — the daemon continues on its own.
    // We intentionally do NOT wait on the child or hold its handle.
    // The process is now a standalone daemon managed by the OS.
    drop(stderr_reader);

    info!(
        profile = %config.bootstrap_label(),
        lang = %config.lang,
        pid = ready.0,
        port = ready.1,
        "TCP worker daemon ready"
    );

    Ok(ready)
}

/// TCP ready signal from stderr: `{"ready": true, "pid": N, "transport": "tcp", "port": P}`.
#[derive(Debug, Deserialize)]
struct TcpReadySignal {
    ready: bool,
    pid: u32,
    #[allow(dead_code)]
    transport: Option<String>,
    port: Option<u16>,
}

/// Read the TCP ready signal from a daemon's stderr.
async fn read_tcp_ready_signal<R: tokio::io::AsyncBufRead + Unpin>(
    reader: &mut R,
) -> Result<(u32, u16), String> {
    let mut line = String::new();
    let mut attempts = 0;
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => return Err("TCP daemon closed stderr without ready signal".into()),
            Ok(_) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }
                if let Ok(signal) = serde_json::from_str::<TcpReadySignal>(trimmed)
                    && signal.ready
                {
                    let port = signal.port.unwrap_or(0);
                    return Ok((signal.pid, port));
                }
                // Not the ready line — might be a log line, skip it.
                attempts += 1;
                if attempts > 100 {
                    return Err("Too many non-ready lines on stderr".into());
                }
            }
            Err(e) => return Err(format!("Failed to read TCP daemon stderr: {e}")),
        }
    }
}

fn worker_provider_envs(
    config: &WorkerConfig,
    sources: &HkAsrCredentialSources,
) -> Vec<(String, String)> {
    // GPU profile includes ASR — inject provider credentials when the profile
    // handles ASR requests or the engine overrides select an HK ASR backend.
    if config.profile != WorkerProfile::Gpu {
        return Vec::new();
    }
    sources
        .provider_envs_for_asr_override(selected_asr_override(&config.engine_overrides).as_deref())
        .into_iter()
        .collect()
}

fn selected_asr_override(engine_overrides: &str) -> Option<String> {
    if engine_overrides.trim().is_empty() {
        return None;
    }
    let parsed = serde_json::from_str::<Value>(engine_overrides).ok()?;
    parsed.get("asr")?.as_str().map(str::to_string)
}

/// Manages a single Python worker child process.
pub struct WorkerHandle {
    config: WorkerConfig,
    child: Child,
    pid: WorkerPid,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    /// Monotonic instant when the last request was dispatched.
    last_activity: tokio::time::Instant,
}

/// Raw parts extracted from a [`WorkerHandle`] via [`WorkerHandle::into_parts`].
///
/// Used by [`SharedGpuWorker`](super::pool::shared_gpu::SharedGpuWorker) to
/// take ownership of the child's stdio channels for concurrent dispatch.
#[allow(dead_code)]
pub(crate) struct WorkerHandleParts {
    /// Worker configuration.
    pub config: WorkerConfig,
    /// The child process (caller must manage lifecycle).
    pub child: Child,
    /// Worker process ID.
    pub pid: WorkerPid,
    /// Child's stdin for writing requests.
    pub stdin: ChildStdin,
    /// Child's stdout for reading responses.
    pub stdout: BufReader<ChildStdout>,
}

impl WorkerHandle {
    /// Spawn a new Python worker and wait for it to become ready.
    pub async fn spawn(config: WorkerConfig) -> Result<Self, WorkerError> {
        // Memory guard: acquire a serialized spawn permit and check available RAM.
        // This prevents the TOCTOU race where N concurrent spawns all see "enough"
        // memory before any model is loaded, then collectively exceed physical RAM.
        // See docs/memory-safety.md for the full crash history and design rationale.
        let startup_reservation = config.startup_reservation_mb();
        let _spawn_permit = crate::worker::memory_guard::acquire_spawn_permit(&config)
            .await
            .map_err(|e| WorkerError::SpawnFailed(format!("memory guard: {e}")))?;

        let mut cmd: Command = build_worker_command(&config).into();

        info!(
            target = %config.bootstrap_label(),
            lang = %config.lang,
            test_echo = config.test_echo,
            force_cpu = config.runtime.force_cpu,
            python = %config.python_path,
            startup_reservation_mb = startup_reservation.0,
            available_memory_mb = crate::worker::memory_guard::available_memory_mb(),
            "Spawning worker (memory guard passed)"
        );

        let mut child = cmd.spawn().map_err(|e| {
            WorkerError::SpawnFailed(format!("failed to spawn {}: {}", config.python_path, e))
        })?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| WorkerError::SpawnFailed("child stdout not captured".into()))?;
        let mut stdout_reader = BufReader::new(stdout);
        let mut stderr_reader = BufReader::new(
            child
                .stderr
                .take()
                .ok_or_else(|| WorkerError::SpawnFailed("child stderr not captured".into()))?,
        );

        let ready = match tokio::time::timeout(
            Duration::from_secs(config.ready_timeout_s),
            Self::read_ready_line(&mut stdout_reader),
        )
        .await
        {
            Ok(Ok(ready)) => ready,
            Ok(Err(error)) => {
                return Err(
                    Self::finalize_startup_failure(&mut child, &mut stderr_reader, error).await,
                );
            }
            Err(_) => {
                return Err(Self::finalize_startup_failure(
                    &mut child,
                    &mut stderr_reader,
                    WorkerError::ReadyTimeout {
                        timeout_s: config.ready_timeout_s,
                    },
                )
                .await);
            }
        };

        if !ready.ready {
            return Err(Self::finalize_startup_failure(
                &mut child,
                &mut stderr_reader,
                WorkerError::ReadyParseFailed(
                    "worker emitted ready line with ready=false".to_string(),
                ),
            )
            .await);
        }

        if let Some(transport) = ready.transport.as_deref()
            && transport != "stdio"
        {
            return Err(Self::finalize_startup_failure(
                &mut child,
                &mut stderr_reader,
                WorkerError::ReadyParseFailed(format!("unexpected worker transport: {transport}")),
            )
            .await);
        }

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| WorkerError::SpawnFailed("child stdin not captured".into()))?;
        let pid = WorkerPid(ready.pid);

        info!(
            target = %config.bootstrap_label(),
            lang = %config.lang,
            pid = %pid,
            "Worker ready"
        );

        // Layer 3: record PID file for orphan reaping.
        super::pool::reaper::record_worker_pid(pid.0);

        let target_label = config.bootstrap_label();
        tokio::spawn(async move {
            let mut line = String::new();
            loop {
                line.clear();
                match stderr_reader.read_line(&mut line).await {
                    Ok(0) => break,
                    Ok(_) => {
                        let trimmed = line.trim_end();
                        if !trimmed.is_empty() {
                            debug!(worker = %target_label, "{}", trimmed);
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        Ok(Self {
            config,
            child,
            pid,
            stdin,
            stdout: stdout_reader,
            last_activity: tokio::time::Instant::now(),
        })
    }

    /// Read and parse the JSON ready signal from the worker's stdout.
    async fn read_ready_line<R: tokio::io::AsyncBufRead + Unpin>(
        reader: &mut R,
    ) -> Result<ReadySignal, WorkerError> {
        let mut line = String::new();
        let mut preamble = Vec::new();
        loop {
            line.clear();
            reader.read_line(&mut line).await.map_err(|e| {
                WorkerError::ReadyParseFailed(format!("failed to read stdout: {e}"))
            })?;

            if line.is_empty() {
                let mut detail = "worker closed stdout without emitting ready signal".to_string();
                if !preamble.is_empty() {
                    detail.push_str("; pre-ready stdout: ");
                    detail.push_str(&preamble.join(" | "));
                }
                return Err(WorkerError::ReadyParseFailed(detail));
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if !trimmed.starts_with('{') {
                preamble.push(trimmed.to_owned());
                if preamble.len() > MAX_READY_STDOUT_PREAMBLE_LINES {
                    let mut detail = format!(
                        "worker emitted more than {MAX_READY_STDOUT_PREAMBLE_LINES} non-JSON line(s) before ready signal"
                    );
                    detail.push_str("; pre-ready stdout: ");
                    detail.push_str(&preamble.join(" | "));
                    return Err(WorkerError::ReadyParseFailed(detail));
                }
                continue;
            }

            return serde_json::from_str::<ReadySignal>(&line).map_err(|e| {
                let mut detail = format!("invalid ready JSON: {e} (line: {line:?})");
                if !preamble.is_empty() {
                    detail.push_str("; pre-ready stdout: ");
                    detail.push_str(&preamble.join(" | "));
                }
                WorkerError::ReadyParseFailed(detail)
            });
        }
    }

    async fn finalize_startup_failure(
        child: &mut Child,
        stderr_reader: &mut BufReader<ChildStderr>,
        error: WorkerError,
    ) -> WorkerError {
        Self::terminate_startup_child(child).await;
        let stderr = Self::drain_startup_stderr(stderr_reader).await;
        Self::augment_startup_error(error, stderr)
    }

    async fn terminate_startup_child(child: &mut Child) {
        #[cfg(unix)]
        {
            if let Some(pid) = child.id() {
                unsafe {
                    libc::killpg(pid as libc::pid_t, libc::SIGTERM);
                }
            }
        }

        let waited = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
        if waited.is_ok() {
            return;
        }

        let _ = child.start_kill();
        let _ = tokio::time::timeout(Duration::from_millis(500), child.wait()).await;
    }

    async fn drain_startup_stderr(stderr_reader: &mut BufReader<ChildStderr>) -> Option<String> {
        let mut stderr = String::new();
        let _ = tokio::time::timeout(
            Duration::from_secs(1),
            stderr_reader.read_to_string(&mut stderr),
        )
        .await;
        Self::compact_stderr(&stderr)
    }

    fn augment_startup_error(error: WorkerError, stderr: Option<String>) -> WorkerError {
        let Some(stderr) = stderr else {
            return error;
        };

        match error {
            WorkerError::SpawnFailed(message) => {
                WorkerError::SpawnFailed(format!("{message}; worker stderr: {stderr}"))
            }
            WorkerError::ReadyParseFailed(message) => {
                WorkerError::ReadyParseFailed(format!("{message}; worker stderr: {stderr}"))
            }
            other => other,
        }
    }

    fn compact_stderr(stderr: &str) -> Option<String> {
        let mut compact = stderr
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" | ");
        if compact.is_empty() {
            return None;
        }

        let chars: Vec<char> = compact.chars().collect();
        if chars.len() > STARTUP_STDERR_TAIL_CHARS {
            let tail = chars[chars.len() - STARTUP_STDERR_TAIL_CHARS..]
                .iter()
                .collect::<String>();
            compact = format!("…{tail}");
        }

        Some(compact)
    }

    async fn write_request(&mut self, request: &WorkerRequest<'_>) -> Result<(), WorkerError> {
        let mut line = serde_json::to_string(request)
            .map_err(|e| WorkerError::Protocol(format!("failed to encode request: {e}")))?;
        line.push('\n');
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_response(&mut self) -> Result<WorkerResponse, WorkerError> {
        let mut skipped_noise_lines = 0usize;

        loop {
            let mut line = String::new();
            let bytes = self.stdout.read_line(&mut line).await?;
            if bytes == 0 {
                let code = self.child.try_wait().ok().flatten().and_then(|s| s.code());
                return Err(WorkerError::ProcessExited { code });
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match serde_json::from_str::<WorkerResponse>(&line) {
                Ok(response) => return Ok(response),
                Err(e) => {
                    if trimmed.starts_with('{') || trimmed.starts_with('[') {
                        return Err(WorkerError::Protocol(format!(
                            "failed to decode response: {e} (line: {line:?})"
                        )));
                    }

                    skipped_noise_lines += 1;
                    warn!(
                        pid = %self.pid,
                        target = %self.config.bootstrap_label(),
                        line = trimmed,
                        skipped_noise_lines,
                        "Ignoring non-protocol stdout while waiting for worker response"
                    );

                    if skipped_noise_lines >= MAX_RESPONSE_STDOUT_NOISE_LINES {
                        return Err(WorkerError::Protocol(format!(
                            "worker emitted too many non-protocol stdout lines while waiting for response; last line: {line:?}"
                        )));
                    }
                }
            }
        }
    }

    /// Check if the worker is healthy.
    pub async fn health_check(&mut self) -> Result<WorkerHealthResponse, WorkerError> {
        self.write_request(&WorkerRequest::Health).await?;

        let response = tokio::time::timeout(Duration::from_secs(10), self.read_response())
            .await
            .map_err(|_| {
                WorkerError::HealthCheckFailed("timeout waiting for health response".into())
            })??;

        let resp = match response {
            WorkerResponse::Health { response } => response,
            WorkerResponse::Error { error } => return Err(WorkerError::HealthCheckFailed(error)),
            other => {
                return Err(WorkerError::HealthCheckFailed(format!(
                    "unexpected response for health: {other:?}"
                )));
            }
        };

        if !resp.status.is_ok() {
            return Err(WorkerError::HealthCheckFailed(format!(
                "status={}",
                resp.status
            )));
        }

        Ok(resp)
    }

    /// Send a single inference request (CHAT-divorced protocol).
    ///
    /// The server owns all CHAT operations; this sends only structured
    /// payloads (words, lang) and receives structured results (mor, gra).
    pub async fn infer(&mut self, request: &InferRequest) -> Result<InferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();

        self.write_request(&WorkerRequest::Infer { request })
            .await?;

        let timeout = Duration::from_secs(120);
        let response = tokio::time::timeout(timeout, self.read_response())
            .await
            .map_err(|_| WorkerError::Protocol("timeout waiting for infer response".into()))??;

        match response {
            WorkerResponse::Infer { response } => Ok(response),
            WorkerResponse::Error { error } => Err(WorkerError::WorkerResponse(error)),
            other => Err(WorkerError::Protocol(format!(
                "unexpected response for infer: {other:?}"
            ))),
        }
    }

    /// Send a batched inference request (multiple items, one model call).
    ///
    /// Pools multiple utterances into a single NLP call for efficiency.
    pub async fn batch_infer(
        &mut self,
        request: &BatchInferRequest,
    ) -> Result<BatchInferResponse, WorkerError> {
        self.last_activity = tokio::time::Instant::now();

        self.write_request(&WorkerRequest::BatchInfer { request })
            .await?;

        // Generous timeout: roughly 5s per item, minimum 120s.
        let timeout_s = (request.items.len() as u64 * 5).max(120);
        let timeout = Duration::from_secs(timeout_s);
        let response = tokio::time::timeout(timeout, self.read_response())
            .await
            .map_err(|_| {
                WorkerError::Protocol(format!(
                    "timeout ({timeout_s}s) waiting for batch_infer response ({} items)",
                    request.items.len()
                ))
            })??;

        match response {
            WorkerResponse::BatchInfer { response } => Ok(response),
            WorkerResponse::Error { error } => Err(WorkerError::WorkerResponse(error)),
            other => Err(WorkerError::Protocol(format!(
                "unexpected response for batch_infer: {other:?}"
            ))),
        }
    }

    /// Send one typed worker-protocol V2 execute request.
    ///
    /// This keeps the live FA migration on the same long-lived worker process
    /// and stdio transport while replacing the request/response payload shape
    /// with the staged V2 contract.
    pub async fn execute_v2(
        &mut self,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        self.last_activity = tokio::time::Instant::now();

        self.write_request(&WorkerRequest::ExecuteV2 { request })
            .await?;

        let timeout_s = request.timeout_seconds_with_config(
            self.config.audio_task_timeout_s,
            self.config.analysis_task_timeout_s,
        );
        let timeout = Duration::from_secs(timeout_s);
        let response = tokio::time::timeout(timeout, self.read_response())
            .await
            .map_err(|_| {
                WorkerError::Protocol(format!(
                    "timeout ({timeout_s}s) waiting for execute_v2 response ({:?})",
                    request.task
                ))
            })??;

        match response {
            WorkerResponse::ExecuteV2 { response } => Ok(response),
            WorkerResponse::Error { error } => Err(WorkerError::WorkerResponse(error)),
            other => Err(WorkerError::Protocol(format!(
                "unexpected response for execute_v2: {other:?}"
            ))),
        }
    }

    /// Query the worker's capabilities.
    pub async fn capabilities(&mut self) -> Result<WorkerCapabilities, WorkerError> {
        self.write_request(&WorkerRequest::Capabilities).await?;

        // Import probes in _capabilities() may load heavy ML libraries (torch,
        // whisper, pyannote) on first invocation. 60s allows for cold imports.
        let response = tokio::time::timeout(Duration::from_secs(60), self.read_response())
            .await
            .map_err(|_| {
                WorkerError::Protocol("timeout waiting for capabilities response".into())
            })??;

        match response {
            WorkerResponse::Capabilities { response } => Ok(response),
            WorkerResponse::Error { error } => Err(WorkerError::WorkerResponse(error)),
            other => Err(WorkerError::Protocol(format!(
                "unexpected response for capabilities: {other:?}"
            ))),
        }
    }

    /// Check if the worker process is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Gracefully shut down the worker in place (shutdown message + SIGTERM to
    /// process group + wait).
    ///
    /// Uses `killpg` to kill the entire process group (the worker + any children
    /// it spawned, e.g. Stanza subprocesses), ensuring no orphans survive.
    pub async fn shutdown_in_place(&mut self) -> Result<(), WorkerError> {
        // Layer 3: remove PID file before killing.
        super::pool::reaper::remove_worker_pid(self.pid.0);

        info!(
            target = %self.config.bootstrap_label(),
            pid = %self.pid,
            "Shutting down worker"
        );

        let _ = self.write_request(&WorkerRequest::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(2), self.read_response()).await;

        #[cfg(unix)]
        {
            let _ = self.child.id().map(|pid| {
                // SAFETY: sending SIGTERM to the worker's process group.
                // The worker was spawned with setpgid(0,0), so its PGID == PID.
                unsafe { libc::killpg(pid as libc::pid_t, libc::SIGTERM) };
            });
        }

        match tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await {
            Ok(Ok(status)) => {
                info!(pid = %self.pid, ?status, "Worker exited gracefully");
            }
            Ok(Err(e)) => {
                warn!(pid = %self.pid, error = %e, "Error waiting for worker");
            }
            Err(_) => {
                warn!(
                    pid = %self.pid,
                    "Worker didn't exit in 5s, killing process group"
                );
                #[cfg(unix)]
                {
                    let _ = self.child.id().map(|pid| {
                        unsafe { libc::killpg(pid as libc::pid_t, libc::SIGKILL) };
                    });
                }
                let _ = self.child.kill().await;
            }
        }

        Ok(())
    }

    /// Gracefully shut down the worker (consuming `self`).
    pub async fn shutdown(mut self) -> Result<(), WorkerError> {
        self.shutdown_in_place().await
    }

    /// The PID of the worker process.
    pub fn pid(&self) -> WorkerPid {
        self.pid
    }

    /// The logical bootstrap target label this worker handles.
    pub fn profile_label(&self) -> String {
        self.config.bootstrap_label()
    }

    /// The language this worker handles.
    pub fn lang(&self) -> &str {
        self.config.lang.as_worker_arg()
    }

    /// The transport this worker uses.
    pub fn transport(&self) -> &'static str {
        "stdio"
    }

    /// Duration since the last request was dispatched.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Reference to this worker's configuration.
    pub(crate) fn config(&self) -> &WorkerConfig {
        &self.config
    }

    /// Consume the handle into its raw parts for concurrent mode setup.
    ///
    /// The returned [`WorkerHandleParts`] owns the child process, stdin, and
    /// stdout. The caller becomes responsible for the child process lifecycle
    /// — the `WorkerHandle::Drop` impl does **not** run.
    pub(crate) fn into_parts(self) -> WorkerHandleParts {
        // Use ManuallyDrop to prevent Drop::drop from killing the child.
        let md = std::mem::ManuallyDrop::new(self);

        // SAFETY: We're moving each field out of a ManuallyDrop wrapper.
        // ManuallyDrop prevents Drop from running. Each field is moved
        // exactly once, so no double-free can occur.
        unsafe {
            WorkerHandleParts {
                config: std::ptr::read(&md.config),
                child: std::ptr::read(&md.child),
                pid: std::ptr::read(&md.pid),
                stdin: std::ptr::read(&md.stdin),
                stdout: std::ptr::read(&md.stdout),
            }
        }
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        // Layer 3: remove PID file on drop (covers panic/unwind paths).
        super::pool::reaper::remove_worker_pid(self.pid.0);

        if self.is_alive() {
            #[cfg(unix)]
            {
                if let Some(pid) = self.child.id() {
                    let pgid = pid as libc::pid_t;
                    // SAFETY: sending SIGTERM to the worker's process group
                    // (PGID == PID via setpgid(0,0) in spawn).
                    unsafe {
                        libc::killpg(pgid, libc::SIGTERM);
                    }
                    // Brief pause to let Python handle SIGTERM. If the worker
                    // is stuck in a C extension (PyTorch, NumPy), SIGTERM may
                    // be ignored — follow up with SIGKILL to prevent zombies
                    // that hold 2-15 GB of RAM.
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    // SAFETY: kill(pid, 0) checks if process still exists.
                    if unsafe { libc::kill(pgid, 0) } == 0 {
                        unsafe {
                            libc::killpg(pgid, libc::SIGKILL);
                        }
                    }
                }
            }
            let _ = self.child.start_kill();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{WorkerConfig, WorkerRuntimeConfig, build_worker_command};
    use crate::api::{LanguageCode3, NumSpeakers, WorkerLanguage};
    use crate::host_memory::HostMemoryRuntimeConfig;
    use crate::worker::provider_credentials::HkAsrCredentialSources;
    use crate::worker::{InferTask, WorkerProfile};

    fn command_args(config: &WorkerConfig) -> Vec<String> {
        build_worker_command(config)
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect()
    }

    fn command_envs(config: &WorkerConfig) -> BTreeMap<String, String> {
        build_worker_command(config)
            .get_envs()
            .filter_map(|(key, value)| {
                value.map(|value| {
                    (
                        key.to_string_lossy().into_owned(),
                        value.to_string_lossy().into_owned(),
                    )
                })
            })
            .collect()
    }

    #[test]
    fn worker_command_forwards_runtime_force_cpu() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 300,
            verbose: 0,
            runtime: WorkerRuntimeConfig::from_sources(
                true,
                None,
                4,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert!(args.iter().any(|arg| arg == "--force-cpu"));
    }

    #[test]
    fn worker_command_uses_task_arg_for_task_bootstrap() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Stanza,
            task: Some(InferTask::Morphosyntax),
            ..Default::default()
        });

        assert!(
            args.windows(2)
                .any(|window| window[0] == "--task" && window[1] == "morphosyntax")
        );
        assert!(!args.iter().any(|arg| arg == "--profile"));
    }

    #[test]
    fn worker_command_forwards_gpu_thread_pool_size() {
        let args = command_args(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            runtime: WorkerRuntimeConfig::from_sources(
                false,
                None,
                7,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert!(
            args.windows(2)
                .any(|window| { window[0] == "--gpu-thread-pool-size" && window[1] == "7" })
        );
    }

    #[test]
    fn worker_command_injects_resolved_revai_key() {
        let envs = command_envs(&WorkerConfig {
            python_path: "python3".to_string(),
            profile: WorkerProfile::Gpu,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 300,
            verbose: 0,
            runtime: WorkerRuntimeConfig::from_sources(
                false,
                Some("  injected-key  ".to_string()),
                4,
                HostMemoryRuntimeConfig::default(),
                crate::types::runtime::MemoryTier::detect(),
            ),
            ..Default::default()
        });

        assert_eq!(
            envs.get("BATCHALIGN_REV_API_KEY").map(String::as_str),
            Some("injected-key")
        );
    }

    #[test]
    fn worker_provider_envs_only_inject_selected_hk_asr_backend() {
        let envs = super::worker_provider_envs(
            &WorkerConfig {
                python_path: "python3".to_string(),
                profile: WorkerProfile::Gpu,
                lang: WorkerLanguage::from(LanguageCode3::yue()),
                num_speakers: NumSpeakers(1),
                engine_overrides: r#"{"asr":"tencent"}"#.to_string(),
                test_echo: false,
                ready_timeout_s: 300,
                verbose: 0,
                runtime: WorkerRuntimeConfig::from_sources(
                    false,
                    None,
                    4,
                    HostMemoryRuntimeConfig::default(),
                    crate::types::runtime::MemoryTier::detect(),
                ),
                ..Default::default()
            },
            &HkAsrCredentialSources::from_sources(
                Some("id"),
                Some("key"),
                Some("ap-guangzhou"),
                Some("bucket"),
                None,
                None,
                None,
                Some("/tmp/unused-home"),
            ),
        )
        .into_iter()
        .collect::<BTreeMap<_, _>>();

        assert_eq!(
            envs.get("BATCHALIGN_TENCENT_ID").map(String::as_str),
            Some("id")
        );
        assert_eq!(
            envs.get("BATCHALIGN_TENCENT_BUCKET").map(String::as_str),
            Some("bucket")
        );
    }
}
