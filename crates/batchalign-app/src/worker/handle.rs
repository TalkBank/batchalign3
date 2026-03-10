//! `WorkerHandle` — manages a single Python worker child process.
//!
//! Lifecycle: spawn -> wait_ready -> dispatch requests over stdio -> shutdown.
//!
//! # Protocol contract
//!
//! Rust sends newline-delimited JSON requests tagged by `op`; Python replies
//! with matching `op` variants. Any `op` mismatch is treated as a protocol
//! violation and fails the request.
//!
//! Request/response DTOs live in `batchalign_types::worker` and are shared with
//! the Python side (`batchalign/worker/_protocol.py`) to keep schema drift
//! visible.
//!
//! # Failure semantics
//!
//! - Spawn/ready failures are surfaced before handle construction.
//! - Protocol or transport errors return `WorkerError` and let pool policy
//!   decide restart behavior.
//! - Shutdown uses process-group semantics on Unix so worker child processes are
//!   not leaked.

use std::process::{Command as StdCommand, Stdio};
use std::time::Duration;

use crate::api::{LanguageCode3, NumSpeakers};
use crate::revai::load_revai_api_key;
use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::{
    BatchInferRequest, BatchInferResponse, InferRequest, InferResponse, WorkerCapabilities,
    WorkerHealthResponse, WorkerPid, WorkerTarget, infer_task_target_name,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tracing::{debug, info, warn};

use crate::worker::error::WorkerError;
use crate::worker::provider_credentials::HkAsrCredentialSources;
use crate::worker::python::resolve_python_executable;

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

/// Runtime-owned launch inputs for one worker subprocess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRuntimeConfig {
    /// Whether the worker should force CPU-only model/device selection.
    pub force_cpu: bool,
    /// Optional Rev.AI key already resolved by the Rust control plane.
    pub revai_api_key: Option<String>,
}

impl Default for WorkerRuntimeConfig {
    fn default() -> Self {
        Self::from_sources(
            false,
            load_revai_api_key()
                .ok()
                .map(|key| key.as_str().to_string()),
        )
    }
}

impl WorkerRuntimeConfig {
    /// Build worker runtime inputs from explicit sources.
    pub fn from_sources(force_cpu: bool, revai_api_key: Option<String>) -> Self {
        Self {
            force_cpu,
            revai_api_key: revai_api_key
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
        }
    }
}

/// Configuration for spawning a worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Path to the Python executable (e.g. "python3", "/usr/bin/python3.14t").
    pub python_path: String,
    /// Bootstrap target describing which runtime role this worker owns.
    pub target: WorkerTarget,
    /// 3-letter ISO language code.
    pub lang: LanguageCode3,
    /// Number of speakers.
    pub num_speakers: NumSpeakers,
    /// Engine overrides as JSON string (empty = none).
    pub engine_overrides: String,
    /// Use test-echo mode (no ML models).
    pub test_echo: bool,
    /// Maximum seconds to wait for the worker to become ready.
    pub ready_timeout_s: u64,
    /// Verbosity level (0=warn, 1=info, 2=debug, 3+=trace).
    /// Forwarded to the Python worker via `--verbose N` to control its logging
    /// level, enabling end-to-end verbosity from a single CLI `-v` flag.
    pub verbose: u8,
    /// Runtime-owned launch inputs resolved before this spawn boundary.
    pub runtime: WorkerRuntimeConfig,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            python_path: resolve_python_executable(),
            target: WorkerTarget::infer_task(crate::worker::InferTask::Morphosyntax),
            lang: LanguageCode3::from("eng"),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 120,
            verbose: 0,
            runtime: WorkerRuntimeConfig::default(),
        }
    }
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
        let WorkerTarget::InferTask(task) = &config.target;
        cmd.arg("--task").arg(infer_task_target_name(*task));
    } else {
        let WorkerTarget::InferTask(task) = &config.target;
        cmd.arg("--task").arg(infer_task_target_name(*task));
    }

    cmd.arg("--lang").arg(&*config.lang);
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

fn worker_provider_envs(
    config: &WorkerConfig,
    sources: &HkAsrCredentialSources,
) -> Vec<(String, String)> {
    let WorkerTarget::InferTask(task) = config.target;
    if task != crate::worker::InferTask::Asr {
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

impl WorkerHandle {
    /// Spawn a new Python worker and wait for it to become ready.
    pub async fn spawn(config: WorkerConfig) -> Result<Self, WorkerError> {
        let mut cmd: Command = build_worker_command(&config).into();

        info!(
            target = %config.target.label(),
            lang = %config.lang,
            test_echo = config.test_echo,
            force_cpu = config.runtime.force_cpu,
            python = %config.python_path,
            "Spawning worker"
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
            target = %config.target.label(),
            lang = %config.lang,
            pid = %pid,
            "Worker ready"
        );

        let target_label = config.target.label();
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
                        target = %self.config.target.label(),
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

        if resp.status != "ok" {
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

        let timeout_s = request.timeout_seconds();
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
        info!(
            target = %self.config.target.label(),
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
    pub fn target_label(&self) -> String {
        self.config.target.label()
    }

    /// The language this worker handles.
    pub fn lang(&self) -> &str {
        &self.config.lang
    }

    /// The transport this worker uses.
    pub fn transport(&self) -> &'static str {
        "stdio"
    }

    /// Duration since the last request was dispatched.
    pub fn idle_duration(&self) -> Duration {
        self.last_activity.elapsed()
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        if self.is_alive() {
            #[cfg(unix)]
            {
                if let Some(pid) = self.child.id() {
                    // Kill the entire process group (worker + children).
                    // SAFETY: sending SIGTERM then SIGKILL to the worker's
                    // process group (PGID == PID via setpgid in spawn).
                    unsafe {
                        libc::killpg(pid as libc::pid_t, libc::SIGTERM);
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
    use crate::api::{LanguageCode3, NumSpeakers};
    use crate::worker::provider_credentials::HkAsrCredentialSources;
    use crate::worker::{InferTask, WorkerTarget};

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
            target: WorkerTarget::infer_task(InferTask::Asr),
            lang: LanguageCode3::from("eng"),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 120,
            verbose: 0,
            runtime: WorkerRuntimeConfig::from_sources(true, None),
        });

        assert!(args.iter().any(|arg| arg == "--force-cpu"));
    }

    #[test]
    fn worker_command_injects_resolved_revai_key() {
        let envs = command_envs(&WorkerConfig {
            python_path: "python3".to_string(),
            target: WorkerTarget::infer_task(InferTask::Asr),
            lang: LanguageCode3::from("eng"),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 120,
            verbose: 0,
            runtime: WorkerRuntimeConfig::from_sources(false, Some("  injected-key  ".to_string())),
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
                target: WorkerTarget::infer_task(InferTask::Asr),
                lang: LanguageCode3::from("yue"),
                num_speakers: NumSpeakers(1),
                engine_overrides: r#"{"asr":"tencent"}"#.to_string(),
                test_echo: false,
                ready_timeout_s: 120,
                verbose: 0,
                runtime: WorkerRuntimeConfig::from_sources(false, None),
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
