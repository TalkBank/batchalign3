//! `SharedGpuWorker` — concurrent dispatch to a single GPU worker process.
//!
//! Unlike [`CheckedOutWorker`] which grants exclusive access via a semaphore,
//! `SharedGpuWorker` allows multiple concurrent V2 requests to one worker.
//! The Python side runs a `ThreadPoolExecutor` so GPU inference (which releases
//! the GIL) runs in parallel, sharing the same loaded models in-process.
//!
//! ## Response routing
//!
//! A background reader task continuously reads JSON-lines from worker stdout.
//! Each `ExecuteResponseV2` carries a `request_id` that maps back to a pending
//! `oneshot::Sender`. Non-V2 responses (health, capabilities, shutdown) are
//! routed via a separate control channel that serializes sequential ops.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout};
use tokio::sync::oneshot;
use tracing::{debug, error, info, warn};

use crate::types::worker_v2::{ExecuteRequestV2, ExecuteResponseV2};
use crate::worker::WorkerPid;
use crate::worker::error::WorkerError;
use crate::worker::handle::{WorkerConfig, WorkerHandle};

use super::lock_recovered;

/// A GPU worker that supports concurrent V2 request dispatch.
///
/// Created from a [`WorkerHandle`] by consuming its stdio channels and setting
/// up a background response router. The worker process itself runs Python's
/// `_serve_stdio_concurrent()` which dispatches requests to a thread pool.
pub(super) struct SharedGpuWorker {
    /// Owned child process handle. Unlike the TCP shared worker, the stdio
    /// variant is the lifecycle owner and must supervise shutdown/kill.
    child: tokio::sync::Mutex<Option<Child>>,

    /// Serialized writes to worker stdin. Multiple async tasks may send
    /// requests concurrently; the mutex ensures JSON lines don't interleave.
    stdin: tokio::sync::Mutex<ChildStdin>,

    /// Pending V2 requests awaiting responses, keyed by request_id.
    /// Shared with the background reader task via `Arc`.
    pending: Arc<std::sync::Mutex<HashMap<String, oneshot::Sender<ExecuteResponseV2>>>>,

    /// Control channel for sequential non-V2 ops (health, capabilities, shutdown).
    /// Only one sequential op can be in-flight at a time.
    /// Shared with the background reader task via `Arc`.
    #[allow(dead_code)]
    control: Arc<tokio::sync::Mutex<Option<oneshot::Sender<WorkerControlResponse>>>>,

    /// Background stdout reader task handle.
    reader_task: tokio::task::JoinHandle<()>,

    /// Worker process ID.
    pid: WorkerPid,

    /// Worker configuration (for logs and restarts).
    config: WorkerConfig,

    /// Prevents new requests once shutdown starts and serializes lifecycle
    /// teardown across explicit shutdown and Drop.
    shutdown_started: AtomicBool,
}

/// Non-V2 responses routed via the control channel.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum WorkerControlResponse {
    Health(crate::worker::WorkerHealthResponse),
    Capabilities(crate::worker::WorkerCapabilities),
    Shutdown,
    Error(String),
}

impl SharedGpuWorker {
    /// Create a shared GPU worker from an existing [`WorkerHandle`].
    ///
    /// Consumes the handle's stdin/stdout and spawns a background reader task
    /// for response routing. The handle's `Drop` impl is bypassed — this
    /// struct takes ownership of the child process lifecycle.
    pub(super) async fn from_handle(handle: WorkerHandle) -> Self {
        let pid = handle.pid();
        let config = handle.config().clone();

        // Decompose the handle into its parts. We use into_parts() to bypass
        // the Drop impl (which would kill the child process).
        let parts = handle.into_parts();

        let stdin = tokio::sync::Mutex::new(parts.stdin);
        let child = tokio::sync::Mutex::new(Some(parts.child));
        let pending = Arc::new(std::sync::Mutex::new(HashMap::<
            String,
            oneshot::Sender<ExecuteResponseV2>,
        >::new()));
        let control = Arc::new(tokio::sync::Mutex::new(
            None::<oneshot::Sender<WorkerControlResponse>>,
        ));

        let reader_pending = pending.clone();
        let reader_control = control.clone();
        let reader_pid = pid;

        let reader_task = tokio::spawn(async move {
            Self::reader_loop(parts.stdout, reader_pending, reader_control, reader_pid).await;
        });

        Self {
            child,
            stdin,
            pending,
            control,
            reader_task,
            pid,
            config,
            shutdown_started: AtomicBool::new(false),
        }
    }

    /// Send one typed V2 execute request and await the response.
    ///
    /// Multiple callers can invoke this concurrently — stdin writes are
    /// serialized by the mutex, and responses are routed by request_id.
    pub(super) async fn execute_v2(
        &self,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        if self.shutdown_started.load(Ordering::Acquire) {
            return Err(WorkerError::Protocol("GPU worker is shutting down".into()));
        }

        let request_id = request.request_id.to_string();
        let (tx, rx) = oneshot::channel();

        // Register the pending response channel before writing the request,
        // so the reader task can route the response as soon as it arrives.
        {
            let mut pending = super::lock_recovered(&self.pending);
            pending.insert(request_id.clone(), tx);
        }

        // Write the request under the stdin mutex.
        {
            let mut stdin = self.stdin.lock().await;
            let envelope = serde_json::json!({
                "op": "execute_v2",
                "request": request
            });
            let mut line = serde_json::to_string(&envelope)
                .map_err(|e| WorkerError::Protocol(format!("failed to encode request: {e}")))?;
            line.push('\n');
            if let Err(e) = stdin.write_all(line.as_bytes()).await {
                // Remove the pending entry on write failure.
                super::lock_recovered(&self.pending).remove(&request_id);
                return Err(e.into());
            }
            if let Err(e) = stdin.flush().await {
                super::lock_recovered(&self.pending).remove(&request_id);
                return Err(e.into());
            }
        }

        // Wait for the response with a timeout.
        let timeout_s = request.timeout_seconds_with_config(
            self.config.audio_task_timeout_s,
            self.config.analysis_task_timeout_s,
        );
        let timeout = Duration::from_secs(timeout_s);
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                // Sender dropped — reader task died or worker exited.
                Err(WorkerError::Protocol(
                    "GPU worker response channel closed (worker may have exited)".into(),
                ))
            }
            Err(_) => {
                // Timeout — remove the pending entry.
                super::lock_recovered(&self.pending).remove(&request_id);
                Err(WorkerError::Protocol(format!(
                    "timeout ({timeout_s}s) waiting for GPU execute_v2 response (request_id={request_id})"
                )))
            }
        }
    }

    /// Run a health check via the control channel.
    #[allow(dead_code)]
    pub(super) async fn health_check(
        &self,
    ) -> Result<crate::worker::WorkerHealthResponse, WorkerError> {
        if self.shutdown_started.load(Ordering::Acquire) {
            return Err(WorkerError::HealthCheckFailed(
                "GPU worker is shutting down".into(),
            ));
        }

        let (tx, rx) = oneshot::channel();
        {
            let mut ctrl = self.control.lock().await;
            *ctrl = Some(tx);
        }

        // Write health request.
        {
            let mut stdin = self.stdin.lock().await;
            let line = b"{\"op\":\"health\"}\n";
            stdin.write_all(line).await?;
            stdin.flush().await?;
        }

        match tokio::time::timeout(Duration::from_secs(10), rx).await {
            Ok(Ok(WorkerControlResponse::Health(response))) => {
                if !response.status.is_ok() {
                    return Err(WorkerError::HealthCheckFailed(format!(
                        "status={}", response.status
                    )));
                }
                Ok(response)
            }
            Ok(Ok(WorkerControlResponse::Error(error))) => {
                Err(WorkerError::HealthCheckFailed(error))
            }
            Ok(Ok(other)) => Err(WorkerError::HealthCheckFailed(format!(
                "unexpected control response for health: {other:?}"
            ))),
            Ok(Err(_)) => Err(WorkerError::HealthCheckFailed(
                "control channel closed".into(),
            )),
            Err(_) => Err(WorkerError::HealthCheckFailed(
                "timeout waiting for health response".into(),
            )),
        }
    }

    /// Gracefully shut down the GPU worker.
    pub(super) async fn shutdown(&self) {
        if self
            .shutdown_started
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }

        info!(
            target = %self.config.profile.label(),
            pid = %self.pid,
            "Shutting down shared GPU worker"
        );

        let shutdown_ack = {
            let (tx, rx) = oneshot::channel();
            let mut ctrl = self.control.lock().await;
            if ctrl.is_some() {
                warn!(
                    pid = %self.pid,
                    "Overriding in-flight GPU worker control request during shutdown"
                );
            }
            *ctrl = Some(tx);
            rx
        };

        let wrote_shutdown = {
            let mut stdin = self.stdin.lock().await;
            match stdin.write_all(b"{\"op\":\"shutdown\"}\n").await {
                Ok(()) => match stdin.flush().await {
                    Ok(()) => true,
                    Err(error) => {
                        warn!(
                            pid = %self.pid,
                            error = %error,
                            "Failed to flush shared GPU shutdown request"
                        );
                        false
                    }
                },
                Err(error) => {
                    warn!(
                        pid = %self.pid,
                        error = %error,
                        "Failed to write shared GPU shutdown request"
                    );
                    false
                }
            }
        };

        if wrote_shutdown {
            match tokio::time::timeout(Duration::from_secs(2), shutdown_ack).await {
                Ok(Ok(WorkerControlResponse::Shutdown)) => {}
                Ok(Ok(other)) => {
                    warn!(
                        pid = %self.pid,
                        response = ?other,
                        "Shared GPU worker returned unexpected shutdown response"
                    );
                }
                Ok(Err(_)) => {
                    debug!(pid = %self.pid, "Shared GPU shutdown ack channel closed");
                }
                Err(_) => {
                    debug!(pid = %self.pid, "Timed out waiting for shared GPU shutdown ack");
                }
            }
        }

        self.finish_shutdown().await;
    }

    /// The worker process ID.
    pub(super) fn pid(&self) -> WorkerPid {
        self.pid
    }

    /// The worker's profile label.
    pub(super) fn profile_label(&self) -> &'static str {
        self.config.profile.label()
    }

    /// The worker's language code.
    pub(super) fn lang(&self) -> &str {
        self.config.lang.as_worker_arg()
    }

    /// Background reader loop that routes responses from worker stdout.
    async fn reader_loop(
        mut stdout: BufReader<ChildStdout>,
        pending: Arc<std::sync::Mutex<HashMap<String, oneshot::Sender<ExecuteResponseV2>>>>,
        control: Arc<tokio::sync::Mutex<Option<oneshot::Sender<WorkerControlResponse>>>>,
        pid: WorkerPid,
    ) {
        Self::reader_loop_generic(&mut stdout, pending, control, pid).await;
    }

    /// Generic reader loop that works with any `AsyncBufRead` — shared between
    /// stdio ([`SharedGpuWorker`]) and TCP ([`SharedGpuTcpWorker`]).
    pub(crate) async fn reader_loop_generic<R: tokio::io::AsyncBufRead + Unpin>(
        reader: &mut R,
        pending: Arc<std::sync::Mutex<HashMap<String, oneshot::Sender<ExecuteResponseV2>>>>,
        control: Arc<tokio::sync::Mutex<Option<oneshot::Sender<WorkerControlResponse>>>>,
        pid: WorkerPid,
    ) {
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    debug!(pid = %pid, "GPU worker stream closed (EOF)");
                    let mut pending = lock_recovered(&pending);
                    for (id, tx) in pending.drain() {
                        debug!(pid = %pid, request_id = %id, "Failing pending request (worker stream closed)");
                        drop(tx);
                    }
                    break;
                }
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    let parsed: Value = match serde_json::from_str(trimmed) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!(
                                pid = %pid,
                                line = trimmed,
                                error = %e,
                                "GPU worker: ignoring non-JSON line"
                            );
                            continue;
                        }
                    };

                    let op = parsed.get("op").and_then(|v| v.as_str()).unwrap_or("");

                    match op {
                        "execute_v2" => {
                            match serde_json::from_value::<ExecuteResponseV2Envelope>(
                                parsed.clone(),
                            ) {
                                Ok(envelope) => {
                                    let request_id = envelope.response.request_id.to_string();
                                    let mut pending = lock_recovered(&pending);
                                    if let Some(tx) = pending.remove(&request_id) {
                                        let _ = tx.send(envelope.response);
                                    } else {
                                        warn!(
                                            pid = %pid,
                                            request_id = %request_id,
                                            "GPU worker: orphaned execute_v2 response"
                                        );
                                    }
                                }
                                Err(e) => {
                                    error!(
                                        pid = %pid,
                                        error = %e,
                                        "GPU worker: failed to parse execute_v2 response"
                                    );
                                }
                            }
                        }
                        "health" => {
                            if let Ok(envelope) =
                                serde_json::from_value::<HealthResponseEnvelope>(parsed)
                            {
                                let mut ctrl = control.lock().await;
                                if let Some(tx) = ctrl.take() {
                                    let _ =
                                        tx.send(WorkerControlResponse::Health(envelope.response));
                                }
                            }
                        }
                        "capabilities" => {
                            if let Ok(envelope) =
                                serde_json::from_value::<CapabilitiesResponseEnvelope>(parsed)
                            {
                                let mut ctrl = control.lock().await;
                                if let Some(tx) = ctrl.take() {
                                    let _ = tx.send(WorkerControlResponse::Capabilities(
                                        envelope.response,
                                    ));
                                }
                            }
                        }
                        "shutdown" => {
                            let mut ctrl = control.lock().await;
                            if let Some(tx) = ctrl.take() {
                                let _ = tx.send(WorkerControlResponse::Shutdown);
                            }
                        }
                        "error" => {
                            let error_msg = parsed
                                .get("error")
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown error")
                                .to_string();
                            let mut ctrl = control.lock().await;
                            if let Some(tx) = ctrl.take() {
                                let _ = tx.send(WorkerControlResponse::Error(error_msg));
                            }
                        }
                        _ => {
                            warn!(
                                pid = %pid,
                                op = op,
                                "GPU worker: unexpected response op"
                            );
                        }
                    }
                }
                Err(e) => {
                    error!(pid = %pid, error = %e, "GPU worker: stream read error");
                    break;
                }
            }
        }
    }

    async fn finish_shutdown(&self) {
        // Layer 3: remove PID file before killing.
        super::reaper::remove_worker_pid(self.pid.0);

        let mut child = {
            let mut child_slot = self.child.lock().await;
            child_slot.take()
        };

        if let Some(mut child) = child.take() {
            #[cfg(unix)]
            {
                let _ = child.id().map(|pid| {
                    // SAFETY: the worker was spawned as its own process group.
                    unsafe { libc::killpg(pid as libc::pid_t, libc::SIGTERM) };
                });
            }

            match tokio::time::timeout(Duration::from_secs(5), child.wait()).await {
                Ok(Ok(status)) => {
                    info!(pid = %self.pid, ?status, "Shared GPU worker exited gracefully");
                }
                Ok(Err(error)) => {
                    warn!(pid = %self.pid, error = %error, "Error waiting for shared GPU worker");
                }
                Err(_) => {
                    warn!(
                        pid = %self.pid,
                        "Shared GPU worker didn't exit in 5s, killing process group"
                    );
                    #[cfg(unix)]
                    {
                        let _ = child.id().map(|pid| {
                            // SAFETY: the worker was spawned as its own process group.
                            unsafe { libc::killpg(pid as libc::pid_t, libc::SIGKILL) };
                        });
                    }
                    let _ = child.kill().await;
                }
            }
        }

        self.reader_task.abort();
        self.fail_pending_requests();
        let mut ctrl = self.control.lock().await;
        ctrl.take();
    }

    fn fail_pending_requests(&self) {
        let mut pending = super::lock_recovered(&self.pending);
        for (_, tx) in pending.drain() {
            drop(tx);
        }
    }
}

impl Drop for SharedGpuWorker {
    fn drop(&mut self) {
        self.shutdown_started.store(true, Ordering::Release);
        super::reaper::remove_worker_pid(self.pid.0);
        self.reader_task.abort();
        self.fail_pending_requests();

        if let Ok(mut child_slot) = self.child.try_lock()
            && let Some(child) = child_slot.as_mut()
        {
            #[cfg(unix)]
            {
                if let Some(pid) = child.id() {
                    let pgid = pid as libc::pid_t;
                    // SAFETY: the worker was spawned as its own process group.
                    unsafe {
                        libc::killpg(pgid, libc::SIGTERM);
                    }
                    // Brief pause then SIGKILL to prevent zombies holding GPU/RAM.
                    std::thread::sleep(std::time::Duration::from_millis(200));
                    if unsafe { libc::kill(pgid, 0) } == 0 {
                        unsafe {
                            libc::killpg(pgid, libc::SIGKILL);
                        }
                    }
                }
            }
            let _ = child.start_kill();
        }
    }
}

// ---------------------------------------------------------------------------
// SharedGpuTcpWorker — concurrent dispatch to a TCP GPU worker
// ---------------------------------------------------------------------------

/// A GPU worker that supports concurrent V2 request dispatch over TCP.
///
/// Similar to [`SharedGpuWorker`] but connects via TCP instead of stdio. Uses
/// a background reader task to route responses by `request_id`, just like the
/// stdio variant. The key difference: dropping does not kill the worker process.
pub(crate) struct SharedGpuTcpWorker {
    /// Serialized writes to the TCP socket.
    writer: tokio::sync::Mutex<tokio::io::WriteHalf<tokio::net::TcpStream>>,

    /// Pending V2 requests awaiting responses, keyed by request_id.
    pending: Arc<std::sync::Mutex<HashMap<String, oneshot::Sender<ExecuteResponseV2>>>>,

    /// Control channel for sequential non-V2 ops.
    #[allow(dead_code)]
    control: Arc<tokio::sync::Mutex<Option<oneshot::Sender<WorkerControlResponse>>>>,

    /// Background reader task handle.
    reader_task: tokio::task::JoinHandle<()>,

    /// Worker process ID (from registry, for display).
    pid: WorkerPid,

    /// Timeout for audio-heavy tasks.
    audio_task_timeout_s: u64,

    /// Timeout for analysis tasks.
    analysis_task_timeout_s: u64,
}

impl SharedGpuTcpWorker {
    /// Connect to a TCP GPU worker and set up concurrent dispatch.
    pub(crate) async fn connect(
        info: crate::worker::tcp_handle::TcpWorkerInfo,
    ) -> Result<Self, WorkerError> {
        let addr = format!("{}:{}", info.host, info.port);
        let stream = tokio::time::timeout(
            Duration::from_secs(10),
            tokio::net::TcpStream::connect(&addr),
        )
        .await
        .map_err(|_| {
            WorkerError::Protocol(format!("timeout connecting to TCP GPU worker at {addr}"))
        })?
        .map_err(|e| {
            WorkerError::Protocol(format!(
                "failed to connect to TCP GPU worker at {addr}: {e}"
            ))
        })?;

        let pid = info.pid;
        let audio_task_timeout_s = info.audio_task_timeout_s;
        let analysis_task_timeout_s = info.analysis_task_timeout_s;

        let (read_half, write_half) = tokio::io::split(stream);
        let writer = tokio::sync::Mutex::new(write_half);
        let pending = Arc::new(std::sync::Mutex::new(HashMap::<
            String,
            oneshot::Sender<ExecuteResponseV2>,
        >::new()));
        let control = Arc::new(tokio::sync::Mutex::new(
            None::<oneshot::Sender<WorkerControlResponse>>,
        ));

        let reader_pending = pending.clone();
        let reader_control = control.clone();
        let reader_pid = pid;

        let reader_task = tokio::spawn(async move {
            let mut reader = BufReader::new(read_half);
            // Reuse the same reader loop logic as stdio SharedGpuWorker.
            SharedGpuWorker::reader_loop_generic(
                &mut reader,
                reader_pending,
                reader_control,
                reader_pid,
            )
            .await;
        });

        Ok(Self {
            writer,
            pending,
            control,
            reader_task,
            pid,
            audio_task_timeout_s,
            analysis_task_timeout_s,
        })
    }

    /// Send one typed V2 execute request concurrently.
    pub(crate) async fn execute_v2(
        &self,
        request: &ExecuteRequestV2,
    ) -> Result<ExecuteResponseV2, WorkerError> {
        let request_id = request.request_id.to_string();
        let (tx, rx) = oneshot::channel();

        {
            let mut pending = super::lock_recovered(&self.pending);
            pending.insert(request_id.clone(), tx);
        }

        {
            let mut writer = self.writer.lock().await;
            let envelope = serde_json::json!({
                "op": "execute_v2",
                "request": request
            });
            let mut line = serde_json::to_string(&envelope)
                .map_err(|e| WorkerError::Protocol(format!("failed to encode request: {e}")))?;
            line.push('\n');
            if let Err(e) = writer.write_all(line.as_bytes()).await {
                super::lock_recovered(&self.pending).remove(&request_id);
                return Err(e.into());
            }
            if let Err(e) = writer.flush().await {
                super::lock_recovered(&self.pending).remove(&request_id);
                return Err(e.into());
            }
        }

        let timeout_s = request
            .timeout_seconds_with_config(self.audio_task_timeout_s, self.analysis_task_timeout_s);
        let timeout = Duration::from_secs(timeout_s);
        match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => Err(WorkerError::Protocol(
                "TCP GPU worker response channel closed (worker may have exited)".into(),
            )),
            Err(_) => {
                super::lock_recovered(&self.pending).remove(&request_id);
                Err(WorkerError::Protocol(format!(
                    "timeout ({timeout_s}s) waiting for TCP GPU execute_v2 response (request_id={request_id})"
                )))
            }
        }
    }

    /// Gracefully shut down the TCP GPU worker connection.
    pub(crate) async fn shutdown(&self) {
        {
            let mut writer = self.writer.lock().await;
            let _ = writer.write_all(b"{\"op\":\"shutdown\"}\n").await;
            let _ = writer.flush().await;
        }
        self.reader_task.abort();
        {
            let mut pending = super::lock_recovered(&self.pending);
            for (_, tx) in pending.drain() {
                drop(tx);
            }
        }
    }

    /// The worker process ID.
    pub(crate) fn pid(&self) -> WorkerPid {
        self.pid
    }
}

/// Helper envelope for deserializing `{"op": "execute_v2", "response": {...}}`.
#[derive(serde::Deserialize)]
struct ExecuteResponseV2Envelope {
    response: ExecuteResponseV2,
}

/// Helper envelope for deserializing `{"op": "health", "response": {...}}`.
#[derive(serde::Deserialize)]
struct HealthResponseEnvelope {
    response: crate::worker::WorkerHealthResponse,
}

/// Helper envelope for deserializing `{"op": "capabilities", "response": {...}}`.
#[derive(serde::Deserialize)]
struct CapabilitiesResponseEnvelope {
    response: crate::worker::WorkerCapabilities,
}
