//! Local daemon lifecycle — mirrors `batchalign/cli/daemon.py`.
//!
//! Key difference from Python: the daemon starts the **Rust server binary**
//! (`batchalign3 serve start --foreground`), not the Python server.
//!
//! ## Port policy
//!
//! The daemon always uses the port from the server config (default 8000).
//! No random ephemeral ports — this makes discovery deterministic and avoids
//! orphaned servers on random ports after crashes.
//!
//! ## Stale-binary detection
//!
//! `DaemonInfo` carries a `build_hash` (set at write time from
//! [`crate::build_hash()`]).  `ensure_daemon_locked()` compares it against
//! the current binary's hash and auto-restarts on mismatch.  Old daemon.json
//! files that lack `build_hash` fall back to version comparison.

use std::path::{Path, PathBuf};
use std::time::Duration;

use batchalign_app::config::RuntimeLayout;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::client::BatchalignClient;
use crate::error::CliError;
use crate::python::resolve_python_executable;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum seconds to wait for a newly spawned daemon to pass its first
/// health check. 90 seconds is generous enough to cover cold-start model
/// loading on slow machines (Stanza downloads can take 30--60s on first run)
/// while still failing within a human-tolerable window if something is
/// genuinely broken (port conflict, missing Python, bad config).
const HEALTH_TIMEOUT: f64 = 90.0;

/// How often to poll the daemon's `/health` endpoint while waiting for
/// startup. 1 second balances responsiveness (the user sees the daemon come
/// up within a second of it being ready) against CPU/network cost (one
/// loopback HTTP request per second is negligible).
const HEALTH_POLL: f64 = 1.0;

fn runtime_layout() -> RuntimeLayout {
    RuntimeLayout::from_env()
}

/// Return the configured daemon port (from server.yaml or the default).
fn config_port(layout: &RuntimeLayout) -> Result<u16, CliError> {
    let (cfg, warnings) = batchalign_app::config::load_validated_config_from_layout(layout, None)?;
    for warning in warnings {
        eprintln!("warning: {warning}");
    }
    Ok(cfg.port)
}

// ---------------------------------------------------------------------------
// Profiles
// ---------------------------------------------------------------------------

/// Which daemon role to start or connect to.
///
/// The CLI can manage two independent daemon processes simultaneously.
/// Each profile gets its own state file, lock file, and log file so they
/// do not interfere with each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DaemonProfile {
    /// The primary managed daemon profile for explicit `serve` lifecycle
    /// commands and compatibility helpers. Local processing commands no longer
    /// select this profile by default; they run through direct mode unless the
    /// user explicitly targets a server. Uses the system or venv Python
    /// resolved by [`resolve_python_executable()`].
    Main,
    /// A secondary daemon dedicated to transcribe workloads that require
    /// a different Python environment (e.g. Python 3.12 with
    /// openai-whisper, while the main daemon runs 3.14t). Selected when
    /// the dispatch layer detects a transcribe command and the sidecar
    /// Python is available. Uses `BATCHALIGN_SIDECAR_PYTHON` or falls
    /// back to `~/.batchalign3/sidecar/.venv/bin/python`.
    Sidecar,
}

impl DaemonProfile {
    fn label(self) -> &'static str {
        match self {
            Self::Main => "local",
            Self::Sidecar => "sidecar",
        }
    }

    fn state_file(self, dir: &Path) -> PathBuf {
        match self {
            Self::Main => dir.join("daemon.json"),
            Self::Sidecar => dir.join("sidecar-daemon.json"),
        }
    }

    fn lock_file(self, dir: &Path) -> PathBuf {
        match self {
            Self::Main => dir.join("daemon.lock"),
            Self::Sidecar => dir.join("sidecar-daemon.lock"),
        }
    }

    fn log_file(self, dir: &Path) -> PathBuf {
        match self {
            Self::Main => dir.join("daemon.log"),
            Self::Sidecar => dir.join("sidecar-daemon.log"),
        }
    }

    fn startup_message(self) -> &'static str {
        match self {
            Self::Main => "Starting local daemon...",
            Self::Sidecar => "Starting sidecar daemon for transcribe workloads...",
        }
    }

    fn check_manual_server(self) -> bool {
        matches!(self, Self::Main)
    }

    fn default_python(self, dir: &Path) -> String {
        match self {
            Self::Main => resolve_python_executable(),
            Self::Sidecar => std::env::var("BATCHALIGN_SIDECAR_PYTHON").unwrap_or_else(|_| {
                dir.join("sidecar")
                    .join(".venv")
                    .join("bin")
                    .join("python")
                    .to_string_lossy()
                    .into_owned()
            }),
        }
    }

    fn require_sidecar_python_file(self) -> bool {
        matches!(self, Self::Sidecar) && std::env::var("BATCHALIGN_SIDECAR_PYTHON").is_err()
    }
}

// ---------------------------------------------------------------------------
// DaemonInfo — state file
// ---------------------------------------------------------------------------

/// State persisted to `daemon.json` so the CLI can reconnect to a running daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonInfo {
    /// OS process ID of the daemon.
    pub pid: u32,
    /// TCP port the daemon is listening on.
    pub port: u16,
    /// Daemon version string (for stale-binary detection).
    #[serde(default)]
    pub version: String,
    /// Unix timestamp when the daemon was started.
    #[serde(default)]
    pub started_at: f64,
    /// Build fingerprint — empty for old state files (falls back to version).
    #[serde(default)]
    pub build_hash: String,
    /// Whether this daemon was started with `--force-cpu`.
    #[serde(default)]
    pub force_cpu: bool,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return the main daemon URL or `None` if the daemon cannot be started.
pub async fn ensure_daemon(
    force_cpu: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<Option<String>, CliError> {
    ensure_daemon_for(DaemonProfile::Main, force_cpu, workers, timeout).await
}

/// Return the sidecar daemon URL or `None` if the daemon cannot be started.
pub async fn ensure_sidecar_daemon(
    force_cpu: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<Option<String>, CliError> {
    ensure_daemon_for(DaemonProfile::Sidecar, force_cpu, workers, timeout).await
}

/// Stop the main daemon. Returns `true` if a process was killed.
pub async fn stop_daemon() -> Result<bool, CliError> {
    stop_profile(DaemonProfile::Main)
}

/// Stop the sidecar daemon. Returns `true` if a process was killed.
pub async fn stop_sidecar_daemon() -> Result<bool, CliError> {
    stop_profile(DaemonProfile::Sidecar)
}

/// Read the main daemon state file. Public for status checks.
pub fn read_daemon_info() -> Option<DaemonInfo> {
    let layout = runtime_layout();
    read_daemon_info_for(DaemonProfile::Main, layout.state_dir())
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

async fn ensure_daemon_for(
    profile: DaemonProfile,
    force_cpu: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<Option<String>, CliError> {
    let layout = runtime_layout();
    let dir = layout.state_dir();
    std::fs::create_dir_all(dir)?;

    let lock_path = profile.lock_file(dir);
    let lock_file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&lock_path)?;

    match lock_file.try_lock_exclusive() {
        Ok(()) => {}
        Err(_) => {
            debug!(profile = profile.label(), "Waiting for daemon lock...");
            lock_file.lock_exclusive()?;
        }
    }

    let result = ensure_daemon_locked(profile, &layout, force_cpu, workers, timeout).await;
    drop(lock_file);
    result
}

fn stop_profile(profile: DaemonProfile) -> Result<bool, CliError> {
    let layout = runtime_layout();
    let dir = layout.state_dir();
    let info = match read_daemon_info_for(profile, dir) {
        Some(i) => i,
        None => return Ok(false),
    };

    let killed = kill_process(info.pid);
    cleanup_state_file_for(profile, dir);
    Ok(killed)
}

/// Check if the daemon state file indicates a stale binary.
///
/// If `build_hash` is present in the state file, compare against the current
/// build hash.  Otherwise, fall back to a version comparison (backward compat
/// with old state files).
fn is_stale(info: &DaemonInfo) -> bool {
    if !info.build_hash.is_empty() {
        return info.build_hash != crate::build_hash();
    }
    // Old state file — fall back to version comparison
    info.version != current_version()
}

fn runtime_mismatch(info: &DaemonInfo, force_cpu: bool) -> bool {
    info.force_cpu != force_cpu
}

async fn ensure_daemon_locked(
    profile: DaemonProfile,
    layout: &RuntimeLayout,
    force_cpu: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<Option<String>, CliError> {
    let dir = layout.state_dir();
    if profile.check_manual_server()
        && let Some(url) = detect_manual_server(layout).await?
    {
        return Ok(Some(url));
    }

    let port = config_port(layout)?;

    if let Some(info) = read_daemon_info_for(profile, dir) {
        if is_process_alive(info.pid) {
            if is_stale(&info) {
                eprintln!(
                    "Restarting {} daemon (stale build: {} -> {})...",
                    profile.label(),
                    if info.build_hash.is_empty() {
                        &info.version
                    } else {
                        &info.build_hash
                    },
                    crate::build_hash(),
                );
                kill_process(info.pid);
                cleanup_state_file_for(profile, dir);
                return start_daemon(profile, layout, port, force_cpu, workers, timeout).await;
            }

            if runtime_mismatch(&info, force_cpu) {
                eprintln!(
                    "Restarting {} daemon (force_cpu {} -> {})...",
                    profile.label(),
                    info.force_cpu,
                    force_cpu,
                );
                kill_process(info.pid);
                cleanup_state_file_for(profile, dir);
                return start_daemon(profile, layout, port, force_cpu, workers, timeout).await;
            }

            if health_check(info.port).await {
                return Ok(Some(format!("http://127.0.0.1:{}", info.port)));
            }

            kill_process(info.pid);
            cleanup_state_file_for(profile, dir);
            return start_daemon(profile, layout, port, force_cpu, workers, timeout).await;
        }
        // Process is dead but state file exists -- stale PID file.
        debug!(
            profile = profile.label(),
            pid = info.pid,
            "Cleaning up stale daemon state file (process is dead)"
        );
        cleanup_state_file_for(profile, dir);
    }

    start_daemon(profile, layout, port, force_cpu, workers, timeout).await
}

async fn detect_manual_server(layout: &RuntimeLayout) -> Result<Option<String>, CliError> {
    let pid_path = layout.server_pid_path();
    let pid_str = match std::fs::read_to_string(&pid_path) {
        Ok(pid_str) => pid_str,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(CliError::Io(err)),
    };
    let pid: u32 = match pid_str.trim().parse() {
        Ok(pid) => pid,
        Err(_) => {
            // Corrupt PID file -- clean it up rather than returning an error
            // that would block the caller from starting a daemon.
            debug!(path = %pid_path.display(), "Removing corrupt server PID file");
            let _ = std::fs::remove_file(&pid_path);
            return Ok(None);
        }
    };

    if !is_process_alive(pid) {
        // Stale PID file from a crashed server -- clean it up.
        debug!(pid, path = %pid_path.display(), "Removing stale server PID file (process is dead)");
        let _ = std::fs::remove_file(&pid_path);
        return Ok(None);
    }

    let (cfg, warnings) = batchalign_app::config::load_validated_config_from_layout(layout, None)?;
    for warning in warnings {
        eprintln!("warning: {warning}");
    }
    let port = cfg.port;

    if startup_health_check(port).await {
        // Warn if the manual server has a stale build hash
        check_manual_server_staleness(port).await;
        debug!(port, pid, "Reusing manual server");
        Ok(Some(format!("http://127.0.0.1:{port}")))
    } else {
        Ok(None)
    }
}

/// Best-effort check: warn if a manual server's build hash differs from ours.
async fn check_manual_server_staleness(port: u16) {
    let client = BatchalignClient::new();
    if let Ok(health) = client
        .health_check(&format!("http://127.0.0.1:{port}"))
        .await
        && !health.build_hash.is_empty()
        && health.build_hash != crate::build_hash()
    {
        eprintln!(
            "warning: manual server on port {port} has a different build ({}). \
             Restart with `batchalign3 serve stop && batchalign3 serve start`.",
            health.build_hash,
        );
    }
}

async fn start_daemon(
    profile: DaemonProfile,
    layout: &RuntimeLayout,
    port: u16,
    force_cpu: bool,
    workers: Option<usize>,
    timeout: Option<u64>,
) -> Result<Option<String>, CliError> {
    let dir = layout.state_dir();
    let python = profile.default_python(dir);
    if profile.require_sidecar_python_file() && !Path::new(&python).is_file() {
        eprintln!(
            "warning: sidecar python not found at {python}. \
             Set BATCHALIGN_SIDECAR_PYTHON to a Python with transcribe deps."
        );
        return Ok(None);
    }

    eprintln!("{}", profile.startup_message());

    let exe = crate::self_exe::resolve_self_exe();
    let mut cmd = std::process::Command::new(&exe);
    cmd.args([
        "serve",
        "start",
        "--foreground",
        "--port",
        &port.to_string(),
        "--host",
        "127.0.0.1",
        "--python",
        &python,
    ]);

    let config_path = layout.config_path();
    if config_path.exists() {
        cmd.arg("--config").arg(config_path);
    } else {
        // Local profile default: optimize for low idle memory/cold startup.
        // Deployments with explicit server.yaml keep their configured policy.
        cmd.args(["--warmup-policy", "off"]);
    }
    if force_cpu {
        cmd.arg("--force-cpu");
    }
    if let Some(n) = workers {
        cmd.args(["--workers", &n.to_string()]);
    }
    if let Some(t) = timeout {
        cmd.args(["--timeout", &t.to_string()]);
    }

    let log_path = profile.log_file(dir);
    // Append mode: preserve previous daemon logs across restarts.
    // Previous behavior (File::create) truncated logs on every restart,
    // destroying crash diagnostics from the previous session.
    let log_file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&log_path)?;
    cmd.stdout(std::process::Stdio::null());
    cmd.stderr(log_file);

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

    let proc = match cmd.spawn() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("warning: failed to spawn daemon process: {e}");
            debug!(profile = profile.label(), error = %e, "Failed to spawn daemon");
            eprintln!(
                "warning: could not start local daemon. Check {}\n\
                 hint: run `batchalign3 serve start --foreground` to see startup errors.",
                log_path.display()
            );
            return Ok(None);
        }
    };

    let pid = proc.id();
    write_daemon_info_for(profile, dir, pid, port, force_cpu)?;

    if wait_for_health(pid, port).await {
        eprintln!(
            "{} daemon ready on port {} (PID {})",
            profile.label(),
            port,
            pid
        );
        return Ok(Some(format!("http://127.0.0.1:{port}")));
    }

    debug!(
        profile = profile.label(),
        port, "Daemon failed to become healthy"
    );
    kill_process(pid);
    cleanup_state_file_for(profile, dir);

    eprintln!(
        "warning: could not start local daemon. Check {}\n\
         hint: run `batchalign3 serve start --foreground` to see startup errors.",
        log_path.display()
    );
    Ok(None)
}

async fn wait_for_health(pid: u32, port: u16) -> bool {
    let deadline = std::time::Instant::now() + Duration::from_secs_f64(HEALTH_TIMEOUT);

    while std::time::Instant::now() < deadline {
        if !is_process_alive(pid) {
            return false;
        }

        if startup_health_check(port).await {
            return true;
        }

        tokio::time::sleep(Duration::from_secs_f64(HEALTH_POLL)).await;
    }

    false
}

/// Quick health probe for localhost daemon startup.
///
/// Uses a short timeout (3s request, 1s connect) instead of the full
/// `BatchalignClient` timeout (120s). A connection to 127.0.0.1 should
/// succeed in well under 1 second, so this gives fast failure detection
/// while still tolerating brief startup latency.
async fn startup_health_check(port: u16) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .connect_timeout(Duration::from_secs(1))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            eprintln!("warning: failed to build localhost health-check client: {error}");
            return false;
        }
    };
    client
        .get(format!("http://127.0.0.1:{port}/health"))
        .send()
        .await
        .is_ok_and(|r| r.status().is_success())
}

/// Full health check via `BatchalignClient` (120s timeout).
/// Used for the reuse path when daemon is already running and we want
/// the richer `HealthResponse`.
async fn health_check(port: u16) -> bool {
    let client = BatchalignClient::new();
    client
        .health_check(&format!("http://127.0.0.1:{port}"))
        .await
        .is_ok()
}

fn read_daemon_info_for(profile: DaemonProfile, dir: &Path) -> Option<DaemonInfo> {
    let path = profile.state_file(dir);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => {
            eprintln!(
                "warning: failed to read daemon state at {}: {}",
                path.display(),
                error
            );
            return None;
        }
    };
    match serde_json::from_str(&text) {
        Ok(info) => Some(info),
        Err(error) => {
            eprintln!(
                "warning: ignoring corrupt daemon state at {}: {}",
                path.display(),
                error
            );
            let _ = std::fs::remove_file(&path);
            None
        }
    }
}

fn write_daemon_info_for(
    profile: DaemonProfile,
    dir: &Path,
    pid: u32,
    port: u16,
    force_cpu: bool,
) -> Result<(), CliError> {
    let started_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| {
            CliError::Io(std::io::Error::other(format!(
                "system clock before unix epoch while writing daemon state: {error}"
            )))
        })?
        .as_secs_f64();
    let info = DaemonInfo {
        pid,
        port,
        version: current_version(),
        started_at,
        build_hash: crate::build_hash().to_string(),
        force_cpu,
    };

    std::fs::create_dir_all(dir)?;
    let state_path = profile.state_file(dir);
    let tmp = state_path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_string(&info)?)?;
    std::fs::rename(&tmp, &state_path)?;
    Ok(())
}

fn cleanup_state_file_for(profile: DaemonProfile, dir: &Path) {
    let _ = std::fs::remove_file(profile.state_file(dir));
}

fn is_process_alive(pid: u32) -> bool {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as i32, 0) == 0
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

/// Kill a daemon process: SIGTERM the process group, then the process
/// directly, then wait up to 3 seconds and escalate to SIGKILL if still
/// alive. Returns `true` if the process was signalled at all.
fn kill_process(pid: u32) -> bool {
    #[cfg(unix)]
    {
        let pgid_ok = unsafe { libc::killpg(pid as libc::pid_t, libc::SIGTERM) == 0 };
        let pid_ok = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) == 0 };

        if !pgid_ok && !pid_ok {
            return false;
        }

        // Wait for the process to exit before returning so the port is
        // released by the time the caller tries to rebind.
        for _ in 0..6 {
            std::thread::sleep(std::time::Duration::from_millis(500));
            if !is_process_alive(pid) {
                return true;
            }
        }

        // Still alive after 3 seconds -- escalate to SIGKILL.
        debug!(pid, "Process did not exit after SIGTERM, sending SIGKILL");
        unsafe {
            libc::killpg(pid as libc::pid_t, libc::SIGKILL);
            libc::kill(pid as libc::pid_t, libc::SIGKILL);
        }
        // Brief wait for SIGKILL to take effect.
        std::thread::sleep(std::time::Duration::from_millis(200));
        true
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
        false
    }
}

fn current_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn daemon_info_roundtrip() {
        let info = DaemonInfo {
            pid: 12345,
            port: 54321,
            version: "0.1.0".to_string(),
            started_at: 1700000000.0,
            build_hash: "0.1.0-abc1234-1700000000".to_string(),
            force_cpu: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: DaemonInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(back.pid, 12345);
        assert_eq!(back.port, 54321);
        assert_eq!(back.version, "0.1.0");
        assert_eq!(back.build_hash, "0.1.0-abc1234-1700000000");
    }

    #[test]
    fn daemon_info_missing_build_hash() {
        // Old state files lack build_hash — should default to empty
        let json = r#"{"pid": 999, "port": 8000, "version": "0.1.0"}"#;
        let info: DaemonInfo = serde_json::from_str(json).unwrap();
        assert_eq!(info.build_hash, "");
    }

    #[test]
    fn is_stale_detects_build_hash_mismatch() {
        let info = DaemonInfo {
            pid: 1,
            port: 8000,
            version: current_version(),
            started_at: 0.0,
            build_hash: "old-build-hash".to_string(),
            force_cpu: false,
        };
        // Our build hash is different from "old-build-hash"
        assert!(is_stale(&info));
    }

    #[test]
    fn is_stale_falls_back_to_version_when_no_build_hash() {
        let info = DaemonInfo {
            pid: 1,
            port: 8000,
            version: "0.0.0-fake".to_string(),
            started_at: 0.0,
            build_hash: String::new(),
            force_cpu: false,
        };
        assert!(is_stale(&info));

        let info_current = DaemonInfo {
            pid: 1,
            port: 8000,
            version: current_version(),
            started_at: 0.0,
            build_hash: String::new(),
            force_cpu: false,
        };
        assert!(!is_stale(&info_current));
    }

    #[test]
    fn is_stale_same_build_hash() {
        let info = DaemonInfo {
            pid: 1,
            port: 8000,
            version: current_version(),
            started_at: 0.0,
            build_hash: crate::build_hash().to_string(),
            force_cpu: false,
        };
        assert!(!is_stale(&info));
    }

    #[test]
    fn read_daemon_info_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_daemon_info_for(DaemonProfile::Main, dir.path()).is_none());
    }

    #[test]
    fn read_daemon_info_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("daemon.json"), "not json at all").unwrap();
        assert!(read_daemon_info_for(DaemonProfile::Main, dir.path()).is_none());
    }

    #[test]
    fn read_daemon_info_missing_version_field() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("daemon.json"),
            r#"{"pid": 999, "port": 8000}"#,
        )
        .unwrap();
        let info = read_daemon_info_for(DaemonProfile::Main, dir.path()).unwrap();
        assert_eq!(info.pid, 999);
        assert_eq!(info.port, 8000);
        // version defaults to "" via serde(default)
        assert_eq!(info.version, "");
        assert_eq!(info.build_hash, "");
        assert!(!info.force_cpu);
    }

    #[test]
    fn write_read_daemon_info_roundtrip() {
        for profile in [DaemonProfile::Main, DaemonProfile::Sidecar] {
            let dir = tempfile::tempdir().unwrap();
            write_daemon_info_for(profile, dir.path(), 42, 9999, true).unwrap();
            let info = read_daemon_info_for(profile, dir.path()).unwrap();
            assert_eq!(info.pid, 42);
            assert_eq!(info.port, 9999);
            assert_eq!(info.version, current_version());
            assert_eq!(info.build_hash, crate::build_hash());
            assert!(info.force_cpu);
        }
    }

    #[test]
    fn cleanup_state_file_removes() {
        let dir = tempfile::tempdir().unwrap();
        write_daemon_info_for(DaemonProfile::Main, dir.path(), 1, 8000, false).unwrap();
        assert!(read_daemon_info_for(DaemonProfile::Main, dir.path()).is_some());
        cleanup_state_file_for(DaemonProfile::Main, dir.path());
        assert!(read_daemon_info_for(DaemonProfile::Main, dir.path()).is_none());
    }

    #[test]
    fn runtime_mismatch_detects_force_cpu_changes() {
        let info = DaemonInfo {
            pid: 1,
            port: 8000,
            version: current_version(),
            started_at: 0.0,
            build_hash: crate::build_hash().to_string(),
            force_cpu: false,
        };
        assert!(runtime_mismatch(&info, true));
        assert!(!runtime_mismatch(&info, false));
    }

    #[test]
    fn cleanup_state_file_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        // Cleaning up a nonexistent file should not panic
        cleanup_state_file_for(DaemonProfile::Main, dir.path());
        cleanup_state_file_for(DaemonProfile::Main, dir.path());
    }

    #[test]
    fn is_process_alive_current_pid() {
        let pid = std::process::id();
        assert!(is_process_alive(pid));
    }

    #[test]
    fn config_port_returns_default() {
        let dir = tempfile::tempdir().unwrap();
        let layout = RuntimeLayout::from_state_dir(dir.path().join("state"));
        let port = config_port(&layout).unwrap();
        assert!(port > 0);
    }
}
