//! Worker registry — discover pre-started TCP workers from `workers.json`.
//!
//! The registry file is the bridge between independently started worker daemons
//! and the Rust server. Python workers write their entries on startup (via
//! `_registry.py`); the server reads and health-checks them on startup and
//! periodically.
//!
//! Registry path: `~/.batchalign3/workers.json` (configurable via
//! [`ServerConfig::worker_registry_path`] or `BATCHALIGN_STATE_DIR`).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::api::LanguageCode3;
use crate::worker::tcp_handle::{TcpWorkerHandle, TcpWorkerInfo};
use crate::worker::{WorkerPid, WorkerProfile};

// ---------------------------------------------------------------------------
// Registry entry (JSON schema matches Python `WorkerRegistryEntry`)
// ---------------------------------------------------------------------------

/// One worker's entry in the `workers.json` registry file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntry {
    /// Worker process ID.
    pub pid: u32,
    /// Bind address (usually `"127.0.0.1"`).
    pub host: String,
    /// TCP port.
    pub port: u16,
    /// Worker profile name (`"gpu"`, `"stanza"`, `"io"`).
    pub profile: String,
    /// 3-letter language code.
    pub lang: String,
    /// Engine overrides JSON string (empty = none).
    #[serde(default)]
    pub engine_overrides: String,
    /// ISO 8601 timestamp when the worker started.
    #[serde(default)]
    pub started_at: String,
}

impl RegistryEntry {
    /// Parse the profile string into a [`WorkerProfile`].
    pub fn worker_profile(&self) -> Option<WorkerProfile> {
        match self.profile.as_str() {
            "gpu" => Some(WorkerProfile::Gpu),
            "stanza" => Some(WorkerProfile::Stanza),
            "io" => Some(WorkerProfile::Io),
            _ => None,
        }
    }
}

/// A discovered worker that has been health-checked and is ready for use.
#[derive(Debug, Clone)]
pub struct DiscoveredWorker {
    /// Registry entry data.
    pub entry: RegistryEntry,
    /// Parsed worker profile.
    pub profile: WorkerProfile,
    /// Parsed language code.
    pub lang: LanguageCode3,
}

// ---------------------------------------------------------------------------
// Registry file I/O
// ---------------------------------------------------------------------------

/// Default registry file path: `~/.batchalign3/workers.json`.
pub fn default_registry_path() -> PathBuf {
    if let Ok(state_dir) = std::env::var("BATCHALIGN_STATE_DIR") {
        let state_dir = state_dir.trim();
        if !state_dir.is_empty() {
            return PathBuf::from(state_dir).join("workers.json");
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".batchalign3")
        .join("workers.json")
}

/// Read all entries from the registry file.
pub fn read_registry(path: &Path) -> Vec<RegistryEntry> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            if e.kind() != std::io::ErrorKind::NotFound {
                warn!(path = %path.display(), error = %e, "Failed to read worker registry");
            }
            return Vec::new();
        }
    };

    match serde_json::from_str::<Vec<RegistryEntry>>(&content) {
        Ok(entries) => entries,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "Failed to parse worker registry");
            Vec::new()
        }
    }
}

/// Write entries back to the registry file (for removing stale entries).
fn write_registry(path: &Path, entries: &[RegistryEntry]) -> Result<(), std::io::Error> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_string_pretty(entries)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, data)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Remove a stale entry by PID (for crash cleanup).
pub fn remove_stale_entry(registry_path: &Path, pid: u32) -> bool {
    let entries = read_registry(registry_path);
    let before = entries.len();
    let remaining: Vec<RegistryEntry> = entries.into_iter().filter(|e| e.pid != pid).collect();
    if remaining.len() == before {
        return false;
    }
    if let Err(e) = write_registry(registry_path, &remaining) {
        warn!(error = %e, "Failed to write registry after stale removal");
    }
    true
}

/// Discover pre-started workers from the registry file.
///
/// Reads `workers.json`, connects to each entry, runs a health check, and
/// removes stale entries (workers that crashed without cleanup). Returns
/// only healthy, connectable workers.
pub async fn discover_workers(
    registry_path: &Path,
    audio_task_timeout_s: u64,
    analysis_task_timeout_s: u64,
) -> Vec<DiscoveredWorker> {
    let entries = read_registry(registry_path);
    if entries.is_empty() {
        return Vec::new();
    }

    info!(
        count = entries.len(),
        path = %registry_path.display(),
        "Checking worker registry"
    );

    let mut discovered = Vec::new();
    let mut stale_indices = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        let Some(profile) = entry.worker_profile() else {
            warn!(
                profile = %entry.profile,
                pid = entry.pid,
                "Unknown worker profile in registry, skipping"
            );
            stale_indices.push(i);
            continue;
        };

        let lang = match LanguageCode3::try_new(&entry.lang) {
            Ok(code) => code,
            Err(e) => {
                warn!(
                    lang = %entry.lang,
                    pid = entry.pid,
                    error = %e,
                    "Registry entry has invalid language code, skipping"
                );
                stale_indices.push(i);
                continue;
            }
        };
        let info = TcpWorkerInfo {
            host: entry.host.clone(),
            port: entry.port,
            profile,
            lang: lang.clone(),
            engine_overrides: entry.engine_overrides.clone(),
            pid: WorkerPid(entry.pid),
            audio_task_timeout_s,
            analysis_task_timeout_s,
        };

        match TcpWorkerHandle::connect(info).await {
            Ok(mut handle) => {
                match handle.health_check().await {
                    Ok(_) => {
                        info!(
                            profile = %entry.profile,
                            lang = %entry.lang,
                            host = %entry.host,
                            port = entry.port,
                            pid = entry.pid,
                            "Discovered healthy TCP worker"
                        );
                        discovered.push(DiscoveredWorker {
                            entry: entry.clone(),
                            profile,
                            lang: lang.clone(),
                        });
                    }
                    Err(e) => {
                        warn!(
                            host = %entry.host,
                            port = entry.port,
                            pid = entry.pid,
                            error = %e,
                            "TCP worker health check failed, marking stale"
                        );
                        stale_indices.push(i);
                    }
                }
                // Drop the handle — the pool will create its own connection.
                drop(handle);
            }
            Err(e) => {
                debug!(
                    host = %entry.host,
                    port = entry.port,
                    pid = entry.pid,
                    error = %e,
                    "Cannot connect to registered worker, marking stale"
                );
                stale_indices.push(i);
            }
        }
    }

    // Remove stale entries from the registry file.
    if !stale_indices.is_empty() {
        let remaining: Vec<RegistryEntry> = entries
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !stale_indices.contains(i))
            .map(|(_, e)| e)
            .collect();

        info!(
            removed = stale_indices.len(),
            remaining = remaining.len(),
            "Removed stale entries from worker registry"
        );

        if let Err(e) = write_registry(registry_path, &remaining) {
            warn!(error = %e, "Failed to update worker registry after stale removal");
        }
    }

    discovered
}
