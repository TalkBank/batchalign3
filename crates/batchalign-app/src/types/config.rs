//! Server configuration — mirrors `batchalign/serve/config.py`.
//!
//! Deserializes from the runtime-owned `server.yaml` under the resolved state
//! directory using serde_yaml.
//! No OmegaConf interpolation needed — plain YAML is sufficient.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::api::{LanguageCode3, MemoryMb};

/// Runtime-owned filesystem layout resolved from env/home defaults at startup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeLayout {
    state_dir: PathBuf,
    config_path: PathBuf,
}

impl RuntimeLayout {
    /// Resolve the runtime layout from ambient environment variables.
    pub fn from_env() -> Self {
        Self::from_sources(
            std::env::var("BATCHALIGN_STATE_DIR").ok().as_deref(),
            std::env::var("HOME").ok().as_deref(),
        )
    }

    /// Resolve the runtime layout from explicit state-dir and home-dir sources.
    pub fn from_sources(state_dir_env: Option<&str>, home_env: Option<&str>) -> Self {
        let state_dir = state_dir_env
            .map(str::trim)
            .filter(|dir| !dir.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                // Documented default: fall back to home dir or platform temp dir.
                let base = home_env
                    .map(PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir);
                base.join(".batchalign3")
            });
        Self::from_state_dir(state_dir)
    }

    /// Build the runtime layout from an explicit state directory.
    pub fn from_state_dir(state_dir: PathBuf) -> Self {
        let config_path = state_dir.join("server.yaml");
        Self {
            state_dir,
            config_path,
        }
    }

    /// Runtime state directory (jobs, DB, daemon metadata, logs, config).
    pub fn state_dir(&self) -> &Path {
        &self.state_dir
    }

    /// Default server config path under the runtime state directory.
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    /// Runtime jobs directory under the owned state root.
    pub fn jobs_dir(&self) -> PathBuf {
        self.state_dir.join("jobs")
    }

    /// Runtime logs directory under the owned state root.
    pub fn logs_dir(&self) -> PathBuf {
        self.state_dir.join("logs")
    }

    /// Runtime bug-report directory under the owned state root.
    pub fn bug_reports_dir(&self) -> PathBuf {
        self.state_dir.join("bug-reports")
    }

    /// Runtime dashboard asset directory under the owned state root.
    pub fn dashboard_dir(&self) -> PathBuf {
        self.state_dir.join("dashboard")
    }

    /// Server PID file under the owned state root.
    pub fn server_pid_path(&self) -> PathBuf {
        self.state_dir.join("server.pid")
    }

    /// Server stderr log file under the owned state root.
    pub fn server_log_path(&self) -> PathBuf {
        self.state_dir.join("server.log")
    }
}

/// Default config file path.
pub fn default_config_path() -> PathBuf {
    RuntimeLayout::from_env().config_path().to_path_buf()
}

/// State directory: `$BATCHALIGN_STATE_DIR` if set, else `$HOME/.batchalign3`.
pub fn ba_state_dir() -> PathBuf {
    RuntimeLayout::from_env().state_dir().to_path_buf()
}

/// Minimal warmup preset — morphotag only.
///
/// The CLI `--warmup minimal` expands to this list.
pub const WARMUP_PRESET_MINIMAL: &[&str] = &["morphotag"];

/// Full warmup preset — morphotag, align, transcribe.
///
/// The CLI `--warmup full` expands to this list. This is also the default
/// when no `--warmup` flag or `warmup_commands` config is given.
pub const WARMUP_PRESET_FULL: &[&str] = &["morphotag", "align", "transcribe"];

/// Configuration for the Batchalign processing server.
///
/// Deserialized from the runtime-owned `server.yaml`. All fields have sensible
/// defaults so an empty YAML file (or a missing file) produces a working
/// configuration.  The [`validate`](Self::validate) method clamps out-of-range
/// values and returns non-fatal warnings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Filesystem directories the server searches when resolving media files
    /// for transcribe/align.  Paths that do not exist at startup produce a
    /// validation warning but are not fatal.
    #[serde(default)]
    pub media_roots: Vec<String>,
    /// Named media directory mappings (e.g. `{"childes-data": "/nfs/childes"}`).
    /// Clients reference the key in `JobSubmission.media_mapping`; the server
    /// resolves it to the filesystem root.  Allows stable logical names even
    /// when mount paths change.
    #[serde(default)]
    pub media_mappings: BTreeMap<String, String>,
    /// 3-letter ISO language code used when the client omits `lang`.
    /// Defaults to `"eng"`.
    #[serde(default = "default_lang")]
    pub default_lang: LanguageCode3,
    /// Maximum number of jobs processed in parallel.  `0` (default) means
    /// auto-tune from the resolved memory tier plus CPU availability.
    /// Must be >= 0; negative values are clamped to 0 by `validate()`.
    #[serde(default)]
    pub max_concurrent_jobs: i32,
    /// TCP port for the HTTP server.  Must be 1..=65535; 0 is clamped to
    /// 8000 by `validate()`.  Default: 8000.
    #[serde(default = "default_port")]
    pub port: u16,
    /// Bind address for the HTTP server.  Default: `"0.0.0.0"` (all interfaces).
    #[serde(default = "default_host")]
    pub host: String,
    /// Maximum Python worker processes per job.  `0` (default) means
    /// auto-tune based on RAM and GPU availability.
    #[serde(default)]
    pub max_workers_per_job: i32,
    /// Number of days to retain completed/failed job metadata in SQLite
    /// before automatic purge.  Must be >= 1; values < 1 are clamped to 1
    /// by `validate()`.  Default: 7.
    #[serde(default = "default_job_ttl_days")]
    pub job_ttl_days: i32,
    /// Redis connection URL for shared utterance cache
    /// (e.g. `"redis://net:6379/0"`).  Empty string (default) disables
    /// Redis caching and uses SQLite-only.
    #[serde(default)]
    pub redis_url: String,
    /// Commands to pre-warm at startup (e.g. `["morphotag", "align"]`).
    /// When empty (default), the server uses the `full` preset
    /// (`morphotag`, `align`, `transcribe`), filtered by actual worker
    /// capabilities.  The CLI `--warmup` flag sets this field.
    #[serde(default = "default_warmup_commands")]
    pub warmup_commands: Vec<String>,
    /// Whether the CLI should auto-spawn a local daemon when no explicit
    /// `--server` is configured. Default: `true`.
    #[serde(default = "default_true")]
    pub auto_daemon: bool,
    /// Minimum host headroom (MB) the coordinator keeps free after worker-start
    /// and job-execution reservations. 0 = disable host-memory headroom checks.
    /// Default: 8192.
    #[serde(default = "default_memory_gate_mb")]
    pub memory_gate_mb: MemoryMb,
    /// Seconds of inactivity before a worker is shut down. 0 = use pool default (600).
    #[serde(default = "default_worker_idle_timeout_s")]
    pub worker_idle_timeout_s: u64,
    /// Seconds between worker health checks. 0 = use pool default (30).
    #[serde(default = "default_worker_health_interval_s")]
    pub worker_health_interval_s: u64,

    /// Maximum number of local worker/model startups allowed at once across all
    /// participating batchalign3 processes on the host. Default: 1.
    #[serde(default = "default_max_concurrent_worker_startups")]
    pub max_concurrent_worker_startups: u32,

    /// Maximum Python worker processes per (profile, lang, engine) key.
    /// 0 = use built-in default (8). Reduces GPU memory pressure on smaller machines.
    #[serde(default)]
    pub max_workers_per_key: i32,

    /// Hard ceiling on total workers across all keys. Prevents OOM when many
    /// different (profile, lang, engine) keys are active simultaneously.
    /// 0 = auto-compute from available RAM (available_memory / 4GB, capped at 32).
    #[serde(default)]
    pub max_total_workers: i32,

    /// Seconds to wait for a Python worker to become ready after spawn.
    /// Default: 120.
    #[serde(default = "default_worker_ready_timeout_s")]
    pub worker_ready_timeout_s: u64,

    /// Maximum HTTP request body size in megabytes. Default: 100.
    #[serde(default = "default_max_body_bytes_mb")]
    pub max_body_bytes_mb: MemoryMb,

    /// Seconds to wait for host-memory reservations to fit before rejecting or
    /// deferring a job. Default: 120. 0 = reject immediately if no plan fits.
    #[serde(default = "default_memory_gate_timeout_s")]
    pub memory_gate_timeout_s: u64,

    /// Seconds between host-memory reservation polling checks. Default: 5.
    #[serde(default = "default_memory_gate_poll_s")]
    pub memory_gate_poll_s: u64,

    /// Low-memory warning threshold in MB. Default: 4096.
    #[serde(default = "default_memory_warning_mb")]
    pub memory_warning_mb: MemoryMb,

    /// Number of threads in the GPU worker's thread pool for concurrent
    /// inference requests. Default: 4.
    #[serde(default = "default_gpu_thread_pool_size")]
    pub gpu_thread_pool_size: u32,

    /// Seconds before a locally-dispatched file lease is considered orphaned.
    /// Default: 300.
    #[serde(default = "default_local_lease_ttl_s")]
    pub local_lease_ttl_s: u64,

    /// Timeout in seconds for audio-heavy worker tasks (ASR, FA, speaker).
    /// 0 = use built-in default (1800). Increase for very long recordings.
    #[serde(default)]
    pub audio_task_timeout_s: u64,

    /// Timeout in seconds for lightweight analysis tasks (OpenSMILE, AVQI).
    /// 0 = use built-in default (120).
    #[serde(default)]
    pub analysis_task_timeout_s: u64,

    /// Path to the worker registry file for discovering pre-started TCP
    /// workers. Empty string (default) uses `~/.batchalign3/workers.json`.
    #[serde(default)]
    pub worker_registry_path: String,

    /// Override the auto-detected memory tier. When absent, the tier is
    /// detected from total system RAM. Valid values: `"small"`, `"medium"`,
    /// `"large"`, `"fleet"`. This overrides all tier-derived defaults
    /// (headroom, startup reservations, idle timeout, max workers) unless
    /// those fields are also explicitly set.
    #[serde(default)]
    pub memory_tier: Option<String>,

    /// Override GPU worker startup reservation (MB). 0 = use tier default.
    #[serde(default)]
    pub gpu_startup_mb: u64,

    /// Override Stanza worker startup reservation (MB). 0 = use tier default.
    #[serde(default)]
    pub stanza_startup_mb: u64,

    /// Override IO worker startup reservation (MB). 0 = use tier default.
    #[serde(default)]
    pub io_startup_mb: u64,
}

fn default_lang() -> LanguageCode3 {
    LanguageCode3::eng()
}

fn default_port() -> u16 {
    8000
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_true() -> bool {
    true
}

fn default_warmup_commands() -> Vec<String> {
    WARMUP_PRESET_FULL
        .iter()
        .map(|s| (*s).to_string())
        .collect()
}

fn default_job_ttl_days() -> i32 {
    7
}

fn default_memory_gate_mb() -> MemoryMb {
    crate::types::runtime::MemoryTier::detect().headroom_mb
}

fn default_worker_idle_timeout_s() -> u64 {
    crate::types::runtime::MemoryTier::detect().idle_timeout_s
}

fn default_worker_health_interval_s() -> u64 {
    30
}

fn default_max_concurrent_worker_startups() -> u32 {
    1
}

fn default_worker_ready_timeout_s() -> u64 {
    300
}

fn default_max_body_bytes_mb() -> MemoryMb {
    MemoryMb(100)
}

fn default_memory_gate_timeout_s() -> u64 {
    120
}

fn default_memory_gate_poll_s() -> u64 {
    5
}

fn default_memory_warning_mb() -> MemoryMb {
    MemoryMb(4096)
}

fn default_gpu_thread_pool_size() -> u32 {
    4
}

fn default_local_lease_ttl_s() -> u64 {
    300
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            media_roots: Vec::new(),
            media_mappings: BTreeMap::new(),
            default_lang: LanguageCode3::eng(),
            max_concurrent_jobs: 0,
            port: 8000,
            host: "0.0.0.0".to_string(),
            max_workers_per_job: 0,
            job_ttl_days: 7,
            redis_url: String::new(),
            warmup_commands: default_warmup_commands(),
            auto_daemon: true,
            memory_gate_mb: default_memory_gate_mb(),
            worker_idle_timeout_s: default_worker_idle_timeout_s(),
            worker_health_interval_s: default_worker_health_interval_s(),
            max_concurrent_worker_startups: default_max_concurrent_worker_startups(),
            max_workers_per_key: 0,
            max_total_workers: 0,
            worker_ready_timeout_s: default_worker_ready_timeout_s(),
            max_body_bytes_mb: default_max_body_bytes_mb(),
            memory_gate_timeout_s: default_memory_gate_timeout_s(),
            memory_gate_poll_s: default_memory_gate_poll_s(),
            memory_warning_mb: default_memory_warning_mb(),
            gpu_thread_pool_size: default_gpu_thread_pool_size(),
            local_lease_ttl_s: default_local_lease_ttl_s(),
            audio_task_timeout_s: 0,
            analysis_task_timeout_s: 0,
            worker_registry_path: String::new(),
            memory_tier: None,
            gpu_startup_mb: 0,
            stanza_startup_mb: 0,
            io_startup_mb: 0,
        }
    }
}

impl ServerConfig {
    /// Resolve the effective memory tier, applying config overrides.
    ///
    /// Priority: explicit `memory_tier` field in config → auto-detect from RAM.
    /// Individual startup reservation overrides (`gpu_startup_mb`, etc.) are
    /// applied on top of the resolved tier.
    pub fn resolved_memory_tier(&self) -> crate::types::runtime::MemoryTier {
        use crate::api::MemoryMb;
        use crate::types::runtime::{MemoryTier, MemoryTierKind};

        let mut tier = match self.memory_tier.as_deref() {
            Some("small") => MemoryTier::from_total_mb(16_000),
            Some("medium") => MemoryTier::from_total_mb(32_000),
            Some("large") => MemoryTier::from_total_mb(64_000),
            Some("fleet") => MemoryTier::from_total_mb(256_000),
            _ => MemoryTier::detect(),
        };
        // Apply individual overrides (0 = use tier default).
        if self.gpu_startup_mb > 0 {
            tier.gpu_startup_mb = MemoryMb(self.gpu_startup_mb);
        }
        if self.stanza_startup_mb > 0 {
            tier.stanza_startup_mb = MemoryMb(self.stanza_startup_mb);
        }
        if self.io_startup_mb > 0 {
            tier.io_startup_mb = MemoryMb(self.io_startup_mb);
        }
        // If memory_tier was explicitly set, also override kind for logging.
        if let Some(ref name) = self.memory_tier {
            tier.kind = match name.as_str() {
                "small" => MemoryTierKind::Small,
                "medium" => MemoryTierKind::Medium,
                "large" => MemoryTierKind::Large,
                "fleet" => MemoryTierKind::Fleet,
                _ => tier.kind,
            };
        }
        tier
    }

    /// Resolve host-memory headroom, honoring a memory-tier override when the
    /// stored field still matches the machine-detected default.
    pub fn resolved_memory_gate_mb(&self) -> crate::api::MemoryMb {
        let detected_default = crate::types::runtime::MemoryTier::detect().headroom_mb;
        if self.memory_gate_mb.0 == 0 {
            return self.resolved_memory_tier().headroom_mb;
        }
        if self.memory_tier.is_some() && self.memory_gate_mb == detected_default {
            return self.resolved_memory_tier().headroom_mb;
        }
        self.memory_gate_mb
    }

    /// Resolve worker idle timeout, honoring a memory-tier override when the
    /// stored field still matches the machine-detected default.
    pub fn resolved_worker_idle_timeout_s(&self) -> u64 {
        let detected_default = crate::types::runtime::MemoryTier::detect().idle_timeout_s;
        if self.worker_idle_timeout_s == 0 {
            return self.resolved_memory_tier().idle_timeout_s;
        }
        if self.memory_tier.is_some() && self.worker_idle_timeout_s == detected_default {
            return self.resolved_memory_tier().idle_timeout_s;
        }
        self.worker_idle_timeout_s
    }

    /// Resolve warmup commands before server-side capability filtering.
    ///
    /// Returns `warmup_commands` directly — the CLI `--warmup` flag and
    /// `server.yaml` both write to this field. An empty list means no warmup.
    pub fn resolved_warmup_commands(&self) -> &[String] {
        &self.warmup_commands
    }

    /// Return a list of warnings (non-fatal) about the config.
    pub fn validate(&mut self) -> Vec<String> {
        let mut warnings = Vec::new();

        for root in &self.media_roots {
            if !Path::new(root).is_dir() {
                warnings.push(format!("media_root does not exist: {root}"));
            }
        }
        for (key, root) in &self.media_mappings {
            if !Path::new(root).is_dir() {
                warnings.push(format!("media_mapping '{key}' root does not exist: {root}"));
            }
        }
        if self.max_concurrent_jobs < 0 {
            warnings.push("max_concurrent_jobs must be >= 0 (0=auto), defaulting to 0".into());
            self.max_concurrent_jobs = 0;
        }
        if self.port == 0 {
            warnings.push(format!(
                "port must be 1-65535 (got {}), defaulting to 8000",
                self.port
            ));
            self.port = 8000;
        }
        if self.job_ttl_days < 1 {
            warnings.push(format!(
                "job_ttl_days must be >= 1 (got {}), defaulting to 1",
                self.job_ttl_days
            ));
            self.job_ttl_days = 1;
        }
        if self.memory_gate_poll_s == 0 {
            warnings.push("memory_gate_poll_s must be >= 1, defaulting to 1".into());
            self.memory_gate_poll_s = 1;
        }
        if self.max_concurrent_worker_startups == 0 {
            warnings.push("max_concurrent_worker_startups must be >= 1, defaulting to 1".into());
            self.max_concurrent_worker_startups = 1;
        }
        if self.gpu_thread_pool_size == 0 {
            warnings.push("gpu_thread_pool_size must be >= 1, defaulting to 1".into());
            self.gpu_thread_pool_size = 1;
        }
        warnings
    }
}

/// Load `ServerConfig` using an explicit runtime layout when no config path is
/// passed.
pub fn load_config_from_layout(
    layout: &RuntimeLayout,
    path: Option<&Path>,
) -> Result<ServerConfig, ConfigError> {
    let path = match path {
        Some(p) => p.to_path_buf(),
        None => layout.config_path().to_path_buf(),
    };

    if !path.exists() {
        return Ok(ServerConfig::default());
    }

    let contents = std::fs::read_to_string(&path).map_err(|e| ConfigError::Io(path.clone(), e))?;
    let config: ServerConfig = serde_yaml::from_str(&contents)
        .map_err(|e| ConfigError::Parse(path.clone(), e.to_string()))?;
    Ok(config)
}

/// Load [`ServerConfig`] and apply non-fatal validation/clamping.
///
/// Returns the validated config plus any warning messages produced by
/// [`ServerConfig::validate`]. Callers that need a working runtime config but
/// still want to surface bad values should prefer this helper.
pub fn load_validated_config_from_layout(
    layout: &RuntimeLayout,
    path: Option<&Path>,
) -> Result<(ServerConfig, Vec<String>), ConfigError> {
    let mut config = load_config_from_layout(layout, path)?;
    let warnings = config.validate();
    Ok((config, warnings))
}

/// Load ServerConfig from a YAML file. Falls back to defaults if the file
/// doesn't exist.
pub fn load_config(path: Option<&Path>) -> Result<ServerConfig, ConfigError> {
    let layout = RuntimeLayout::from_env();
    load_config_from_layout(&layout, path)
}

/// Errors that can occur when loading config.
///
/// Callers should distinguish between these variants to provide actionable
/// messages: `Io` typically means a permissions problem, while `Parse` means
/// the user has a syntax error in their YAML.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// The config file exists but could not be read (e.g. permission denied,
    /// I/O error).  Contains the path that was attempted and the underlying
    /// OS error.
    #[error("failed to read config at {0}: {1}")]
    Io(PathBuf, #[source] std::io::Error),
    /// The config file was read but its contents are not valid YAML or do
    /// not match the expected `ServerConfig` schema.  Contains the path and
    /// a human-readable parse error.
    #[error("failed to parse config at {0}: {1}")]
    Parse(PathBuf, String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.port, 8000);
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.default_lang, "eng"); // PartialEq<&str>
        assert_eq!(cfg.job_ttl_days, 7);
        assert!(cfg.auto_daemon);
        assert_eq!(cfg.worker_idle_timeout_s, 600);
        assert_eq!(cfg.worker_health_interval_s, 30);
        // memory_gate_mb is now tier-dependent (detect() reads system RAM)
        let tier = crate::types::runtime::MemoryTier::detect();
        assert_eq!(cfg.memory_gate_mb, tier.headroom_mb);
        assert_eq!(cfg.max_concurrent_worker_startups, 1);
        assert_eq!(cfg.max_workers_per_key, 0);
        assert_eq!(cfg.worker_ready_timeout_s, 300);
        assert_eq!(cfg.max_body_bytes_mb, MemoryMb(100));
        assert_eq!(cfg.memory_gate_timeout_s, 120);
        assert_eq!(cfg.memory_gate_poll_s, 5);
        assert_eq!(cfg.memory_warning_mb, MemoryMb(4096));
        assert_eq!(cfg.gpu_thread_pool_size, 4);
        assert_eq!(cfg.local_lease_ttl_s, 300);
        assert_eq!(cfg.audio_task_timeout_s, 0);
        assert_eq!(cfg.analysis_task_timeout_s, 0);
    }

    #[test]
    fn deserialize_yaml() {
        let yaml = r#"
media_roots:
  - /data/media
  - /data/media2
media_mappings:
  childes-data: /nfs/childes
default_lang: spa
port: 9000
max_concurrent_jobs: 4
warmup_commands:
  - morphotag
  - align
auto_daemon: true
"#;
        let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg.media_roots, vec!["/data/media", "/data/media2"]);
        assert_eq!(cfg.media_mappings["childes-data"], "/nfs/childes");
        assert_eq!(cfg.default_lang, "spa");
        assert_eq!(cfg.port, 9000);
        assert_eq!(cfg.max_concurrent_jobs, 4);
        assert_eq!(cfg.warmup_commands, vec!["morphotag", "align"]);
        assert!(cfg.auto_daemon);
    }

    #[test]
    fn deserialize_empty_yaml() {
        let yaml = "{}";
        let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(cfg, ServerConfig::default());
    }

    #[test]
    fn deserialize_rejects_unknown_fields() {
        let yaml = r#"
port: 9000
warmup: false
"#;
        let error = serde_yaml::from_str::<ServerConfig>(yaml).unwrap_err();
        assert!(
            error.to_string().contains("unknown field `warmup`"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn validate_fixes_bad_values() {
        let mut cfg = ServerConfig {
            max_concurrent_jobs: -1,
            job_ttl_days: 0,
            memory_gate_poll_s: 0,
            max_concurrent_worker_startups: 0,
            gpu_thread_pool_size: 0,
            ..Default::default()
        };
        let warnings = cfg.validate();
        assert_eq!(cfg.max_concurrent_jobs, 0);
        assert_eq!(cfg.job_ttl_days, 1);
        assert_eq!(cfg.memory_gate_poll_s, 1);
        assert_eq!(cfg.max_concurrent_worker_startups, 1);
        assert_eq!(cfg.gpu_thread_pool_size, 1);
        assert_eq!(warnings.len(), 5);
    }

    #[test]
    fn load_missing_file_returns_defaults() {
        let cfg = load_config(Some(Path::new("/nonexistent/server.yaml"))).unwrap();
        assert_eq!(cfg, ServerConfig::default());
    }

    #[test]
    fn load_validated_config_clamps_bad_values() {
        let dir = tempfile::tempdir().unwrap();
        let state_dir = dir.path().join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        let config_path = state_dir.join("server.yaml");
        std::fs::write(
            &config_path,
            "port: 0\njob_ttl_days: 0\nmemory_gate_poll_s: 0\nmax_concurrent_worker_startups: 0\ngpu_thread_pool_size: 0\n",
        )
        .unwrap();

        let layout = RuntimeLayout::from_state_dir(state_dir);
        let (cfg, warnings) = load_validated_config_from_layout(&layout, None).unwrap();
        assert_eq!(cfg.port, 8000);
        assert_eq!(cfg.job_ttl_days, 1);
        assert_eq!(cfg.memory_gate_poll_s, 1);
        assert_eq!(cfg.max_concurrent_worker_startups, 1);
        assert_eq!(cfg.gpu_thread_pool_size, 1);
        assert_eq!(warnings.len(), 5);
    }

    #[test]
    fn runtime_layout_prefers_explicit_state_dir() {
        let layout =
            RuntimeLayout::from_sources(Some("/tmp/batchalign-state"), Some("/Users/test"));
        assert_eq!(layout.state_dir(), Path::new("/tmp/batchalign-state"));
        assert_eq!(
            layout.config_path(),
            Path::new("/tmp/batchalign-state/server.yaml")
        );
    }

    #[test]
    fn runtime_layout_falls_back_to_home_dir() {
        let layout = RuntimeLayout::from_sources(None, Some("/Users/test"));
        assert_eq!(layout.state_dir(), Path::new("/Users/test/.batchalign3"));
        assert_eq!(
            layout.config_path(),
            Path::new("/Users/test/.batchalign3/server.yaml")
        );
    }

    #[test]
    fn runtime_layout_derives_owned_subpaths() {
        let layout = RuntimeLayout::from_state_dir(PathBuf::from("/tmp/batchalign-state"));
        assert_eq!(
            layout.jobs_dir(),
            PathBuf::from("/tmp/batchalign-state/jobs")
        );
        assert_eq!(
            layout.logs_dir(),
            PathBuf::from("/tmp/batchalign-state/logs")
        );
        assert_eq!(
            layout.bug_reports_dir(),
            PathBuf::from("/tmp/batchalign-state/bug-reports")
        );
        assert_eq!(
            layout.dashboard_dir(),
            PathBuf::from("/tmp/batchalign-state/dashboard")
        );
        assert_eq!(
            layout.server_pid_path(),
            PathBuf::from("/tmp/batchalign-state/server.pid")
        );
        assert_eq!(
            layout.server_log_path(),
            PathBuf::from("/tmp/batchalign-state/server.log")
        );
    }

    #[test]
    fn runtime_layout_load_config_uses_layout_config_path() {
        let dir = tempfile::tempdir().unwrap();
        let layout = RuntimeLayout::from_state_dir(dir.path().join("state"));
        std::fs::create_dir_all(layout.state_dir()).unwrap();
        std::fs::write(layout.config_path(), "port: 9123\n").unwrap();

        let cfg = load_config_from_layout(&layout, None).unwrap();
        assert_eq!(cfg.port, 9123);
    }

    #[test]
    fn roundtrip_json() {
        let cfg = ServerConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    // ---- Memory tier overrides ----

    #[test]
    fn resolved_tier_defaults_to_detect() {
        let cfg = ServerConfig::default();
        let tier = cfg.resolved_memory_tier();
        let detected = crate::types::runtime::MemoryTier::detect();
        assert_eq!(tier.kind, detected.kind);
    }

    #[test]
    fn resolved_tier_override_small() {
        let cfg = ServerConfig {
            memory_tier: Some("small".to_string()),
            ..Default::default()
        };
        let tier = cfg.resolved_memory_tier();
        assert_eq!(tier.kind, crate::types::runtime::MemoryTierKind::Small);
        assert_eq!(tier.headroom_mb.0, 2_000);
        assert_eq!(tier.stanza_startup_mb.0, 3_000);
    }

    #[test]
    fn resolved_tier_individual_startup_overrides() {
        let cfg = ServerConfig {
            stanza_startup_mb: 5_000,
            gpu_startup_mb: 8_000,
            ..Default::default()
        };
        let tier = cfg.resolved_memory_tier();
        assert_eq!(tier.stanza_startup_mb.0, 5_000);
        assert_eq!(tier.gpu_startup_mb.0, 8_000);
        // IO should be unchanged (0 = use tier default)
        assert!(tier.io_startup_mb.0 > 0);
    }

    #[test]
    fn resolved_tier_yaml_with_overrides() {
        let yaml = r#"
memory_tier: small
stanza_startup_mb: 4000
"#;
        let cfg: ServerConfig = serde_yaml::from_str(yaml).unwrap();
        let tier = cfg.resolved_memory_tier();
        assert_eq!(tier.kind, crate::types::runtime::MemoryTierKind::Small);
        // Stanza overridden from small default (3000) to 4000
        assert_eq!(tier.stanza_startup_mb.0, 4_000);
        // GPU stays at small default
        assert_eq!(tier.gpu_startup_mb.0, 6_000);
    }

    #[test]
    fn resolved_memory_gate_uses_tier_override_when_headroom_is_default() {
        let cfg = ServerConfig {
            memory_tier: Some("small".to_string()),
            ..Default::default()
        };
        assert_eq!(cfg.resolved_memory_gate_mb().0, 2_000);
    }

    #[test]
    fn resolved_worker_idle_timeout_uses_tier_override_when_default_matches_detected() {
        let cfg = ServerConfig {
            memory_tier: Some("small".to_string()),
            ..Default::default()
        };
        assert_eq!(cfg.resolved_worker_idle_timeout_s(), 60);
    }

    #[test]
    fn explicit_headroom_and_idle_timeout_override_memory_tier_defaults() {
        let cfg = ServerConfig {
            memory_tier: Some("small".to_string()),
            memory_gate_mb: crate::api::MemoryMb(9_999),
            worker_idle_timeout_s: 777,
            ..Default::default()
        };
        assert_eq!(cfg.resolved_memory_gate_mb().0, 9_999);
        assert_eq!(cfg.resolved_worker_idle_timeout_s(), 777);
    }
}
