//! Configuration types for spawning Python worker processes.

use crate::api::{LanguageCode3, NumSpeakers, WorkerLanguage};
use crate::host_memory::HostMemoryRuntimeConfig;
use crate::revai::load_revai_api_key;
use crate::worker::WorkerProfile;
use crate::worker::python::resolve_python_executable;

/// Runtime-owned launch inputs for one worker subprocess.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerRuntimeConfig {
    /// Whether the worker should force CPU-only model/device selection.
    pub force_cpu: bool,
    /// Optional Rev.AI key already resolved by the Rust control plane.
    pub revai_api_key: Option<String>,
    /// Maximum concurrent requests served inside one GPU worker process.
    pub gpu_thread_pool_size: u32,
    /// Host-memory coordination settings shared with the worker spawn path.
    pub host_memory: HostMemoryRuntimeConfig,
}

impl Default for WorkerRuntimeConfig {
    fn default() -> Self {
        Self::from_sources(
            false,
            load_revai_api_key()
                .ok()
                .map(|key| key.as_str().to_string()),
            crate::config::ServerConfig::default().gpu_thread_pool_size,
            HostMemoryRuntimeConfig::default(),
        )
    }
}

impl WorkerRuntimeConfig {
    /// Build worker runtime inputs from explicit sources.
    pub fn from_sources(
        force_cpu: bool,
        revai_api_key: Option<String>,
        gpu_thread_pool_size: u32,
        host_memory: HostMemoryRuntimeConfig,
    ) -> Self {
        Self {
            force_cpu,
            revai_api_key: revai_api_key
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty()),
            gpu_thread_pool_size,
            host_memory,
        }
    }
}

/// Configuration for spawning a worker.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Path to the Python executable (e.g. "python3", "/usr/bin/python3.14t").
    pub python_path: String,
    /// Worker profile describing which task group this worker owns.
    pub profile: WorkerProfile,
    /// Worker-runtime language string.
    pub lang: WorkerLanguage,
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
    /// Timeout override for audio-heavy tasks (ASR, FA, speaker).
    /// 0 = use built-in default (1800).
    pub audio_task_timeout_s: u64,
    /// Timeout override for lightweight analysis tasks (OpenSMILE, AVQI).
    /// 0 = use built-in default (120).
    pub analysis_task_timeout_s: u64,
    /// Test-only: artificial delay in milliseconds before each response.
    /// 0 = no delay. Only effective when `test_echo` is true.
    pub test_delay_ms: u64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            python_path: resolve_python_executable(),
            profile: WorkerProfile::Stanza,
            lang: WorkerLanguage::from(LanguageCode3::eng()),
            num_speakers: NumSpeakers(1),
            engine_overrides: String::new(),
            test_echo: false,
            ready_timeout_s: 300,
            verbose: 0,
            runtime: WorkerRuntimeConfig::default(),
            audio_task_timeout_s: 0,
            analysis_task_timeout_s: 0,
            test_delay_ms: 0,
        }
    }
}
