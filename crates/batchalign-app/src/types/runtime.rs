//! Runtime constants — loaded from `runtime_constants.toml` at compile time.
//!
//! Command-to-task mapping, memory budgets, and command classification.
//! The TOML file is the single source of truth shared with Python.

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

/// Raw TOML content, embedded at compile time.
const TOML_SRC: &str = include_str!("../../../../batchalign/runtime_constants.toml");

/// Parsed TOML structure.
#[derive(Deserialize)]
struct RuntimeConstants {
    cmd2task: HashMap<String, String>,
    worker_caps: WorkerCaps,
    memory: MemoryConstants,
    gpu_heavy_commands: GpuHeavy,
    process_commands: ProcessCommands,
    command_base_mb: CommandBaseMb,
    known_engine_keys: KnownEngineKeys,
}

#[derive(Deserialize)]
struct WorkerCaps {
    max_gpu_workers: usize,
    max_process_workers: usize,
    max_thread_workers: usize,
}

#[derive(Deserialize)]
struct MemoryConstants {
    default_base_mb: u64,
    mb_per_file_mb: u64,
    loading_overhead: f64,
}

#[derive(Deserialize)]
struct GpuHeavy {
    commands: Vec<String>,
}

#[derive(Deserialize)]
struct ProcessCommands {
    gil: Vec<String>,
    free_threaded: Vec<String>,
}

#[derive(Deserialize)]
struct CommandBaseMb {
    process: HashMap<String, u64>,
    threaded: HashMap<String, u64>,
}

#[derive(Deserialize)]
struct KnownEngineKeys {
    keys: Vec<String>,
}

static CONSTANTS: LazyLock<RuntimeConstants> =
    LazyLock::new(|| toml::from_str(TOML_SRC).expect("runtime_constants.toml must be valid TOML"));

/// Command name -> pipeline task string (e.g. "align" -> "fa").
pub fn cmd2task() -> HashMap<&'static str, &'static str> {
    CONSTANTS
        .cmd2task
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect()
}

/// GPU-bound commands where MPS/CUDA is the bottleneck.
pub fn gpu_heavy_commands() -> &'static [String] {
    &CONSTANTS.gpu_heavy_commands.commands
}

/// Known engine-override keys (passed via --engine-overrides).
pub fn known_engine_keys() -> &'static [String] {
    &CONSTANTS.known_engine_keys.keys
}

/// Hard cap on concurrent GPU-bound workers (transcribe, align, benchmark).
pub fn max_gpu_workers() -> usize {
    CONSTANTS.worker_caps.max_gpu_workers
}

/// Hard cap on concurrent process-isolated workers (non-free-threaded Python).
pub fn max_process_workers() -> usize {
    CONSTANTS.worker_caps.max_process_workers
}

/// Hard cap on concurrent thread workers (free-threaded Python 3.14t+).
pub fn max_thread_workers() -> usize {
    CONSTANTS.worker_caps.max_thread_workers
}

/// Per-command base memory (MB) — non-free-threaded (process workers).
pub fn command_base_mb_process() -> HashMap<&'static str, u64> {
    CONSTANTS
        .command_base_mb
        .process
        .iter()
        .map(|(k, &v)| (k.as_str(), v))
        .collect()
}

/// Per-command base memory (MB) — free-threaded (thread workers, shared models).
pub fn command_base_mb_threaded() -> HashMap<&'static str, u64> {
    CONSTANTS
        .command_base_mb
        .threaded
        .iter()
        .map(|(k, &v)| (k.as_str(), v))
        .collect()
}

/// Fallback per-worker memory budget (MB) when a command is not listed.
pub fn default_base_mb() -> u64 {
    CONSTANTS.memory.default_base_mb
}

/// Additional memory budget (MB) allocated per file queued to a worker.
pub fn mb_per_file_mb() -> u64 {
    CONSTANTS.memory.mb_per_file_mb
}

/// Multiplier applied to the static memory budget to account for transient
/// allocation spikes during model loading.
pub fn loading_overhead() -> f64 {
    CONSTANTS.memory.loading_overhead
}

/// CPU-bound commands that need process isolation (non-free-threaded).
pub fn process_commands_gil() -> &'static [String] {
    &CONSTANTS.process_commands.gil
}

/// CPU-bound commands that need process isolation (free-threaded).
pub fn process_commands_free_threaded() -> &'static [String] {
    &CONSTANTS.process_commands.free_threaded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toml_parses_successfully() {
        // Force LazyLock initialization — panics if TOML is malformed.
        let _ = cmd2task();
    }

    #[test]
    fn cmd2task_contains_core_commands() {
        let map = cmd2task();
        assert_eq!(map["align"], "fa");
        assert_eq!(map["morphotag"], "morphosyntax");
        assert_eq!(map["transcribe"], "asr");
    }

    #[test]
    fn worker_caps_are_positive() {
        assert!(max_gpu_workers() > 0);
        assert!(max_process_workers() > 0);
        assert!(max_thread_workers() > 0);
    }

    #[test]
    fn memory_constants_are_sane() {
        assert!(default_base_mb() > 0);
        assert!(mb_per_file_mb() > 0);
        assert!(loading_overhead() > 1.0);
    }

    #[test]
    fn gpu_heavy_non_empty() {
        assert!(!gpu_heavy_commands().is_empty());
    }

    #[test]
    fn process_commands_non_empty() {
        assert!(!process_commands_gil().is_empty());
        assert!(!process_commands_free_threaded().is_empty());
    }

    #[test]
    fn command_base_mb_has_all_commands() {
        let proc = command_base_mb_process();
        let thread = command_base_mb_threaded();
        // Both maps should have the same keys
        let mut proc_keys: Vec<_> = proc.keys().collect();
        let mut thread_keys: Vec<_> = thread.keys().collect();
        proc_keys.sort();
        thread_keys.sort();
        assert_eq!(proc_keys, thread_keys);
    }
}
