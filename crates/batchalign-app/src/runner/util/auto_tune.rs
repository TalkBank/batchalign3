//! Per-job file parallelism auto-tuning and media constants.

use crate::api::{CommandName, MemoryMb, NumWorkers};
use crate::config::ServerConfig;
use crate::runtime;

/// Known audio/video file extensions for media pre-validation.
pub(crate) const KNOWN_MEDIA_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "mp4", "m4a", "flac", "ogg", "aac", "wma", "webm",
];

/// Per-command memory budget (MB) for each concurrent worker.
/// Uses free-threaded (thread worker) values from `runtime_constants.toml`.
pub(in crate::runner) fn command_base_mb(command: &CommandName) -> MemoryMb {
    let map = runtime::command_base_mb_threaded();
    map.get(command.as_ref())
        .copied()
        .unwrap_or(runtime::default_base_mb())
}

/// Compute the number of parallel file workers for a job.
///
/// Matches batchalign-next's `_server_auto_tune_workers()`: memory-based
/// scaling for all commands, with a per-category cap (GPU/process/thread).
/// GPU commands are NOT hardcoded to 1 — they use the same memory formula
/// but are capped at `max_gpu_workers` (default 8) from runtime constants.
pub(in crate::runner) fn compute_job_workers(
    command: &CommandName,
    num_files: usize,
    config: &ServerConfig,
) -> NumWorkers {
    if num_files <= 1 {
        return NumWorkers(1);
    }

    if config.max_workers_per_job > 0 {
        return NumWorkers(
            (config.max_workers_per_job as usize)
                .min(num_files)
                .clamp(1, runtime::max_thread_workers()),
        );
    }

    let is_gpu_heavy = runtime::gpu_heavy_commands()
        .iter()
        .any(|c| c == command.as_ref());

    // Auto-tune based on available memory and CPU
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let available_mb = sys.available_memory() / (1024 * 1024);

    let base_mb = command_base_mb(command);
    let budget_per_worker = (base_mb.0 as f64 * runtime::loading_overhead()) as u64;

    let by_memory = if budget_per_worker > 0 {
        (available_mb / budget_per_worker) as usize
    } else {
        4
    };

    let by_cpu = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    // Apply per-category cap: GPU commands share one model process so the
    // cap reflects GPU contention, not RAM.  BA-next used MAX_GPU_WORKERS=8.
    let category_cap = if is_gpu_heavy {
        runtime::max_gpu_workers()
    } else {
        runtime::max_thread_workers()
    };

    NumWorkers(num_files.min(by_cpu).min(by_memory).clamp(1, category_cap))
}
