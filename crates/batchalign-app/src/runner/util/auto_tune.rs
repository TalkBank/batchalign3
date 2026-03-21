//! Per-job file parallelism auto-tuning and media constants.

use crate::api::{CommandName, NumWorkers};
use crate::config::ServerConfig;
use crate::runtime;

/// Known audio/video file extensions for media pre-validation.
pub(crate) const KNOWN_MEDIA_EXTENSIONS: &[&str] = &[
    "wav", "mp3", "mp4", "m4a", "flac", "ogg", "aac", "wma", "webm",
];

/// Compute the number of parallel file workers for a job.
///
/// This function intentionally does **not** do host-memory math anymore. It
/// only applies file-count, operator-configured, CPU, and per-category caps.
/// Host-wide memory clamping now happens in the coordinator-backed admission
/// step so worker startup and job execution share one memory model.
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

    let by_cpu = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    // Apply per-category cap: GPU commands share one model process and should
    // not dispatch more in-process requests than the configured GPU thread
    // pool intends to serve.
    let category_cap = if is_gpu_heavy {
        runtime::max_gpu_workers().min(config.gpu_thread_pool_size as usize)
    } else {
        runtime::max_thread_workers()
    };

    NumWorkers(num_files.min(by_cpu).clamp(1, category_cap))
}
