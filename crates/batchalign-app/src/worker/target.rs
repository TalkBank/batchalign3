//! Typed worker bootstrap targets for the Rust control plane.
//!
//! The released worker pool is now infer-task-only. Rust still reasons in terms
//! of top-level commands for warmup, memory checks, and scheduling, but Python
//! workers are always bootstrapped around one inference task.

use crate::api::CommandName;
use crate::workflow::command_workflow_descriptor;

use super::InferTask;

// ---------------------------------------------------------------------------
// WorkerProfile
// ---------------------------------------------------------------------------

/// Worker profile grouping related [`InferTask`]s into fewer processes.
///
/// Instead of spawning one worker per `InferTask`, profiles group related tasks
/// so that loaded models are shared within a single process:
///
/// - **Gpu**: ASR, FA, Speaker — GPU-bound models, concurrent via Python
///   `ThreadPoolExecutor` (PyTorch releases the GIL during CUDA kernels).
///   Max 1 process per (lang, engine_overrides) key.
/// - **Stanza**: Morphosyntax, Utseg, Coref — Stanza NLP processors, sequential
///   per process. Multiple processes for CPU parallelism (auto-tuned).
/// - **Io**: Translate, OpenSMILE, AVQI — lightweight API/library calls.
///   Max 1 process per key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProfile {
    /// GPU-bound models (ASR, FA, Speaker). Concurrent via threads inside one process.
    Gpu,
    /// Stanza NLP processors (Morphosyntax, Utseg, Coref). Multi-process for CPU parallelism.
    Stanza,
    /// Lightweight API/library calls (Translate, OpenSMILE, AVQI).
    Io,
}

impl WorkerProfile {
    /// Map one [`InferTask`] to its profile.
    pub fn for_task(task: InferTask) -> Self {
        match task {
            InferTask::Asr | InferTask::Fa | InferTask::Speaker => Self::Gpu,
            InferTask::Morphosyntax | InferTask::Utseg | InferTask::Coref => Self::Stanza,
            InferTask::Translate | InferTask::Opensmile | InferTask::Avqi => Self::Io,
        }
    }

    /// The string label used in logs and worker keys.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Gpu => "profile:gpu",
            Self::Stanza => "profile:stanza",
            Self::Io => "profile:io",
        }
    }

    /// The profile name used in the ``--profile`` CLI arg sent to Python.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Stanza => "stanza",
            Self::Io => "io",
        }
    }

    /// Whether this profile uses concurrent request handling inside one process.
    pub fn is_concurrent(&self) -> bool {
        matches!(self, Self::Gpu)
    }

    /// Default maximum worker processes per ``(profile, lang, engine_overrides)`` key.
    ///
    /// GPU: 1 process (concurrent via threads).
    /// Stanza: `auto_tune` (multiple processes for CPU parallelism).
    /// IO: 1 process (lightweight).
    pub fn default_max_workers(&self, auto_tune: usize) -> usize {
        match self {
            Self::Gpu => 1,
            Self::Stanza => auto_tune,
            Self::Io => 1,
        }
    }

    /// Map a command name to the profile needed for that command's infer-task worker.
    pub fn for_command(command: &CommandName) -> Option<Self> {
        WorkerTarget::for_command(command).map(|target| {
            let WorkerTarget::InferTask(task) = target;
            Self::for_task(task)
        })
    }
}

/// Bootstrap target for one Python worker process.
///
/// Python workers are model hosts for one infer task such as ASR or forced
/// alignment. Top-level commands are mapped onto these infer-task workers by the
/// Rust control plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerTarget {
    /// Launch a worker around one pure inference task.
    InferTask(InferTask),
}

impl WorkerTarget {
    /// Build a pure inference worker target.
    pub fn infer_task(task: InferTask) -> Self {
        Self::InferTask(task)
    }

    /// Return the string label used in logs, health responses, and worker keys.
    pub fn label(&self) -> String {
        match self {
            Self::InferTask(task) => format!("infer:{}", task_name(*task)),
        }
    }

    /// Return the infer-task worker target used for one released command.
    pub(crate) fn for_command(command: &CommandName) -> Option<Self> {
        let task = command_workflow_descriptor(command)?.infer_task;
        Some(Self::InferTask(task))
    }
}

/// Convert one infer task into the stable snake_case label used across Rust and
/// Python bootstrap code.
pub(crate) fn task_name(task: InferTask) -> &'static str {
    match task {
        InferTask::Morphosyntax => "morphosyntax",
        InferTask::Utseg => "utseg",
        InferTask::Translate => "translate",
        InferTask::Coref => "coref",
        InferTask::Fa => "fa",
        InferTask::Asr => "asr",
        InferTask::Opensmile => "opensmile",
        InferTask::Avqi => "avqi",
        InferTask::Speaker => "speaker",
    }
}

#[cfg(test)]
mod tests {
    use super::{InferTask, WorkerProfile, WorkerTarget};

    #[test]
    fn command_target_maps_transcribe_to_asr() {
        let target = WorkerTarget::for_command(&"transcribe".into());
        assert_eq!(target, Some(WorkerTarget::InferTask(InferTask::Asr)));
    }

    #[test]
    fn command_target_maps_compare_to_morphosyntax() {
        assert_eq!(
            WorkerTarget::for_command(&"compare".into()),
            Some(WorkerTarget::InferTask(InferTask::Morphosyntax))
        );
    }

    #[test]
    fn infer_target_label_is_prefixed() {
        assert_eq!(WorkerTarget::infer_task(InferTask::Fa).label(), "infer:fa");
    }

    #[test]
    fn unknown_command_has_no_worker_target() {
        assert_eq!(WorkerTarget::for_command(&"unknown".into()), None);
    }

    // -- WorkerProfile tests --

    #[test]
    fn gpu_tasks_map_to_gpu_profile() {
        assert_eq!(WorkerProfile::for_task(InferTask::Asr), WorkerProfile::Gpu);
        assert_eq!(WorkerProfile::for_task(InferTask::Fa), WorkerProfile::Gpu);
        assert_eq!(
            WorkerProfile::for_task(InferTask::Speaker),
            WorkerProfile::Gpu
        );
    }

    #[test]
    fn stanza_tasks_map_to_stanza_profile() {
        assert_eq!(
            WorkerProfile::for_task(InferTask::Morphosyntax),
            WorkerProfile::Stanza
        );
        assert_eq!(
            WorkerProfile::for_task(InferTask::Utseg),
            WorkerProfile::Stanza
        );
        assert_eq!(
            WorkerProfile::for_task(InferTask::Coref),
            WorkerProfile::Stanza
        );
    }

    #[test]
    fn io_tasks_map_to_io_profile() {
        assert_eq!(
            WorkerProfile::for_task(InferTask::Translate),
            WorkerProfile::Io
        );
        assert_eq!(
            WorkerProfile::for_task(InferTask::Opensmile),
            WorkerProfile::Io
        );
        assert_eq!(WorkerProfile::for_task(InferTask::Avqi), WorkerProfile::Io);
    }

    #[test]
    fn gpu_profile_is_concurrent() {
        assert!(WorkerProfile::Gpu.is_concurrent());
        assert!(!WorkerProfile::Stanza.is_concurrent());
        assert!(!WorkerProfile::Io.is_concurrent());
    }

    #[test]
    fn profile_for_command_maps_align_to_gpu() {
        assert_eq!(
            WorkerProfile::for_command(&"align".into()),
            Some(WorkerProfile::Gpu)
        );
    }

    #[test]
    fn profile_for_command_maps_morphotag_to_stanza() {
        assert_eq!(
            WorkerProfile::for_command(&"morphotag".into()),
            Some(WorkerProfile::Stanza)
        );
    }

    #[test]
    fn gpu_default_max_workers_is_one() {
        assert_eq!(WorkerProfile::Gpu.default_max_workers(4), 1);
        assert_eq!(WorkerProfile::Stanza.default_max_workers(4), 4);
        assert_eq!(WorkerProfile::Io.default_max_workers(4), 1);
    }
}
