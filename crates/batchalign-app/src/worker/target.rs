//! Typed worker bootstrap targets for the Rust control plane.
//!
//! The released worker pool is now infer-task-only. Rust still reasons in terms
//! of top-level commands for warmup, memory checks, and scheduling, but Python
//! workers are always bootstrapped around one inference task.

use crate::api::CommandName;

use super::InferTask;

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
        let task = match command.as_ref() {
            "morphotag" | "compare" => InferTask::Morphosyntax,
            "utseg" => InferTask::Utseg,
            "translate" => InferTask::Translate,
            "coref" => InferTask::Coref,
            "align" => InferTask::Fa,
            "transcribe" | "transcribe_s" | "benchmark" => InferTask::Asr,
            "opensmile" => InferTask::Opensmile,
            "avqi" => InferTask::Avqi,
            _ => return None,
        };
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
    use super::{InferTask, WorkerTarget};

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
}
