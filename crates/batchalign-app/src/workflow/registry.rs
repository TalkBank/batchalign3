//! Released-command workflow registry.
//!
//! This is the code-first catalog answering “what family is command X, which
//! infer task does it rely on, and is the released surface direct or
//! Rust-composed?”

use crate::api::{CommandName, ReleasedCommand};
use crate::worker::InferTask;

use super::traits::WorkflowFamily;

/// How one released command is surfaced relative to the worker infer-task layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandCapabilityKind {
    /// Command is advertised directly from one infer task.
    DirectInfer,
    /// Command is synthesized by Rust from lower-level infer capability.
    ServerComposed,
}

/// How one released command maps an input filename to its primary output filename.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandOutputPathKind {
    /// Keep the incoming relative path and extension unchanged.
    PreserveInputName,
    /// Replace the input extension with a fixed output extension.
    ReplaceExtension(&'static str),
}

/// Which server-side runtime path owns one released command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RunnerDispatchKind {
    /// Text-only commands pooled through the batched infer path.
    BatchedTextInfer,
    /// Forced alignment with per-file audio/media resolution.
    ForcedAlignment,
    /// Transcribe audio through the Rust-owned ASR orchestration path.
    TranscribeAudioInfer,
    /// Benchmark audio through the composite benchmark orchestrator.
    BenchmarkAudioInfer,
    /// Media-analysis V2 path for commands like openSMILE and AVQI.
    MediaAnalysisV2,
}

/// Typed descriptor for one released command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandWorkflowDescriptor {
    /// Stable command name exposed to users.
    pub command: ReleasedCommand,
    /// Workflow family that owns the command semantics.
    pub family: WorkflowFamily,
    /// Primary infer task required by the worker layer.
    pub infer_task: InferTask,
    /// How the command is surfaced relative to the worker layer.
    pub capability_kind: CommandCapabilityKind,
    /// Whether this command requires client-local audio access in the CLI.
    pub uses_local_audio: bool,
    /// How this command derives its primary output path.
    pub output_path_kind: CommandOutputPathKind,
    /// Which server-side runtime path owns this command.
    pub runner_dispatch_kind: RunnerDispatchKind,
}

const RELEASED_COMMAND_WORKFLOWS: &[CommandWorkflowDescriptor] = &[
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Morphotag,
        family: WorkflowFamily::CrossFileBatchTransform,
        infer_task: InferTask::Morphosyntax,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Utseg,
        family: WorkflowFamily::CrossFileBatchTransform,
        infer_task: InferTask::Utseg,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Translate,
        family: WorkflowFamily::CrossFileBatchTransform,
        infer_task: InferTask::Translate,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Coref,
        family: WorkflowFamily::CrossFileBatchTransform,
        infer_task: InferTask::Coref,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Align,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Fa,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::ForcedAlignment,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Transcribe,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Asr,
        capability_kind: CommandCapabilityKind::ServerComposed,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::ReplaceExtension("cha"),
        runner_dispatch_kind: RunnerDispatchKind::TranscribeAudioInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::TranscribeS,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Asr,
        capability_kind: CommandCapabilityKind::ServerComposed,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::ReplaceExtension("cha"),
        runner_dispatch_kind: RunnerDispatchKind::TranscribeAudioInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Compare,
        family: WorkflowFamily::ReferenceProjection,
        infer_task: InferTask::Morphosyntax,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Benchmark,
        family: WorkflowFamily::Composite,
        infer_task: InferTask::Asr,
        capability_kind: CommandCapabilityKind::ServerComposed,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BenchmarkAudioInfer,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Opensmile,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Opensmile,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::MediaAnalysisV2,
    },
    CommandWorkflowDescriptor {
        command: ReleasedCommand::Avqi,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Avqi,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::MediaAnalysisV2,
    },
];

/// Return the typed workflow descriptor for one released command.
pub(crate) fn released_command_descriptor(command: ReleasedCommand) -> CommandWorkflowDescriptor {
    RELEASED_COMMAND_WORKFLOWS
        .iter()
        .copied()
        .find(|descriptor| descriptor.command == command)
        .expect("released command missing workflow descriptor")
}

/// Return the typed workflow descriptor for one released command.
pub(crate) fn command_workflow_descriptor(
    command: &CommandName,
) -> Option<CommandWorkflowDescriptor> {
    ReleasedCommand::try_from(command)
        .ok()
        .map(released_command_descriptor)
}

/// Return the typed workflow descriptor for one released command name.
#[cfg(test)]
pub(crate) fn command_workflow_descriptor_by_name(
    command: &str,
) -> Option<CommandWorkflowDescriptor> {
    ReleasedCommand::try_from(command)
        .ok()
        .map(released_command_descriptor)
}

/// Return whether one closed released command requires client-local audio access.
pub fn released_command_uses_local_audio(command: ReleasedCommand) -> bool {
    released_command_descriptor(command).uses_local_audio
}

/// Return whether one released command requires client-local audio access in the CLI.
pub fn command_uses_local_audio(command: &str) -> bool {
    ReleasedCommand::try_from(command)
        .ok()
        .map(released_command_uses_local_audio)
        .unwrap_or(false)
}

/// Return the released command workflow descriptors.
pub(crate) fn released_command_workflows() -> &'static [CommandWorkflowDescriptor] {
    RELEASED_COMMAND_WORKFLOWS
}

/// Return the runner dispatch kind for one released command.
pub(crate) fn command_runner_dispatch_kind(command: &CommandName) -> Option<RunnerDispatchKind> {
    command_workflow_descriptor(command).map(|descriptor| descriptor.runner_dispatch_kind)
}

/// Derive the primary output filename for one released command.
pub(crate) fn result_filename_for_released_command(
    command: ReleasedCommand,
    filename: &str,
) -> String {
    match released_command_descriptor(command).output_path_kind {
        CommandOutputPathKind::ReplaceExtension(extension) => std::path::Path::new(filename)
            .with_extension(extension)
            .to_string_lossy()
            .to_string(),
        CommandOutputPathKind::PreserveInputName => filename.to_string(),
    }
}

/// Derive the primary output filename for one released command name.
pub(crate) fn result_filename_for_command_name(command: &str, filename: &str) -> String {
    ReleasedCommand::try_from(command)
        .map(|command| result_filename_for_released_command(command, filename))
        .unwrap_or_else(|_| filename.to_string())
}

#[cfg(test)]
mod tests {
    use crate::api::ReleasedCommand;

    use super::{
        CommandCapabilityKind, WorkflowFamily, command_workflow_descriptor_by_name,
        released_command_descriptor, released_command_workflows,
    };

    #[test]
    fn compare_is_reference_projection() {
        let descriptor = released_command_descriptor(ReleasedCommand::Compare);
        assert_eq!(descriptor.family, WorkflowFamily::ReferenceProjection);
    }

    #[test]
    fn benchmark_is_composite() {
        let descriptor = released_command_descriptor(ReleasedCommand::Benchmark);
        assert_eq!(descriptor.family, WorkflowFamily::Composite);
        assert_eq!(
            descriptor.capability_kind,
            CommandCapabilityKind::ServerComposed
        );
    }

    #[test]
    fn released_command_registry_has_unique_names() {
        let mut names: Vec<&str> = released_command_workflows()
            .iter()
            .map(|descriptor| descriptor.command.as_ref())
            .collect();
        let original_len = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), original_len, "duplicate command descriptors");
    }

    #[test]
    fn parse_wire_name_roundtrips() {
        let command = ReleasedCommand::try_from("transcribe_s").expect("released command");
        assert_eq!(command, ReleasedCommand::TranscribeS);
        assert_eq!(command.as_str(), "transcribe_s");
        assert!(command_workflow_descriptor_by_name(command.as_str()).is_some());
    }
}
