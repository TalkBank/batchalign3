//! Command-routing policy helpers for the job runner.
//!
//! Keep these small and declarative. `runner/mod.rs` should read as job
//! lifecycle orchestration, not as a mixed bag of routing tables and filename
//! conventions.

use crate::api::ReleasedCommand;
use crate::worker::InferTask;
use crate::workflow::{
    RunnerDispatchKind, command_runner_dispatch_kind, command_workflow_descriptor,
    result_filename_for_command_name,
};

/// Return the primary infer task backing one released command.
pub(crate) fn infer_task_for_command(command: ReleasedCommand) -> Option<InferTask> {
    command_workflow_descriptor(command).map(|descriptor| descriptor.infer_task)
}

/// Return `true` when the released command must use a Rust-owned infer-backed
/// dispatch path instead of a pure content relay.
///
/// Compare is excluded: its core algorithm is pure Rust and morphosyntax
/// enrichment is optional (the workflow falls back to raw transcripts).
pub(crate) fn command_requires_infer(command: ReleasedCommand) -> bool {
    if command == ReleasedCommand::Compare {
        return false;
    }
    matches!(
        command_runner_dispatch_kind(command),
        Some(
            RunnerDispatchKind::BatchedTextInfer
                | RunnerDispatchKind::ForcedAlignment
                | RunnerDispatchKind::MediaAnalysisV2
        )
    )
}

/// Derive the result filename for one released command.
pub(crate) fn result_filename_for_command(command: ReleasedCommand, filename: &str) -> String {
    result_filename_for_command_name(command.as_ref(), filename)
}
