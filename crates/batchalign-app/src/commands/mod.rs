//! Command-owned catalog and derived runtime metadata.

use crate::ReleasedCommand;

pub(crate) mod align;
pub(crate) mod avqi;
pub(crate) mod benchmark;
mod catalog;
pub(crate) mod compare;
pub(crate) mod coref;
mod kernel;
pub(crate) mod morphotag;
pub(crate) mod opensmile;
pub(crate) mod spec;
pub(crate) mod transcribe;
pub(crate) mod translate;
pub(crate) mod utseg;

pub(crate) use catalog::{
    command_workflow_descriptor, released_command_definition, released_command_definitions,
    released_command_descriptor,
};
pub(crate) use kernel::CommandKernelPlan;
pub(crate) use spec::RunnerDispatchKind;

/// Return whether one closed released command requires shared-filesystem audio access.
pub fn released_command_uses_local_audio(command: ReleasedCommand) -> bool {
    released_command_descriptor(command).uses_local_audio
}

/// Return whether one released command requires shared-filesystem audio access.
pub fn command_uses_local_audio(command: &str) -> bool {
    ReleasedCommand::try_from(command)
        .ok()
        .map(released_command_uses_local_audio)
        .unwrap_or(false)
}

/// Return the runner dispatch kind for one released command.
pub(crate) fn command_runner_dispatch_kind(command: ReleasedCommand) -> Option<RunnerDispatchKind> {
    command_workflow_descriptor(command).map(|descriptor| descriptor.runner_dispatch_kind)
}
