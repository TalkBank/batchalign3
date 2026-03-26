//! Static command-owned catalog.

use crate::ReleasedCommand;

use super::align::ALIGN_SPEC;
use super::avqi::AVQI_SPEC;
use super::benchmark::BENCHMARK_SPEC;
use super::compare::COMPARE_SPEC;
use super::coref::COREF_SPEC;
use super::morphotag::MORPHOTAG_SPEC;
use super::opensmile::OPENSMILE_SPEC;
use super::spec::{CommandModuleSpec, CommandPerformanceProfile, CommandWorkflowDescriptor};
use super::transcribe::{TRANSCRIBE_S_SPEC, TRANSCRIBE_SPEC};
use super::translate::TRANSLATE_SPEC;
use super::utseg::UTSEG_SPEC;

const RELEASED_COMMAND_SPECS: &[CommandModuleSpec] = &[
    MORPHOTAG_SPEC,
    UTSEG_SPEC,
    TRANSLATE_SPEC,
    COREF_SPEC,
    ALIGN_SPEC,
    TRANSCRIBE_SPEC,
    TRANSCRIBE_S_SPEC,
    COMPARE_SPEC,
    BENCHMARK_SPEC,
    OPENSMILE_SPEC,
    AVQI_SPEC,
];

const RELEASED_COMMAND_WORKFLOWS: &[CommandWorkflowDescriptor] = &[
    MORPHOTAG_SPEC.descriptor,
    UTSEG_SPEC.descriptor,
    TRANSLATE_SPEC.descriptor,
    COREF_SPEC.descriptor,
    ALIGN_SPEC.descriptor,
    TRANSCRIBE_SPEC.descriptor,
    TRANSCRIBE_S_SPEC.descriptor,
    COMPARE_SPEC.descriptor,
    BENCHMARK_SPEC.descriptor,
    OPENSMILE_SPEC.descriptor,
    AVQI_SPEC.descriptor,
];

/// Return the full command-owned spec for one released command.
pub(crate) fn released_command_spec(command: ReleasedCommand) -> CommandModuleSpec {
    RELEASED_COMMAND_SPECS
        .iter()
        .copied()
        .find(|spec| spec.descriptor.command == command)
        .expect("released command missing module spec")
}

/// Return the command-owned spec for one released command if present.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn command_module_spec(command: ReleasedCommand) -> Option<CommandModuleSpec> {
    Some(released_command_spec(command))
}

/// Return the released command specs.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn released_command_specs() -> &'static [CommandModuleSpec] {
    RELEASED_COMMAND_SPECS
}

/// Return the compatibility workflow descriptor for one released command.
pub(crate) fn released_command_descriptor(command: ReleasedCommand) -> CommandWorkflowDescriptor {
    released_command_spec(command).descriptor
}

/// Return the compatibility workflow descriptor for one released command if present.
pub(crate) fn command_workflow_descriptor(
    command: ReleasedCommand,
) -> Option<CommandWorkflowDescriptor> {
    Some(released_command_descriptor(command))
}

/// Return the compatibility descriptor slice for old registry callers.
pub(crate) fn released_command_workflows() -> &'static [CommandWorkflowDescriptor] {
    RELEASED_COMMAND_WORKFLOWS
}

/// Return the explicit performance profile for one released command.
pub(crate) fn command_performance_profile(command: ReleasedCommand) -> CommandPerformanceProfile {
    released_command_spec(command).performance
}

#[cfg(test)]
mod tests {
    use super::{command_performance_profile, released_command_specs};
    use crate::ReleasedCommand;
    use crate::commands::spec::{
        BatchingPolicy, ConstrainedHostPolicy, SchedulingPolicy, WarmupPolicy,
    };

    #[test]
    fn command_specs_have_unique_names() {
        let mut names: Vec<&str> = released_command_specs()
            .iter()
            .map(|spec| spec.descriptor.command.as_ref())
            .collect();
        let original_len = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), original_len, "duplicate command specs");
    }

    #[test]
    fn compare_keeps_paired_reference_profile() {
        let profile = command_performance_profile(ReleasedCommand::Compare);
        assert_eq!(profile.scheduling, SchedulingPolicy::ReferenceProjection);
        assert_eq!(profile.batching, BatchingPolicy::PairedInputs);
    }

    #[test]
    fn morphotag_keeps_cross_file_batch_profile() {
        let profile = command_performance_profile(ReleasedCommand::Morphotag);
        assert_eq!(profile.scheduling, SchedulingPolicy::CrossFileBatch);
        assert_eq!(profile.batching, BatchingPolicy::CrossFileBatch);
    }

    #[test]
    fn transcribe_profile_is_background_eligible_but_can_fallback() {
        let profile = command_performance_profile(ReleasedCommand::Transcribe);
        assert_eq!(profile.warmup, WarmupPolicy::BackgroundEligible);
        assert_eq!(
            profile.constrained_host,
            ConstrainedHostPolicy::SequentialFallback
        );
    }

    #[test]
    fn benchmark_profile_delegates_constrained_host_behavior() {
        let profile = command_performance_profile(ReleasedCommand::Benchmark);
        assert_eq!(profile.warmup, WarmupPolicy::DelegatedToSubcommands);
        assert_eq!(
            profile.constrained_host,
            ConstrainedHostPolicy::DelegatedToSubcommands
        );
    }
}
