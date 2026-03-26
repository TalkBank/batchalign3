//! Command-owned metadata for `transcribe` and `transcribe_s`.

use std::sync::Arc;

use crate::ReleasedCommand;
use crate::command_family::WorkflowFamily;
use crate::commands::spec::{
    BatchingPolicy, CommandCapabilityKind, CommandModuleSpec, CommandOutputPathKind,
    CommandPerformanceProfile, CommandWorkflowDescriptor, ConstrainedHostPolicy,
    ModelSharingPolicy, ParallelismPolicy, ResourceLane, RunnerDispatchKind, SchedulingPolicy,
    WarmupPolicy,
};
use crate::config::ServerConfig;
use crate::runner::{TranscribeDispatchPlan, TranscribeDispatchRuntime, dispatch_transcribe_infer};
use crate::store::{JobStore, RunnerJobSnapshot};
use crate::worker::InferTask;
use tracing::warn;

const TRANSCRIBE_PERFORMANCE: CommandPerformanceProfile = CommandPerformanceProfile {
    scheduling: SchedulingPolicy::PerFileAudio,
    model_sharing: ModelSharingPolicy::SharedWarmWorkers,
    batching: BatchingPolicy::InternalStageBatching,
    parallelism: ParallelismPolicy::BoundedFileWorkers,
    resource_lane: ResourceLane::GpuHeavy,
    constrained_host: ConstrainedHostPolicy::SequentialFallback,
    warmup: WarmupPolicy::BackgroundEligible,
    uses_host_memory_gate: true,
};

/// Command-owned spec for `transcribe`.
pub(crate) const TRANSCRIBE_SPEC: CommandModuleSpec = CommandModuleSpec {
    descriptor: CommandWorkflowDescriptor {
        command: ReleasedCommand::Transcribe,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Asr,
        capability_kind: CommandCapabilityKind::ServerComposed,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::ReplaceExtension("cha"),
        runner_dispatch_kind: RunnerDispatchKind::TranscribeAudioInfer,
    },
    performance: TRANSCRIBE_PERFORMANCE,
};

/// Command-owned spec for `transcribe_s`.
pub(crate) const TRANSCRIBE_S_SPEC: CommandModuleSpec = CommandModuleSpec {
    descriptor: CommandWorkflowDescriptor {
        command: ReleasedCommand::TranscribeS,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Asr,
        capability_kind: CommandCapabilityKind::ServerComposed,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::ReplaceExtension("cha"),
        runner_dispatch_kind: RunnerDispatchKind::TranscribeAudioInfer,
    },
    performance: TRANSCRIBE_PERFORMANCE,
};

/// Build the command-owned transcribe plan from a persisted runner snapshot.
pub(crate) fn build_plan(
    job: &RunnerJobSnapshot,
    config: &ServerConfig,
) -> Option<TranscribeDispatchPlan> {
    debug_assert!(matches!(
        job.dispatch.command,
        ReleasedCommand::Transcribe | ReleasedCommand::TranscribeS
    ));
    TranscribeDispatchPlan::from_job(job, config)
}

/// Run the transcribe command through the shared runner kernel.
pub(crate) async fn run(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    runtime: TranscribeDispatchRuntime,
) {
    let Some(plan) = build_plan(job, store.config()) else {
        warn!(
            job_id = %job.identity.job_id,
            correlation_id = %job.identity.correlation_id,
            command = %job.dispatch.command,
            "Transcribe command plan could not be built from job options"
        );
        return;
    };

    dispatch_transcribe_infer(job, store, runtime, plan).await;
}
