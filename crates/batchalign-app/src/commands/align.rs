//! Command-owned metadata for `align`.

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
use crate::runner::{FaDispatchPlan, FaDispatchRuntime, dispatch_fa_infer};
use crate::store::{JobStore, RunnerJobSnapshot};
use crate::worker::InferTask;
use tracing::warn;

/// Command-owned spec for `align`.
pub(crate) const ALIGN_SPEC: CommandModuleSpec = CommandModuleSpec {
    descriptor: CommandWorkflowDescriptor {
        command: ReleasedCommand::Align,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Fa,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::ForcedAlignment,
    },
    performance: CommandPerformanceProfile {
        scheduling: SchedulingPolicy::PerFileAudio,
        model_sharing: ModelSharingPolicy::SharedWarmWorkers,
        batching: BatchingPolicy::InternalStageBatching,
        parallelism: ParallelismPolicy::BoundedFileWorkers,
        resource_lane: ResourceLane::GpuHeavy,
        constrained_host: ConstrainedHostPolicy::SequentialFallback,
        warmup: WarmupPolicy::BackgroundEligible,
        uses_host_memory_gate: true,
    },
};

/// Build the command-owned align plan from a persisted runner snapshot.
pub(crate) fn build_plan(job: &RunnerJobSnapshot, config: &ServerConfig) -> Option<FaDispatchPlan> {
    debug_assert_eq!(job.dispatch.command, ReleasedCommand::Align);
    FaDispatchPlan::from_job(job, config)
}

/// Run the align command through the shared audio-alignment kernel.
pub(crate) async fn run(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    runtime: FaDispatchRuntime,
) {
    let Some(plan) = build_plan(job, store.config()) else {
        warn!(
            job_id = %job.identity.job_id,
            correlation_id = %job.identity.correlation_id,
            command = %job.dispatch.command,
            "Align command plan could not be built from job options"
        );
        return;
    };

    dispatch_fa_infer(job, store, runtime, plan).await;
}
