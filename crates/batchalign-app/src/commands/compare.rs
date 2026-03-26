//! Command-owned metadata for `compare`.

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
use crate::pipeline::PipelineServices;
use crate::runner::{BatchedInferDispatchPlan, dispatch_batched_infer};
use crate::store::{JobStore, RunnerJobSnapshot};
use crate::worker::InferTask;

/// Command-owned spec for `compare`.
pub(crate) const COMPARE_SPEC: CommandModuleSpec = CommandModuleSpec {
    descriptor: CommandWorkflowDescriptor {
        command: ReleasedCommand::Compare,
        family: WorkflowFamily::ReferenceProjection,
        infer_task: InferTask::Morphosyntax,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: false,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BatchedTextInfer,
    },
    performance: CommandPerformanceProfile {
        scheduling: SchedulingPolicy::ReferenceProjection,
        model_sharing: ModelSharingPolicy::SharedWarmWorkers,
        batching: BatchingPolicy::PairedInputs,
        parallelism: ParallelismPolicy::SingleDispatchPerJob,
        resource_lane: ResourceLane::Mixed,
        constrained_host: ConstrainedHostPolicy::SequentialFallback,
        warmup: WarmupPolicy::BackgroundEligible,
        uses_host_memory_gate: true,
    },
};

/// Build the command-owned compare plan from a persisted runner snapshot.
pub(crate) fn build_plan(
    job: &RunnerJobSnapshot,
    config: &ServerConfig,
) -> BatchedInferDispatchPlan {
    debug_assert_eq!(job.dispatch.command, ReleasedCommand::Compare);
    BatchedInferDispatchPlan::from_job(job, config)
}

/// Run the compare command through the shared reference-aware batched-text kernel.
pub(crate) async fn run(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    services: PipelineServices<'_>,
) {
    dispatch_batched_infer(job, store, services, build_plan(job, store.config())).await;
}
