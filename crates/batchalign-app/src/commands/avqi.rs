//! Command-owned metadata for `avqi`.

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
use crate::runner::{
    MediaAnalysisDispatchPlan, MediaAnalysisDispatchRuntime, dispatch_media_analysis_v2,
};
use crate::store::{JobStore, RunnerJobSnapshot};
use crate::worker::InferTask;
use tracing::warn;

/// Command-owned spec for `avqi`.
pub(crate) const AVQI_SPEC: CommandModuleSpec = CommandModuleSpec {
    descriptor: CommandWorkflowDescriptor {
        command: ReleasedCommand::Avqi,
        family: WorkflowFamily::PerFileTransform,
        infer_task: InferTask::Avqi,
        capability_kind: CommandCapabilityKind::DirectInfer,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::MediaAnalysisV2,
    },
    performance: CommandPerformanceProfile {
        scheduling: SchedulingPolicy::PerFileMediaAnalysis,
        model_sharing: ModelSharingPolicy::SharedWarmWorkers,
        batching: BatchingPolicy::None,
        parallelism: ParallelismPolicy::BoundedFileWorkers,
        resource_lane: ResourceLane::IoBound,
        constrained_host: ConstrainedHostPolicy::SequentialFallback,
        warmup: WarmupPolicy::LazyOnDemand,
        uses_host_memory_gate: true,
    },
};

/// Build the command-owned AVQI plan from a persisted runner snapshot.
pub(crate) fn build_plan(
    job: &RunnerJobSnapshot,
    config: &ServerConfig,
) -> Option<MediaAnalysisDispatchPlan> {
    debug_assert_eq!(job.dispatch.command, ReleasedCommand::Avqi);
    MediaAnalysisDispatchPlan::from_job(job, config)
}

/// Run the AVQI command through the shared media-analysis kernel.
pub(crate) async fn run(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    runtime: MediaAnalysisDispatchRuntime,
) {
    let Some(plan) = build_plan(job, store.config()) else {
        warn!(
            job_id = %job.identity.job_id,
            correlation_id = %job.identity.correlation_id,
            command = %job.dispatch.command,
            "AVQI command plan could not be built from job options"
        );
        return;
    };

    dispatch_media_analysis_v2(job, store, runtime, plan).await;
}
