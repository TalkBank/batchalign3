//! Command-owned metadata for `benchmark`.

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
use crate::runner::{BenchmarkDispatchPlan, BenchmarkDispatchRuntime, dispatch_benchmark_infer};
use crate::store::{JobStore, RunnerJobSnapshot};
use crate::worker::InferTask;
use tracing::warn;

/// Command-owned spec for `benchmark`.
pub(crate) const BENCHMARK_SPEC: CommandModuleSpec = CommandModuleSpec {
    descriptor: CommandWorkflowDescriptor {
        command: ReleasedCommand::Benchmark,
        family: WorkflowFamily::Composite,
        infer_task: InferTask::Asr,
        capability_kind: CommandCapabilityKind::ServerComposed,
        uses_local_audio: true,
        output_path_kind: CommandOutputPathKind::PreserveInputName,
        runner_dispatch_kind: RunnerDispatchKind::BenchmarkAudioInfer,
    },
    performance: CommandPerformanceProfile {
        scheduling: SchedulingPolicy::Composite,
        model_sharing: ModelSharingPolicy::DelegatedToSubcommands,
        batching: BatchingPolicy::None,
        parallelism: ParallelismPolicy::DelegatedToSubcommands,
        resource_lane: ResourceLane::Mixed,
        constrained_host: ConstrainedHostPolicy::DelegatedToSubcommands,
        warmup: WarmupPolicy::DelegatedToSubcommands,
        uses_host_memory_gate: true,
    },
};

/// Build the command-owned benchmark plan from a persisted runner snapshot.
pub(crate) fn build_plan(
    job: &RunnerJobSnapshot,
    config: &ServerConfig,
) -> Option<BenchmarkDispatchPlan> {
    debug_assert_eq!(job.dispatch.command, ReleasedCommand::Benchmark);
    BenchmarkDispatchPlan::from_job(job, config)
}

/// Run the benchmark command through the shared benchmark kernel.
pub(crate) async fn run(
    job: &RunnerJobSnapshot,
    store: &Arc<JobStore>,
    runtime: BenchmarkDispatchRuntime,
) {
    let Some(plan) = build_plan(job, store.config()) else {
        warn!(
            job_id = %job.identity.job_id,
            correlation_id = %job.identity.correlation_id,
            command = %job.dispatch.command,
            "Benchmark command plan could not be built from job options"
        );
        return;
    };

    dispatch_benchmark_infer(job, store, runtime, plan).await;
}
