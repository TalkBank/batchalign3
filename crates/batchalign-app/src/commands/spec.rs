//! Command-owned catalog types and performance profiles.

use crate::ReleasedCommand;
use crate::command_family::WorkflowFamily;
use crate::worker::InferTask;

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

/// Which server-side runtime path currently owns one released command.
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

/// High-level scheduling shape the command expects from the shared kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SchedulingPolicy {
    /// One audio/media file at a time, with bounded per-job parallelism.
    PerFileAudio,
    /// Many text files pooled into one or more shared infer batches.
    CrossFileBatch,
    /// One primary file plus one paired reference artifact.
    ReferenceProjection,
    /// The command is built by composing other command-owned flows.
    Composite,
    /// Per-file media analysis over non-CHAT inputs.
    PerFileMediaAnalysis,
}

/// How the command expects model state to be shared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelSharingPolicy {
    /// Reuse warm workers and shared model state whenever possible.
    SharedWarmWorkers,
    /// Let composed child commands own model sharing.
    DelegatedToSubcommands,
}

/// Whether the command benefits from cross-file or internal batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchingPolicy {
    /// No profitable batching beyond ordinary per-file execution.
    None,
    /// Pool many files together into shared worker requests.
    CrossFileBatch,
    /// Keep the top-level unit per file, but allow internal stage batching.
    InternalStageBatching,
    /// One main file plus one paired reference artifact.
    PairedInputs,
}

/// How much per-command parallelism the shared kernel should expose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ParallelismPolicy {
    /// Bound file-level concurrency and let the kernel auto-tune worker counts.
    BoundedFileWorkers,
    /// Keep one command-level dispatch at a time per job.
    SingleDispatchPerJob,
    /// Let composed child commands own their own parallelism.
    DelegatedToSubcommands,
}

/// How one command should behave on constrained-memory hosts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstrainedHostPolicy {
    /// Allow the host to clamp execution to one worker and rely on lazy startup
    /// rather than speculative resident state.
    SequentialFallback,
    /// Let composed child commands own constrained-host behavior.
    DelegatedToSubcommands,
}

/// Whether the command should participate in optional background warmup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WarmupPolicy {
    /// The command should stay lazy/on-demand by default.
    LazyOnDemand,
    /// The host may warm this command in the background when capacity allows.
    BackgroundEligible,
    /// Let composed child commands own warmup behavior.
    DelegatedToSubcommands,
}

/// Dominant resource lane for the command's hot path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResourceLane {
    /// GPU-backed workloads where device memory is the main bottleneck.
    GpuHeavy,
    /// CPU-bound workloads that still reuse warm model workers.
    CpuBound,
    /// Mostly IO / media feature extraction.
    IoBound,
    /// Mixed pipelines touching both CPU and GPU stages.
    Mixed,
}

/// Explicit performance contract for one command module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandPerformanceProfile {
    /// High-level scheduling shape.
    pub scheduling: SchedulingPolicy,
    /// How model state should be shared or delegated.
    pub model_sharing: ModelSharingPolicy,
    /// Whether the command benefits from batching.
    pub batching: BatchingPolicy,
    /// How the kernel should expose per-command concurrency.
    pub parallelism: ParallelismPolicy,
    /// Dominant resource lane for the command.
    pub resource_lane: ResourceLane,
    /// How the command should behave on constrained-memory hosts.
    pub constrained_host: ConstrainedHostPolicy,
    /// Whether the command is eligible for speculative/background warmup.
    pub warmup: WarmupPolicy,
    /// Whether the host-memory gate must stay in play for this command.
    pub uses_host_memory_gate: bool,
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
    /// Which server-side runtime path currently owns this command.
    pub runner_dispatch_kind: RunnerDispatchKind,
}

/// Command-owned metadata plus the explicit performance contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandModuleSpec {
    /// Released command descriptor used by the compatibility facade.
    pub descriptor: CommandWorkflowDescriptor,
    /// Resource-aware execution profile for the new command-owned architecture.
    pub performance: CommandPerformanceProfile,
}
