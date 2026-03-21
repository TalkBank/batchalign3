//! Typed workflow families for batchalign3 command orchestration.
//!
//! Read this module in three layers:
//!
//! 1. [`traits`] for the contributor-facing workflow families.
//! 2. [`registry`] for the released command catalog and infer-task mapping.
//! 3. family modules such as [`transcribe`], [`fa`], [`morphosyntax`],
//!    [`text_batch`], [`compare`], and [`benchmark`] for concrete workflow
//!    implementations.

pub(crate) mod benchmark;
pub(crate) mod compare;
pub(crate) mod fa;
pub(crate) mod morphosyntax;
pub(crate) mod registry;
pub(crate) mod text_batch;
pub(crate) mod traits;
pub(crate) mod transcribe;

pub(crate) use registry::{
    CommandCapabilityKind, RunnerDispatchKind, command_runner_dispatch_kind,
    command_workflow_descriptor, released_command_workflows, result_filename_for_command_name,
};
pub use registry::{command_uses_local_audio, released_command_uses_local_audio};
pub(crate) use traits::{
    CompositeWorkflow, CrossFileBatchWorkflow, Materializer, PerFileWorkflow,
    ReferenceProjectionWorkflow,
};
