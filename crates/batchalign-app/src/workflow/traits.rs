//! Contributor-facing workflow families and materialization contracts.

use async_trait::async_trait;

use crate::error::ServerError;

/// High-level workflow family for one released command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkflowFamily {
    /// One file goes in, one primary output comes out.
    PerFileTransform,
    /// Many files go in, pooled work happens internally, then per-file outputs fan back out.
    CrossFileBatchTransform,
    /// Two artifacts are jointly primary and materialize from a comparison/projection bundle.
    ReferenceProjection,
    /// One workflow composes other workflows rather than reimplementing them.
    Composite,
}

/// Turn typed intermediate workflow artifacts into user-facing outputs.
pub(crate) trait Materializer<Artifacts> {
    /// Final output produced by this materializer.
    type Output;

    /// Materialize the final output from typed intermediate artifacts.
    fn materialize(&self, artifacts: Artifacts) -> Result<Self::Output, ServerError>;
}

/// One file goes in, one primary output comes out.
#[async_trait]
pub(crate) trait PerFileWorkflow {
    /// Final output of the workflow.
    type Output;
    /// Borrowed request bundle for one workflow execution.
    type Request<'a>
    where
        Self: 'a;

    /// Run the workflow for one file-scoped request.
    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError>;
}

/// Many files go in, pooled work happens internally, and per-file outputs fan
/// back out.
#[async_trait]
pub(crate) trait CrossFileBatchWorkflow {
    /// Final output of the batch workflow.
    type Output;
    /// Borrowed request bundle for one batch execution.
    type Request<'a>
    where
        Self: 'a;

    /// Run the workflow over a batch of files.
    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError>;
}

/// Two artifacts are jointly primary, producing a typed comparison/projection
/// bundle that can later be materialized in more than one form.
#[async_trait]
pub(crate) trait ReferenceProjectionWorkflow {
    /// Typed internal artifact bundle produced before final materialization.
    type ArtifactBundle;
    /// Final output shape selected by the current materializer.
    type Output;
    /// Materializer responsible for the released output shape.
    type Materializer: Materializer<Self::ArtifactBundle, Output = Self::Output> + Send + Sync;
    /// Borrowed request bundle for one workflow execution.
    type Request<'a>
    where
        Self: 'a;

    /// Build the typed intermediate artifacts for this workflow family.
    async fn build_artifacts(
        &self,
        request: Self::Request<'_>,
    ) -> Result<Self::ArtifactBundle, ServerError>;

    /// Run the workflow through artifact construction plus materialization.
    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError>;
}

/// Compose typed sub-workflows rather than reimplementing their internals.
#[async_trait]
pub(crate) trait CompositeWorkflow {
    /// Final output of the composed workflow.
    type Output;
    /// Borrowed request bundle for one workflow execution.
    type Request<'a>
    where
        Self: 'a;

    /// Run the composite workflow.
    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError>;
}
