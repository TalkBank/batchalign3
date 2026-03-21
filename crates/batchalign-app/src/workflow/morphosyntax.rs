//! Trait-oriented wrappers for morphosyntax workflows.

use async_trait::async_trait;

use crate::api::ChatText;
use crate::error::ServerError;
use crate::morphosyntax::{run_morphosyntax_batch_impl, run_morphosyntax_impl};
use crate::params::MorphosyntaxParams;
use crate::pipeline::PipelineServices;
use crate::workflow::text_batch::{TextBatchFileInput, TextBatchFileResults};

use super::{CrossFileBatchWorkflow, PerFileWorkflow};

/// Borrowed request bundle for one per-file morphosyntax execution.
pub(crate) struct MorphosyntaxWorkflowRequest<'a> {
    /// CHAT text to tag.
    pub chat_text: ChatText<'a>,
    /// Shared runtime services.
    pub services: PipelineServices<'a>,
    /// Morphotag parameters for this file.
    pub params: &'a MorphosyntaxParams<'a>,
}

/// Borrowed request bundle for one cross-file batch morphosyntax execution.
pub(crate) struct MorphosyntaxBatchWorkflowRequest<'a> {
    /// Files and their CHAT text payloads.
    pub files: &'a [TextBatchFileInput],
    /// Shared runtime services.
    pub services: PipelineServices<'a>,
    /// Morphotag parameters shared across the batch.
    pub params: &'a MorphosyntaxParams<'a>,
}

/// Per-file morphosyntax workflow.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MorphosyntaxWorkflow;

/// Cross-file batch morphosyntax workflow.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MorphosyntaxBatchWorkflow;

#[async_trait]
impl PerFileWorkflow for MorphosyntaxWorkflow {
    type Output = String;
    type Request<'a>
        = MorphosyntaxWorkflowRequest<'a>
    where
        Self: 'a;

    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError> {
        run_morphosyntax_impl(request.chat_text.as_ref(), request.services, request.params).await
    }
}

#[async_trait]
impl CrossFileBatchWorkflow for MorphosyntaxBatchWorkflow {
    type Output = TextBatchFileResults;
    type Request<'a>
        = MorphosyntaxBatchWorkflowRequest<'a>
    where
        Self: 'a;

    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError> {
        Ok(run_morphosyntax_batch_impl(request.files, request.services, request.params).await)
    }
}
