//! Trait-oriented wrapper for forced alignment.

use async_trait::async_trait;

use crate::api::{ChatText, LanguageCode3};
use crate::error::ServerError;
use crate::fa::run_fa_impl;
use crate::params::{AudioContext, FaParams};
use crate::pipeline::PipelineServices;
use crate::runner::util::ProgressSender;
use crate::types::results::FaResult;

use super::PerFileWorkflow;

/// Borrowed request bundle for one forced-alignment execution.
pub(crate) struct ForcedAlignmentWorkflowRequest<'a> {
    /// CHAT text to align.
    pub chat_text: ChatText<'a>,
    /// Typed audio context for this file.
    pub audio: &'a AudioContext<'a>,
    /// Resolved worker language for the FA runtime.
    pub worker_lang: &'a LanguageCode3,
    /// Shared runtime services.
    pub services: PipelineServices<'a>,
    /// Forced-alignment parameters for this run.
    pub params: &'a FaParams,
    /// Optional progress sink.
    pub progress: Option<&'a ProgressSender>,
}

/// Per-file forced-alignment workflow.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct ForcedAlignmentWorkflow;

#[async_trait]
impl PerFileWorkflow for ForcedAlignmentWorkflow {
    type Output = FaResult;
    type Request<'a>
        = ForcedAlignmentWorkflowRequest<'a>
    where
        Self: 'a;

    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError> {
        run_fa_impl(
            request.chat_text.as_ref(),
            request.audio,
            request.worker_lang,
            request.services,
            request.params,
            request.progress,
        )
        .await
    }
}
