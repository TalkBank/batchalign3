//! Trait-oriented wrapper for the transcribe pipeline.

use std::path::Path;

use async_trait::async_trait;

use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::pipeline::transcribe::run_transcribe_pipeline;
use crate::runner::util::ProgressSender;
use crate::transcribe::TranscribeOptions;

use super::PerFileWorkflow;

/// Borrowed request bundle for one transcribe workflow execution.
pub(crate) struct TranscribeWorkflowRequest<'a> {
    /// Audio file to transcribe.
    pub audio_path: &'a Path,
    /// Shared runtime services for ASR and downstream NLP steps.
    pub services: PipelineServices<'a>,
    /// Transcribe configuration for this file.
    pub options: &'a TranscribeOptions,
    /// Optional progress sink.
    pub progress: Option<&'a ProgressSender>,
    /// Optional debug directory for pipeline traces.
    pub debug_dir: Option<&'a Path>,
}

/// Per-file audio-to-CHAT workflow.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct TranscribeWorkflow;

#[async_trait]
impl PerFileWorkflow for TranscribeWorkflow {
    type Output = String;
    type Request<'a>
        = TranscribeWorkflowRequest<'a>
    where
        Self: 'a;

    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError> {
        run_transcribe_pipeline(
            request.audio_path,
            request.services,
            request.options,
            request.progress,
            request.debug_dir,
        )
        .await
    }
}
