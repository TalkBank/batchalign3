//! Trait-oriented composite benchmark workflow.

use std::path::Path;

use async_trait::async_trait;
use batchalign_chat_ops::morphosyntax::MwtDict;

use crate::api::{ChatText, LanguageCode3};
use crate::error::ServerError;
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::runner::util::ProgressSender;
use crate::transcribe::TranscribeOptions;

use super::compare::{
    CompareMaterializedOutputs, CompareWorkflow, CompareWorkflowRequest,
    MainAnnotatedCompareMaterializer,
};
use super::transcribe::{TranscribeWorkflow, TranscribeWorkflowRequest};
use super::{CompositeWorkflow, PerFileWorkflow, ReferenceProjectionWorkflow};

/// Borrowed request bundle for one benchmark workflow execution.
pub(crate) struct BenchmarkWorkflowRequest<'a> {
    /// Audio file to transcribe before comparison.
    pub audio_path: &'a Path,
    /// Gold-standard CHAT transcript to compare against.
    pub gold_text: ChatText<'a>,
    /// Primary language used for comparison and downstream NLP shaping.
    pub lang: &'a LanguageCode3,
    /// Shared worker/cache services used by the transcribe and compare phases.
    pub services: PipelineServices<'a>,
    /// Typed transcription options for the Rust-owned transcribe pipeline.
    pub transcribe_options: &'a TranscribeOptions,
    /// Cache policy used by the compare-side morphosyntax helpers.
    pub cache_policy: CachePolicy,
    /// Multi-word-token dictionary shared with the compare pipeline.
    pub mwt: &'a MwtDict,
    /// Optional progress sink for the transcribe sub-pipeline.
    pub progress: Option<&'a ProgressSender>,
}

/// Composite benchmark workflow: transcribe first, then compare.
pub(crate) struct BenchmarkWorkflow {
    transcribe: TranscribeWorkflow,
    compare: CompareWorkflow<MainAnnotatedCompareMaterializer>,
}

impl Default for BenchmarkWorkflow {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkWorkflow {
    /// Construct the default benchmark workflow.
    pub(crate) fn new() -> Self {
        Self {
            transcribe: TranscribeWorkflow,
            compare: CompareWorkflow::released(),
        }
    }
}

#[async_trait]
impl CompositeWorkflow for BenchmarkWorkflow {
    type Output = CompareMaterializedOutputs;
    type Request<'a>
        = BenchmarkWorkflowRequest<'a>
    where
        Self: 'a;

    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError> {
        let transcribed_chat = self
            .transcribe
            .run(TranscribeWorkflowRequest {
                audio_path: request.audio_path,
                services: request.services,
                options: request.transcribe_options,
                progress: request.progress,
                debug_dir: None,
            })
            .await?;

        self.compare
            .run(CompareWorkflowRequest {
                main_text: ChatText::from(transcribed_chat.as_str()),
                gold_text: request.gold_text,
                lang: request.lang,
                services: request.services,
                cache_policy: request.cache_policy,
                mwt: request.mwt,
            })
            .await
    }
}
