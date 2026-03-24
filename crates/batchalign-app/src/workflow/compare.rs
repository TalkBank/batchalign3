//! Trait-oriented compare workflow and materializers.

use async_trait::async_trait;
use tracing::{info, warn};

use crate::api::{ChatText, LanguageCode3};
use crate::error::ServerError;
use crate::params::{CachePolicy, MorphosyntaxParams};
use crate::pipeline::PipelineServices;
use batchalign_chat_ops::compare::{
    ComparisonBundle, clear_comparison, compare, format_metrics_csv, inject_comparison,
};
use batchalign_chat_ops::morphosyntax::{MultilingualPolicy, MwtDict, TokenizationMode};
use batchalign_chat_ops::parse::parse_lenient;
use batchalign_chat_ops::serialize::to_chat_string;

use super::{Materializer, ReferenceProjectionWorkflow};

/// Current released compare outputs.
pub(crate) struct CompareMaterializedOutputs {
    /// CHAT text for the main transcript annotated with `%xsrep`.
    pub annotated_main_chat: String,
    /// CSV sidecar containing aggregate compare metrics.
    pub metrics_csv: String,
}

/// Internal skeletal gold-projection output.
///
/// This exists to make the reference-projection seam explicit for contributors
/// without hardening the final gold-projected semantics before Houjun's BA3
/// compare branch is visible.
#[allow(dead_code)]
pub(crate) struct GoldProjectedCompareOutputs {
    /// Current skeletal gold-shaped CHAT output.
    pub projected_gold_chat: String,
    /// CSV sidecar containing aggregate compare metrics.
    pub metrics_csv: String,
    /// Marker making it explicit that this materializer is only a scaffold.
    pub projection_mode: GoldProjectionMode,
}

/// Status of the current gold-projection materializer.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GoldProjectionMode {
    /// The materializer currently preserves the gold scaffold without doing the
    /// full annotation/timing/dependency projection that the future workflow
    /// will support.
    SkeletalPassthrough,
}

/// Borrowed request bundle for one compare execution.
pub(crate) struct CompareWorkflowRequest<'a> {
    /// Main transcript text.
    pub main_text: ChatText<'a>,
    /// Gold transcript text.
    pub gold_text: ChatText<'a>,
    /// Primary language for morphosyntax and compare shaping.
    pub lang: &'a LanguageCode3,
    /// Shared runtime services.
    pub services: PipelineServices<'a>,
    /// Cache policy used by compare-side morphosyntax.
    pub cache_policy: CachePolicy,
    /// Multi-word-token dictionary shared with morphosyntax.
    pub mwt: &'a MwtDict,
}

/// Typed intermediate artifacts for compare workflows.
pub(crate) struct ComparisonArtifacts {
    /// Parsed morphotagged main transcript.
    pub main_file: batchalign_chat_ops::ChatFile,
    /// Parsed gold transcript.
    pub gold_file: batchalign_chat_ops::ChatFile,
    /// Typed comparison/alignment bundle.
    pub bundle: ComparisonBundle,
}

/// Released main-annotated compare materializer.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MainAnnotatedCompareMaterializer;

impl Materializer<ComparisonArtifacts> for MainAnnotatedCompareMaterializer {
    type Output = CompareMaterializedOutputs;

    fn materialize(&self, artifacts: ComparisonArtifacts) -> Result<Self::Output, ServerError> {
        let ComparisonArtifacts {
            mut main_file,
            bundle,
            ..
        } = artifacts;
        clear_comparison(&mut main_file);
        inject_comparison(&mut main_file, &bundle);
        Ok(CompareMaterializedOutputs {
            annotated_main_chat: to_chat_string(&main_file),
            metrics_csv: format_metrics_csv(&bundle.metrics),
        })
    }
}

/// Skeletal gold-projection materializer retained as a merge-friendly seam.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GoldProjectedCompareSkeletonMaterializer;

#[allow(dead_code)]
impl Materializer<ComparisonArtifacts> for GoldProjectedCompareSkeletonMaterializer {
    type Output = GoldProjectedCompareOutputs;

    fn materialize(&self, artifacts: ComparisonArtifacts) -> Result<Self::Output, ServerError> {
        Ok(GoldProjectedCompareOutputs {
            projected_gold_chat: to_chat_string(&artifacts.gold_file),
            metrics_csv: format_metrics_csv(&artifacts.bundle.metrics),
            projection_mode: GoldProjectionMode::SkeletalPassthrough,
        })
    }
}

/// Reference-projection compare workflow.
pub(crate) struct CompareWorkflow<M> {
    materializer: M,
}

impl<M> CompareWorkflow<M> {
    /// Create a compare workflow with an explicit materializer.
    pub(crate) fn new(materializer: M) -> Self {
        Self { materializer }
    }
}

impl CompareWorkflow<MainAnnotatedCompareMaterializer> {
    /// Current released compare workflow shape.
    pub(crate) fn released() -> Self {
        Self::new(MainAnnotatedCompareMaterializer)
    }
}

impl CompareWorkflow<GoldProjectedCompareSkeletonMaterializer> {
    /// Internal skeletal gold-projection workflow shape.
    #[allow(dead_code)]
    pub(crate) fn skeletal_gold_projection() -> Self {
        Self::new(GoldProjectedCompareSkeletonMaterializer)
    }
}

#[async_trait]
impl<M> ReferenceProjectionWorkflow for CompareWorkflow<M>
where
    M: Materializer<ComparisonArtifacts> + Send + Sync + 'static,
{
    type ArtifactBundle = ComparisonArtifacts;
    type Output = M::Output;
    type Materializer = M;
    type Request<'a>
        = CompareWorkflowRequest<'a>
    where
        Self: 'a;

    async fn build_artifacts(
        &self,
        request: Self::Request<'_>,
    ) -> Result<Self::ArtifactBundle, ServerError> {
        let mor_params = MorphosyntaxParams {
            lang: request.lang,
            tokenization_mode: TokenizationMode::Preserve,
            cache_policy: request.cache_policy,
            multilingual_policy: MultilingualPolicy::ProcessAll,
            mwt: request.mwt,
        };
        let morphotagged = crate::morphosyntax::process_morphosyntax(
            request.main_text.as_ref(),
            request.services,
            &mor_params,
        )
        .await?;

        let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
            .expect("tree-sitter CHAT grammar must load");
        let (main_file, main_errors) = parse_lenient(&parser, &morphotagged);
        if !main_errors.is_empty() {
            warn!(
                num_errors = main_errors.len(),
                "Parse errors in morphotagged main (continuing)"
            );
        }

        let (gold_file, gold_errors) = parse_lenient(&parser, request.gold_text.as_ref());
        if !gold_errors.is_empty() {
            warn!(
                num_errors = gold_errors.len(),
                "Parse errors in gold file (continuing)"
            );
        }

        let bundle = compare(&main_file, &gold_file);

        info!(
            matches = bundle.metrics.matches,
            insertions = bundle.metrics.insertions,
            deletions = bundle.metrics.deletions,
            wer = %format!("{:.4}", bundle.metrics.wer),
            "Compare alignment complete"
        );

        Ok(ComparisonArtifacts {
            main_file,
            gold_file,
            bundle,
        })
    }

    async fn run(&self, request: Self::Request<'_>) -> Result<Self::Output, ServerError> {
        let artifacts = self.build_artifacts(request).await?;
        self.materializer.materialize(artifacts)
    }
}
