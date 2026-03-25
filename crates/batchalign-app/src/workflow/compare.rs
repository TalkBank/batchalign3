//! Trait-oriented compare workflow and materializers.
//!
//! Two materializer paths:
//! - **Main-annotated** (released): injects `%xsrep` tiers on the main transcript.
//! - **Gold-projected**: per-gold-utterance windowed alignment, timing/`%mor`
//!   projection from main to gold, per-POS metrics. Replicates BA2 `CompareEngine`.

use async_trait::async_trait;
use tracing::{info, warn};

use crate::api::{ChatText, LanguageCode3};
use crate::error::ServerError;
use crate::params::{CachePolicy, MorphosyntaxParams};
use crate::pipeline::PipelineServices;
use batchalign_chat_ops::compare::{
    ComparisonBundle, clear_comparison, compare, format_metrics_csv, inject_comparison,
};
use batchalign_chat_ops::compare::gold_projection::{
    GoldProjectionBundle, apply_gold_projection, compare_gold_projection,
};
use batchalign_chat_ops::morphosyntax::{MultilingualPolicy, MwtDict, TokenizationMode};
use batchalign_chat_ops::parse::parse_lenient;
use batchalign_chat_ops::serialize::to_chat_string;

use super::{Materializer, ReferenceProjectionWorkflow};

/// Try to run morphosyntax on the main transcript. Returns the enriched CHAT
/// text on success, or an error if the worker is unavailable.
async fn try_morphosyntax(request: &CompareWorkflowRequest<'_>) -> Result<String, ServerError> {
    let mor_params = MorphosyntaxParams {
        lang: request.lang,
        tokenization_mode: TokenizationMode::Preserve,
        cache_policy: request.cache_policy,
        multilingual_policy: MultilingualPolicy::ProcessAll,
        mwt: request.mwt,
    };
    crate::morphosyntax::process_morphosyntax(
        request.main_text.as_ref(),
        request.services,
        &mor_params,
    )
    .await
}

/// Current released compare outputs.
pub(crate) struct CompareMaterializedOutputs {
    /// CHAT text for the main transcript annotated with `%xsrep`.
    pub annotated_main_chat: String,
    /// CSV sidecar containing aggregate compare metrics.
    pub metrics_csv: String,
}

/// Gold-projected compare outputs.
///
/// The gold transcript is annotated with timing and `%mor` projected from the
/// main transcript's morphosyntax, plus `%xsrep` comparison tiers. Metrics
/// include per-POS breakdown.
pub(crate) struct GoldProjectedCompareOutputs {
    /// Gold CHAT text with timing, `%mor`, and `%xsrep` projected from main.
    pub projected_gold_chat: String,
    /// CSV sidecar containing aggregate + per-POS compare metrics.
    pub metrics_csv: String,
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
    /// Typed comparison/alignment bundle (main-annotated path).
    pub bundle: ComparisonBundle,
}

/// Typed intermediate artifacts for gold-projected compare workflows.
pub(crate) struct GoldProjectionArtifacts {
    /// Parsed morphotagged main transcript.
    pub main_file: batchalign_chat_ops::ChatFile,
    /// Parsed gold transcript.
    pub gold_file: batchalign_chat_ops::ChatFile,
    /// Gold-projection bundle with per-utterance windowed alignment.
    pub bundle: GoldProjectionBundle,
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

/// Gold-projection materializer: timing + `%mor` + `%xsrep` projection from main to gold.
///
/// Replicates BA2 `CompareEngine` output: the gold transcript carries timing
/// bullets and morphosyntax from matched main words, plus comparison annotations.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct GoldProjectedCompareMaterializer;

impl Materializer<GoldProjectionArtifacts> for GoldProjectedCompareMaterializer {
    type Output = GoldProjectedCompareOutputs;

    fn materialize(
        &self,
        artifacts: GoldProjectionArtifacts,
    ) -> Result<Self::Output, ServerError> {
        let GoldProjectionArtifacts {
            main_file,
            mut gold_file,
            bundle,
        } = artifacts;
        apply_gold_projection(&main_file, &mut gold_file, &bundle);
        Ok(GoldProjectedCompareOutputs {
            projected_gold_chat: to_chat_string(&gold_file),
            metrics_csv: format_metrics_csv(&bundle.metrics),
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

/// Gold-projected compare workflow.
///
/// Uses per-gold-utterance windowed alignment (bag-of-words + DP) and
/// projects timing/`%mor` from main to gold. Replicates BA2 `CompareEngine`.
pub(crate) struct GoldProjectedCompareWorkflow {
    materializer: GoldProjectedCompareMaterializer,
}

impl GoldProjectedCompareWorkflow {
    /// Create a gold-projected compare workflow.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self {
            materializer: GoldProjectedCompareMaterializer,
        }
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
        // Morphosyntax is optional for compare — it enriches POS tags but the
        // core alignment works without it. Try morphosyntax; fall back to raw.
        let main_text = match try_morphosyntax(&request).await {
            Ok(tagged) => tagged,
            Err(e) => {
                warn!(error = %e, "Morphosyntax unavailable, comparing raw transcripts");
                request.main_text.as_ref().to_string()
            }
        };

        let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
            .expect("tree-sitter CHAT grammar must load");
        let (main_file, main_errors) = parse_lenient(&parser, &morphotagged);
        if !main_errors.is_empty() {
            warn!(
                num_errors = main_errors.len(),
                "Parse errors in main (continuing)"
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

#[async_trait]
impl ReferenceProjectionWorkflow for GoldProjectedCompareWorkflow {
    type ArtifactBundle = GoldProjectionArtifacts;
    type Output = GoldProjectedCompareOutputs;
    type Materializer = GoldProjectedCompareMaterializer;
    type Request<'a> = CompareWorkflowRequest<'a>;

    async fn build_artifacts(
        &self,
        request: Self::Request<'_>,
    ) -> Result<Self::ArtifactBundle, ServerError> {
        let main_text = match try_morphosyntax(&request).await {
            Ok(tagged) => tagged,
            Err(e) => {
                warn!(error = %e, "Morphosyntax unavailable, comparing raw transcripts");
                request.main_text.as_ref().to_string()
            }
        };

        let (main_file, main_errors) = parse_lenient(&main_text);
        if !main_errors.is_empty() {
            warn!(
                num_errors = main_errors.len(),
                "Parse errors in main (continuing)"
            );
        }

        let (gold_file, gold_errors) = parse_lenient(request.gold_text.as_ref());
        if !gold_errors.is_empty() {
            warn!(
                num_errors = gold_errors.len(),
                "Parse errors in gold file (continuing)"
            );
        }

        // Gold-projected compare: per-gold-utterance windowed alignment
        let bundle = compare_gold_projection(&main_file, &gold_file);

        info!(
            matches = bundle.metrics.matches,
            insertions = bundle.metrics.insertions,
            deletions = bundle.metrics.deletions,
            wer = %format!("{:.4}", bundle.metrics.wer),
            "Gold-projected compare alignment complete"
        );

        Ok(GoldProjectionArtifacts {
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
