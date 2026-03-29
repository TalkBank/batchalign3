//! Morphosyntax pipeline built on the internal stage runner.

use std::collections::HashMap;

use batchalign_chat_ops::morphosyntax::{
    BatchItemWithPosition, MultilingualPolicy, MwtDict, TokenizationMode, cache_key,
    clear_morphosyntax, collect_payloads, declared_languages, extract_strings, inject_results,
    validate_mor_alignment,
};
use batchalign_chat_ops::nlp::UdResponse;
use batchalign_chat_ops::parse::{is_dummy, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{CacheKey, ChatFile, LanguageCode};
use tracing::warn;

use crate::api::LanguageCode3;
use crate::error::ServerError;
use crate::morphosyntax::{cache_put_entries, infer_batch, inject_cache_hits, partition_by_cache};
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;
use crate::pipeline::plan::{PipelinePlan, StageFuture, StageId, StageSpec, run_plan};

/// Per-file morphosyntax pipeline state.
pub(crate) struct MorphosyntaxPipelineContext<'a> {
    /// Shared services for the run.
    pub services: PipelineServices<'a>,
    /// Original chat text.
    pub chat_text: &'a str,
    /// Job language.
    pub lang: &'a LanguageCode3,
    /// Injection tokenization mode.
    pub tokenization_mode: TokenizationMode,
    /// Multilingual payload collection policy.
    pub multilingual_policy: MultilingualPolicy,
    /// Cache lookup policy.
    pub cache_policy: CachePolicy,
    /// MWT lexicon for retokenization overrides and cache key differentiation.
    pub mwt: &'a MwtDict,
    /// Parsed chat file.
    pub chat_file: Option<ChatFile>,
    /// Parse error count from lenient parse.
    pub parse_error_count: usize,
    /// Whether the file is a dummy transcript.
    pub is_dummy: bool,
    /// Collected worker payloads.
    pub batch_items: Vec<BatchItemWithPosition>,
    /// Cache hits keyed by utterance line index.
    pub hits: HashMap<usize, serde_json::Value>,
    /// Cache keys for misses.
    pub miss_keys: Vec<CacheKey>,
    /// Cache misses needing worker inference.
    pub misses: Vec<BatchItemWithPosition>,
    /// Line indices corresponding to misses.
    pub miss_line_indices: Vec<usize>,
    /// Inferred worker responses.
    pub ud_responses: Vec<UdResponse>,
    /// Final serialized output.
    pub final_chat_text: Option<String>,
}

impl<'a> MorphosyntaxPipelineContext<'a> {
    fn new(
        chat_text: &'a str,
        lang: &'a LanguageCode3,
        services: PipelineServices<'a>,
        tokenization_mode: TokenizationMode,
        cache_policy: CachePolicy,
        multilingual_policy: MultilingualPolicy,
        mwt: &'a MwtDict,
    ) -> Self {
        Self {
            services,
            chat_text,
            lang,
            tokenization_mode,
            multilingual_policy,
            cache_policy,
            mwt,
            chat_file: None,
            parse_error_count: 0,
            is_dummy: false,
            batch_items: Vec::new(),
            hits: HashMap::new(),
            miss_keys: Vec::new(),
            misses: Vec::new(),
            miss_line_indices: Vec::new(),
            ud_responses: Vec::new(),
            final_chat_text: None,
        }
    }
}

/// Run the morphosyntax pipeline for a single CHAT file.
pub(crate) async fn run_morphosyntax_pipeline(
    chat_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    tokenization_mode: TokenizationMode,
    cache_policy: CachePolicy,
    multilingual_policy: MultilingualPolicy,
    mwt: &MwtDict,
) -> Result<String, ServerError> {
    let plan = morphosyntax_plan();
    let mut ctx = MorphosyntaxPipelineContext::new(
        chat_text,
        lang,
        services,
        tokenization_mode,
        cache_policy,
        multilingual_policy,
        mwt,
    );
    let _ = run_plan("morphotag", &plan, &mut ctx, None).await?;
    ctx.final_chat_text.ok_or_else(|| {
        ServerError::Validation("morphotag pipeline completed without output".to_string())
    })
}

fn morphosyntax_plan<'a>() -> PipelinePlan<MorphosyntaxPipelineContext<'a>> {
    PipelinePlan::new(vec![
        StageSpec::new(StageId::Parse, vec![], always_enabled, stage_parse),
        StageSpec::new(
            StageId::PreValidate,
            vec![StageId::Parse],
            always_enabled,
            stage_prevalidate,
        ),
        StageSpec::new(
            StageId::ClearExisting,
            vec![StageId::PreValidate],
            always_enabled,
            stage_clear_existing,
        ),
        StageSpec::new(
            StageId::CollectPayloads,
            vec![StageId::ClearExisting],
            always_enabled,
            stage_collect_payloads,
        ),
        StageSpec::new(
            StageId::PartitionCache,
            vec![StageId::CollectPayloads],
            always_enabled,
            stage_partition_cache,
        ),
        StageSpec::new(
            StageId::InjectCacheHits,
            vec![StageId::PartitionCache],
            always_enabled,
            stage_inject_cache_hits,
        ),
        StageSpec::new(
            StageId::Infer,
            vec![StageId::InjectCacheHits],
            always_enabled,
            stage_infer,
        ),
        StageSpec::new(
            StageId::ApplyResults,
            vec![StageId::Infer],
            always_enabled,
            stage_apply_results,
        ),
        StageSpec::new(
            StageId::CacheStore,
            vec![StageId::ApplyResults],
            always_enabled,
            stage_cache_store,
        ),
        StageSpec::new(
            StageId::PostValidate,
            vec![StageId::CacheStore],
            always_enabled,
            stage_postvalidate,
        ),
        StageSpec::new(
            StageId::Serialize,
            vec![StageId::PostValidate],
            always_enabled,
            stage_serialize,
        ),
    ])
}

fn always_enabled(_: &MorphosyntaxPipelineContext<'_>) -> bool {
    true
}

fn stage_parse<'a, 'ctx>(ctx: &'a mut MorphosyntaxPipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
            .expect("tree-sitter CHAT grammar must load");
        let (chat_file, parse_errors) = parse_lenient(&parser, ctx.chat_text);
        if !parse_errors.is_empty() {
            warn!(
                num_errors = parse_errors.len(),
                "Parse errors in morphosyntax input (continuing with recovery)"
            );
        }
        ctx.parse_error_count = parse_errors.len();
        ctx.is_dummy = is_dummy(&chat_file);
        ctx.chat_file = Some(chat_file);
        Ok(())
    })
}

fn stage_prevalidate<'a, 'ctx>(ctx: &'a mut MorphosyntaxPipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.is_dummy {
            return Ok(());
        }
        let chat_file = ctx.chat_file.as_ref().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before morphotag pre-validation".into())
        })?;
        if let Err(errors) = validate_to_level(
            chat_file,
            ctx.parse_error_count,
            ValidityLevel::MainTierValid,
        ) {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            return Err(ServerError::Validation(format!(
                "morphotag pre-validation failed: {}",
                msgs.join("; ")
            )));
        }
        Ok(())
    })
}

fn stage_clear_existing<'a, 'ctx>(
    ctx: &'a mut MorphosyntaxPipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.is_dummy {
            return Ok(());
        }
        let chat_file = ctx.chat_file.as_mut().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before clearing morphosyntax".into())
        })?;
        clear_morphosyntax(chat_file);
        Ok(())
    })
}

fn stage_collect_payloads<'a, 'ctx>(
    ctx: &'a mut MorphosyntaxPipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.is_dummy {
            return Ok(());
        }
        let primary_lang = LanguageCode::new(ctx.lang.as_ref());
        let chat_file = ctx.chat_file.as_ref().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before payload collection".into())
        })?;
        let langs = declared_languages(chat_file, &primary_lang);
        let (batch_items, _total) =
            collect_payloads(chat_file, &primary_lang, &langs, ctx.multilingual_policy);
        ctx.batch_items = batch_items;
        Ok(())
    })
}

fn stage_partition_cache<'a, 'ctx>(
    ctx: &'a mut MorphosyntaxPipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.is_dummy || ctx.batch_items.is_empty() {
            return Ok(());
        }
        if ctx.cache_policy.should_skip() {
            ctx.miss_keys = ctx
                .batch_items
                .iter()
                .map(|(_, _, item, _)| {
                    let retok = ctx.tokenization_mode == TokenizationMode::StanzaRetokenize;
                    cache_key(&item.words, &item.lang, ctx.mwt, retok)
                })
                .collect();
            ctx.misses = ctx.batch_items.clone();
        } else {
            let retok = ctx.tokenization_mode == TokenizationMode::StanzaRetokenize;
            let (hits, miss_keys, misses) = partition_by_cache(
                &ctx.batch_items,
                ctx.services.cache,
                ctx.services.engine_version,
                ctx.mwt,
                retok,
            )
            .await;
            ctx.hits = hits;
            ctx.miss_keys = miss_keys;
            ctx.misses = misses;
        }
        Ok(())
    })
}

fn stage_inject_cache_hits<'a, 'ctx>(
    ctx: &'a mut MorphosyntaxPipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.hits.is_empty() {
            return Ok(());
        }
        let chat_file = ctx.chat_file.as_mut().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before cache-hit injection".into())
        })?;
        inject_cache_hits(chat_file, &ctx.hits)?;
        Ok(())
    })
}

fn stage_infer<'a, 'ctx>(ctx: &'a mut MorphosyntaxPipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.misses.is_empty() {
            return Ok(());
        }
        ctx.miss_line_indices = ctx.misses.iter().map(|(idx, ..)| *idx).collect();
        let lang_code = ctx.lang.clone();
        let retokenize = ctx.tokenization_mode == TokenizationMode::StanzaRetokenize;
        ctx.ud_responses = infer_batch(
            ctx.services.pool,
            &ctx.misses,
            &lang_code,
            ctx.mwt,
            retokenize,
            None,
        )
        .await?;
        Ok(())
    })
}

fn stage_apply_results<'a, 'ctx>(
    ctx: &'a mut MorphosyntaxPipelineContext<'ctx>,
) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.misses.is_empty() {
            return Ok(());
        }
        let chat_file = ctx.chat_file.as_mut().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before result injection".into())
        })?;
        let lang_code = LanguageCode::new(ctx.lang.as_ref());
        let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
            .expect("tree-sitter CHAT grammar must load");
        let _retokenize_traces = inject_results(
            &parser,
            chat_file,
            std::mem::take(&mut ctx.misses),
            std::mem::take(&mut ctx.ud_responses),
            &lang_code,
            ctx.tokenization_mode,
            ctx.mwt,
        )
        .map_err(|e| ServerError::Validation(format!("Result injection failed: {e}")))?;

        let alignment_warnings = validate_mor_alignment(chat_file);
        for warning in &alignment_warnings {
            warn!(warning = %warning, "Morphosyntax alignment mismatch");
        }
        Ok(())
    })
}

fn stage_cache_store<'a, 'ctx>(ctx: &'a mut MorphosyntaxPipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.miss_line_indices.is_empty() {
            return Ok(());
        }
        let chat_file = ctx.chat_file.as_ref().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before cache store".into())
        })?;
        match extract_strings(chat_file, &ctx.miss_line_indices) {
            Ok(entries) => {
                cache_put_entries(
                    ctx.services.cache,
                    &ctx.miss_keys,
                    &entries,
                    ctx.services.engine_version,
                )
                .await;
            }
            Err(e) => {
                warn!(error = %e, "Failed to extract strings for caching (non-fatal)");
            }
        }
        Ok(())
    })
}

fn stage_postvalidate<'a, 'ctx>(ctx: &'a mut MorphosyntaxPipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        if ctx.is_dummy {
            return Ok(());
        }
        let chat_file = ctx.chat_file.as_ref().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before morphotag post-validation".into())
        })?;
        if let Err(errors) = validate_output(chat_file, "morphotag") {
            let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
            warn!(errors = ?msgs, "morphotag post-validation warnings (non-fatal)");
        }
        Ok(())
    })
}

fn stage_serialize<'a, 'ctx>(ctx: &'a mut MorphosyntaxPipelineContext<'ctx>) -> StageFuture<'a> {
    Box::pin(async move {
        let chat_file = ctx.chat_file.as_ref().ok_or_else(|| {
            ServerError::Validation("Parsed chat missing before morphotag serialize".into())
        })?;
        ctx.final_chat_text = Some(to_chat_string(chat_file));
        Ok(())
    })
}
