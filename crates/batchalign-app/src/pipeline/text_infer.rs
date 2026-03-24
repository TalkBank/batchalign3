//! Shared single-file text-infer pipeline skeleton.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use crate::api::EngineVersion;
use crate::api::LanguageCode3;
use crate::cache::UtteranceCache;
use crate::worker::pool::WorkerPool;
use batchalign_chat_ops::parse::{is_dummy, parse_lenient};
use batchalign_chat_ops::serialize::to_chat_string;
use batchalign_chat_ops::validate::{ValidityLevel, validate_output, validate_to_level};
use batchalign_chat_ops::{CacheKey, ChatFile};
use tracing::warn;

use crate::error::ServerError;
use crate::params::CachePolicy;
use crate::pipeline::PipelineServices;

type PartitionFuture<'a, State, Item> = Pin<
    Box<
        dyn Future<Output = (HashMap<usize, State>, Vec<CacheKey>, Vec<(usize, Item)>)> + Send + 'a,
    >,
>;
type InferFuture<'a, Response> =
    Pin<Box<dyn Future<Output = Result<Vec<Response>, ServerError>> + Send + 'a>>;
type CachePutFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

type PartitionFn<Item, State> = for<'a> fn(
    &'a [(usize, Item)],
    &'a LanguageCode3,
    &'a UtteranceCache,
    &'a EngineVersion,
    CachePolicy,
) -> PartitionFuture<'a, State, Item>;
type InferFn<Item, Response> =
    for<'a> fn(&'a WorkerPool, &'a [(usize, Item)], &'a LanguageCode3) -> InferFuture<'a, Response>;
type IntegrateFn<Item, State, Response> =
    fn(&mut HashMap<usize, State>, &[(usize, Item)], &[Response]);
type CachePutFn<Item, Response> = for<'a> fn(
    &'a UtteranceCache,
    &'a [CacheKey],
    &'a [(usize, Item)],
    &'a [Response],
    &'a LanguageCode3,
    &'a EngineVersion,
) -> CachePutFuture<'a>;

/// Hooks for a cached text-only single-file pipeline.
pub(crate) struct CachedTextPipelineHooks<Item, State, Response> {
    /// User-visible command name for validation and error strings.
    pub command: &'static str,
    /// Pre-validation gate required by the command.
    pub validity: ValidityLevel,
    /// Extract worker payloads from the parsed chat file.
    pub collect: fn(&ChatFile) -> Vec<(usize, Item)>,
    /// Partition payloads into cache hits and misses.
    pub partition: PartitionFn<Item, State>,
    /// Run worker inference for cache misses.
    pub infer: InferFn<Item, Response>,
    /// Merge inferred responses into the final application map.
    pub integrate: IntegrateFn<Item, State, Response>,
    /// Persist cacheable responses.
    pub cache_put: CachePutFn<Item, Response>,
    /// Apply all cached + inferred results to the parsed chat file.
    pub apply: fn(&mut ChatFile, &HashMap<usize, State>),
}

/// Run a cached text-only pipeline for a single CHAT file.
pub(crate) async fn run_cached_text_pipeline<Item, State, Response>(
    chat_text: &str,
    lang: &LanguageCode3,
    services: PipelineServices<'_>,
    cache_policy: CachePolicy,
    hooks: CachedTextPipelineHooks<Item, State, Response>,
) -> Result<String, ServerError> {
    let parser = batchalign_chat_ops::parse::TreeSitterParser::new()
        .expect("tree-sitter CHAT grammar must load");
    let (mut chat_file, parse_errors) = parse_lenient(&parser, chat_text);
    if !parse_errors.is_empty() {
        warn!(
            command = hooks.command,
            num_errors = parse_errors.len(),
            "Parse errors in input (continuing with recovery)"
        );
    }

    if is_dummy(&chat_file) {
        return Ok(to_chat_string(&chat_file));
    }

    if let Err(errors) = validate_to_level(&chat_file, parse_errors.len(), hooks.validity) {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        return Err(ServerError::Validation(format!(
            "{} pre-validation failed: {}",
            hooks.command,
            msgs.join("; ")
        )));
    }

    let batch_items = (hooks.collect)(&chat_file);
    if batch_items.is_empty() {
        return Ok(to_chat_string(&chat_file));
    }

    let (hits, miss_keys, misses) = (hooks.partition)(
        &batch_items,
        lang,
        services.cache,
        services.engine_version,
        cache_policy,
    )
    .await;

    let mut state_map = hits;
    if !misses.is_empty() {
        let responses = (hooks.infer)(services.pool, &misses, lang).await?;
        (hooks.integrate)(&mut state_map, &misses, &responses);
        (hooks.cache_put)(
            services.cache,
            &miss_keys,
            &misses,
            &responses,
            lang,
            services.engine_version,
        )
        .await;
    }

    (hooks.apply)(&mut chat_file, &state_map);

    if let Err(errors) = validate_output(&chat_file, hooks.command) {
        let msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
        warn!(command = hooks.command, errors = ?msgs, "post-validation warnings (non-fatal)");
    }

    Ok(to_chat_string(&chat_file))
}
