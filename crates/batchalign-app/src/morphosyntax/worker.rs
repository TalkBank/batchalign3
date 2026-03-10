//! Worker dispatch for morphosyntax inference and cache storage.

use crate::api::{EngineVersion, LanguageCode3};
use crate::cache::{CacheBackend, UtteranceCache};
use crate::error::ServerError;
use crate::infer_retry::dispatch_execute_v2_with_retry;
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::pool::WorkerPool;
use crate::worker::text_request_v2::{PreparedTextRequestIdsV2, build_morphosyntax_request_v2};
use crate::worker::text_result_v2::parse_morphosyntax_result_v2;
use batchalign_chat_ops::CacheKey;
use batchalign_chat_ops::morphosyntax::{
    BatchItemWithPosition, MorphosyntaxStringsEntry, MwtDict, stanza_raw,
};
use batchalign_chat_ops::nlp::UdResponse;
use tracing::{info, warn};

use super::CACHE_TASK;

/// Send batch items to a worker for NLP inference via batched `execute_v2`.
pub(crate) async fn infer_batch(
    pool: &WorkerPool,
    items: &[BatchItemWithPosition],
    lang: &LanguageCode3,
    mwt: &MwtDict,
) -> Result<Vec<UdResponse>, ServerError> {
    let payload_items: Vec<_> = items.iter().map(|(_, _, item, _)| item.clone()).collect();

    let artifacts = PreparedArtifactRuntimeV2::new("morphosyntax_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create morphosyntax V2 artifact runtime: {error}"
        ))
    })?;
    let request_ids = PreparedTextRequestIdsV2::for_task("morphosyntax");
    let request =
        build_morphosyntax_request_v2(artifacts.store(), &request_ids, lang, &payload_items, mwt)
            .map_err(|error| {
            ServerError::Validation(format!(
                "failed to build morphosyntax V2 worker request: {error}"
            ))
        })?;

    info!(
        num_items = items.len(),
        lang = %lang,
        "Dispatching morphosyntax execute_v2 batch"
    );

    let response = dispatch_execute_v2_with_retry(pool, lang, &request).await?;
    let result = parse_morphosyntax_result_v2(&response).map_err(|error| {
        ServerError::Validation(format!("invalid morphosyntax V2 result: {error}"))
    })?;
    if result.items.len() != items.len() {
        return Err(ServerError::Validation(format!(
            "morphosyntax V2 returned {} items for {} requests",
            result.items.len(),
            items.len()
        )));
    }

    let mut ud_responses = Vec::with_capacity(result.items.len());
    for (i, item_result) in result.items.iter().enumerate() {
        if let Some(error) = &item_result.error {
            warn!(item = i, error = %error, "Infer error for item (using empty response)");
            ud_responses.push(UdResponse {
                sentences: Vec::new(),
            });
            continue;
        }

        if let Some(raw_sentences) = &item_result.raw_sentences {
            let ud = stanza_raw::parse_raw_stanza_output(raw_sentences).map_err(|error| {
                ServerError::Validation(format!(
                    "Failed to parse raw Stanza output for item {i}: {error}"
                ))
            })?;
            ud_responses.push(ud);
            continue;
        }

        warn!(
            item = i,
            "Morphosyntax V2 returned no raw_sentences and no error (using empty response)"
        );
        ud_responses.push(UdResponse {
            sentences: Vec::new(),
        });
    }

    Ok(ud_responses)
}

/// Store morphosyntax results in cache.
pub(crate) async fn cache_put_entries(
    cache: &UtteranceCache,
    keys: &[CacheKey],
    entries: &[MorphosyntaxStringsEntry],
    engine_version: &EngineVersion,
) {
    let ba_version = env!("CARGO_PKG_VERSION");
    let cache_entries: Vec<(String, serde_json::Value)> = keys
        .iter()
        .zip(entries.iter())
        .filter(|(_, e)| !e.mor.is_empty())
        .map(|(key, entry)| {
            let data = serde_json::json!({
                "mor": entry.mor,
                "gra": entry.gra,
            });
            (key.as_str().to_string(), data)
        })
        .collect();

    if !cache_entries.is_empty()
        && let Err(e) = cache
            .put_batch(
                &cache_entries,
                CACHE_TASK.as_str(),
                engine_version,
                ba_version,
            )
            .await
    {
        warn!(error = %e, "Failed to store cache entries (non-fatal)");
    }
}
