//! Transport adapter for forced-alignment worker inference.
//!
//! The FA pipeline delegates worker interaction through this module so the
//! orchestration code can ask for "timings for these miss groups" without
//! depending on the concrete worker-protocol V2 request-building details.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::{DurationMs, WorkerLanguage};
use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::fa_result_v2::parse_forced_alignment_result_v2;
use crate::worker::request_builder_v2::{
    ForcedAlignmentBuildInputV2, PreparedFaRequestIdsV2, build_forced_alignment_request_v2,
};
use batchalign_chat_ops::fa::{FaEngineType, FaGroup, FaInferItem, FaTimingMode, WordTiming};
use tracing::warn;

static NEXT_FA_REQUEST_NAMESPACE: AtomicU64 = AtomicU64::new(1);

/// Shared FA worker batch input independent of the concrete worker transport.
pub(crate) struct FaWorkerBatch<'a> {
    /// Precomputed cleaned word texts keyed by group index.
    pub word_texts: &'a [Vec<String>],
    /// FA groups for the current file.
    pub groups: &'a [FaGroup],
    /// Indices of groups that still need worker inference.
    pub miss_indices: &'a [usize],
    /// Source audio path for the current file.
    pub audio_path: &'a Path,
    /// Worker-runtime language hint for FA model bootstrap.
    pub worker_lang: WorkerLanguage,
    /// FA backend selected by the Rust control plane.
    pub engine: FaEngineType,
    /// Timing mode selected by the Rust control plane.
    pub timing_mode: FaTimingMode,
}

/// Parsed FA timings for one inferred miss group.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FaWorkerGroupResult {
    /// Original group index inside the current file.
    pub group_index: usize,
    /// Parsed timings in the established Rust FA timing domain.
    pub timings: Vec<Option<WordTiming>>,
}

/// Narrow transport adapter for FA worker inference.
#[derive(Clone, Copy)]
pub(crate) enum FaWorkerTransport<'a> {
    /// Live typed worker-protocol V2 transport using prepared artifacts.
    V2 {
        /// Shared pipeline services with worker-pool access.
        services: PipelineServices<'a>,
    },
}

impl<'a> FaWorkerTransport<'a> {
    /// Return the production FA worker transport.
    pub(crate) fn production(services: PipelineServices<'a>) -> Self {
        Self::V2 { services }
    }

    /// Infer timings for the requested FA miss groups.
    pub(crate) async fn infer_groups(
        self,
        batch: FaWorkerBatch<'_>,
    ) -> Result<Vec<FaWorkerGroupResult>, ServerError> {
        match self {
            Self::V2 { services } => infer_groups_v2(services, batch).await,
        }
    }
}

/// Dispatch staged worker-protocol V2 requests for each FA miss group and
/// parse successful results back into the established Rust timing domain.
async fn infer_groups_v2(
    services: PipelineServices<'_>,
    batch: FaWorkerBatch<'_>,
) -> Result<Vec<FaWorkerGroupResult>, ServerError> {
    let request_namespace = NEXT_FA_REQUEST_NAMESPACE.fetch_add(1, Ordering::Relaxed);
    let artifacts = PreparedArtifactRuntimeV2::new("fa_v2").map_err(|error| {
        ServerError::Validation(format!("failed to create FA V2 artifact runtime: {error}"))
    })?;

    let mut parsed_results = Vec::with_capacity(batch.miss_indices.len());
    for group_index in batch.miss_indices.iter().copied() {
        let infer_item = build_fa_infer_item(&batch, group_index);
        let request_ids = build_fa_request_ids(request_namespace, group_index);
        let request = build_forced_alignment_request_v2(
            artifacts.store(),
            ForcedAlignmentBuildInputV2 {
                ids: &request_ids,
                infer_item: &infer_item,
                engine: batch.engine,
            },
        )
        .await
        .map_err(|error| {
            ServerError::Validation(format!(
                "failed to build worker protocol V2 FA request for group {group_index}: {error}"
            ))
        })?;

        let response = services
            .pool
            .dispatch_execute_v2(&batch.worker_lang, &request)
            .await
            .map_err(ServerError::Worker)?;

        match parse_forced_alignment_result_v2(
            &response,
            &batch.groups[group_index].words,
            DurationMs(batch.groups[group_index].audio_start_ms()),
            batch.timing_mode,
        ) {
            Ok(timings) => parsed_results.push(FaWorkerGroupResult {
                group_index,
                timings,
            }),
            Err(error) => {
                warn!(
                    group = group_index,
                    error = %error,
                    "worker protocol V2 FA response parsing failed for group"
                );
            }
        }
    }

    Ok(parsed_results)
}

/// Build one production-domain `FaInferItem` from the transport-neutral batch
/// view.
fn build_fa_infer_item(batch: &FaWorkerBatch<'_>, group_index: usize) -> FaInferItem {
    let group = &batch.groups[group_index];
    FaInferItem {
        words: batch.word_texts[group_index].clone(),
        word_ids: group.words.iter().map(|word| word.stable_id()).collect(),
        word_utterance_indices: group
            .words
            .iter()
            .map(|word| word.utterance_index.raw())
            .collect(),
        word_utterance_word_indices: group
            .words
            .iter()
            .map(|word| word.utterance_word_index.raw())
            .collect(),
        audio_path: batch.audio_path.to_string_lossy().into_owned(),
        audio_start_ms: group.audio_start_ms(),
        audio_end_ms: group.audio_end_ms(),
        timing_mode: batch.timing_mode,
    }
}

/// Build unique request and artifact ids for one FA V2 request.
///
/// The request namespace is allocated once per `infer_groups_v2` call so two
/// concurrent files cannot collide on `fa-v2-request-0`, `fa-v2-request-1`,
/// and so on while sharing the same GPU worker.
fn build_fa_request_ids(request_namespace: u64, group_index: usize) -> PreparedFaRequestIdsV2 {
    PreparedFaRequestIdsV2::new(
        format!("fa-v2-request-{request_namespace}-{group_index}"),
        format!("fa-v2-payload-{request_namespace}-{group_index}"),
        format!("fa-v2-audio-{request_namespace}-{group_index}"),
    )
}

#[cfg(test)]
mod tests {
    use batchalign_chat_ops::fa::{FaWord, TimeSpan};
    use batchalign_chat_ops::indices::{UtteranceIdx, WordIdx};

    use super::*;

    /// Build a small FA word for transport unit tests.
    fn make_word(index: usize, text: &str) -> FaWord {
        FaWord {
            utterance_index: UtteranceIdx(0),
            utterance_word_index: WordIdx(index),
            text: text.into(),
        }
    }

    #[test]
    fn builds_fa_infer_item_from_transport_neutral_batch() {
        let word_texts = vec![vec!["hello".to_string(), "world".to_string()]];
        let groups = vec![FaGroup {
            audio_span: TimeSpan::new(100, 900),
            words: vec![make_word(0, "hello"), make_word(1, "world")],
            utterance_indices: vec![UtteranceIdx(0)],
        }];
        let batch = FaWorkerBatch {
            word_texts: &word_texts,
            groups: &groups,
            miss_indices: &[0],
            audio_path: Path::new("/tmp/input.wav"),
            worker_lang: WorkerLanguage::from(crate::api::LanguageCode3::eng()),
            engine: FaEngineType::WhisperFa,
            timing_mode: FaTimingMode::WithPauses,
        };

        let item = build_fa_infer_item(&batch, 0);
        assert_eq!(item.words, vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(
            item.word_ids,
            vec!["u0:w0".to_string(), "u0:w1".to_string()]
        );
        assert_eq!(item.word_utterance_indices, vec![0, 0]);
        assert_eq!(item.word_utterance_word_indices, vec![0, 1]);
        assert_eq!(item.audio_path, "/tmp/input.wav");
        assert_eq!(item.audio_start_ms, 100);
        assert_eq!(item.audio_end_ms, 900);
        assert_eq!(item.timing_mode, FaTimingMode::WithPauses);
    }

    #[test]
    fn builds_namespaced_v2_request_ids_from_group_index() {
        let ids = build_fa_request_ids(42, 7);
        assert_eq!(&*ids.request_id, "fa-v2-request-42-7");
        assert_eq!(&*ids.payload_ref_id, "fa-v2-payload-42-7");
        assert_eq!(&*ids.audio_ref_id, "fa-v2-audio-42-7");
    }

    #[test]
    fn namespaces_v2_request_ids_across_concurrent_files() {
        let first = build_fa_request_ids(1, 0);
        let second = build_fa_request_ids(2, 0);

        assert_ne!(first.request_id, second.request_id);
        assert_ne!(first.payload_ref_id, second.payload_ref_id);
        assert_ne!(first.audio_ref_id, second.audio_ref_id);
    }
}
