//! Internal pipeline helpers for command-local orchestration.
//!
//! This module intentionally stays private to `batchalign-server`. It is not a
//! general executor; it is a small sequential stage runner used to make
//! per-command orchestration explicit.

use crate::api::EngineVersion;
use crate::cache::UtteranceCache;
use crate::worker::pool::WorkerPool;

pub(crate) mod morphosyntax;
pub(crate) mod plan;
pub(crate) mod text_infer;
pub(crate) mod transcribe;

/// Shared services used by pipeline helpers.
#[derive(Clone, Copy)]
pub(crate) struct PipelineServices<'a> {
    /// Worker pool for inference.
    pub pool: &'a WorkerPool,
    /// Shared utterance cache.
    pub cache: &'a UtteranceCache,
    /// Current engine version for cache keying.
    pub engine_version: &'a EngineVersion,
}
