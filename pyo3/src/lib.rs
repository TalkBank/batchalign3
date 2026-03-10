//! PyO3 bridge between batchalign (Python) and the talkbank Rust crates.
//!
//! All CHAT manipulation happens through AST types from `talkbank-model`.
//! The Python pipeline passes an opaque `ParsedChat` handle between engines:
//! parse once, mutate in place, serialize once.
//!
//! # Module Organization
//!
//! - `parsed_chat` — `#[pymethods]` on `ParsedChat`, split by domain:
//!   - `mod.rs` — constructors, serialization, validation, metadata, simple mutations
//!   - `morphosyntax` — %mor/%gra annotation methods
//!   - `fa` — forced alignment methods
//!   - `text` — translation and utterance segmentation methods
//!   - `speakers` — speaker reassignment and utterance timing methods
//!   - `cleanup` — disfluency and retrace marker methods
//! - `forced_alignment` — FA grouping logic, timing injection, %wor tier generation
//! - `extract` — NLP word extraction from AST (domains: mor, wor, pho, sin)
//! - `inject` — Morphosyntax/retokenize injection into AST from callback response
//! - `dp_align` — Hirschberg DP alignment (used by FA and utterance timing recovery)
//! - `utterance_segmentation` — Utterance splitting based on segmentation callback
//! - `retokenize` — Token retokenization (maps Stanza re-tokenized output back to CHAT words)
//! - `parse` — Pure-Rust parse helpers
//! - `metadata` — Metadata extraction from CHAT headers
//! - `morphosyntax_ops` — Morphosyntax inner functions (per-utterance and batched)
//! - `fa_ops` — Forced alignment orchestration
//! - `text_ops` — Translation and utterance segmentation inner functions
//! - `cleanup_ops` — Disfluency and retrace markers
//! - `speaker_ops` — Speaker reassignment and utterance timing
//! - `tier_ops` — Dependent tier management
//! - `build` — Build CHAT files from JSON transcript descriptions
//! - `pyfunctions` — Standalone #[pyfunction]s
//! - `revai` — PyO3 wrappers over the shared Rust Rev.AI client
//!
//! # Related CHAT Manual Sections
//!
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Format>
//! - <https://talkbank.org/0info/manuals/CHAT.html#File_Headers>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Main_Tier>
//! - <https://talkbank.org/0info/manuals/CHAT.html#Dependent_Tiers>

mod dp_align;
mod extract;
mod forced_alignment;
mod hk_asr_bridge;
mod inject;
pub(crate) mod nlp;
mod retokenize;
mod utterance_segmentation;

mod build;
mod cleanup_ops;
mod cli_entry;
mod fa_ops;
mod metadata;
mod morphosyntax_ops;
mod parse;
mod parsed_chat;
mod provider_pipeline;
mod py_json_bridge;
pub(crate) mod pyfunctions;
mod revai;
mod speaker_ops;
mod text_ops;
mod tier_ops;
mod worker_artifacts;
mod worker_asr_exec;
mod worker_fa_exec;
mod worker_media_exec;
mod worker_protocol;
mod worker_text_results;

#[cfg(test)]
mod test_helpers;
#[cfg(test)]
mod tests;

use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Serde structs for typed JSON deserialization
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
pub(crate) struct AsrWordJson {
    pub(crate) word: String,
    pub(crate) start_ms: u64,
    pub(crate) end_ms: u64,
    #[serde(default)]
    pub(crate) word_id: Option<String>,
}

#[derive(serde::Deserialize)]
pub(crate) struct TierEntryJson {
    pub(crate) utterance_index: usize,
    pub(crate) label: String,
    pub(crate) content: String,
}

#[cfg(test)]
#[derive(serde::Deserialize)]
pub(crate) struct FaTimingsResponse {
    pub(crate) timings: Vec<Option<[u64; 2]>>,
}

// ---------------------------------------------------------------------------
// ParsedChat pyclass: wraps a ChatFile for stateful manipulation
// ---------------------------------------------------------------------------

/// A parsed CHAT file handle.
///
/// Wraps the Rust AST (`ChatFile`) and exposes methods for querying and
/// mutating the document.  Python callers can hold the handle across
/// multiple mutations without paying re-parse costs.
#[pyclass]
pub(crate) struct ParsedChat {
    pub(crate) inner: talkbank_model::model::ChatFile,
    pub(crate) warnings: Vec<talkbank_model::ParseError>,
}

// All #[pymethods] impl blocks live in the `parsed_chat` submodules.

// ===========================================================================
// Module registration
// ===========================================================================

/// Initialize tracing subscriber for structured logging.
///
/// Uses the `BATCHALIGN_RUST_LOG` env var for filtering (default: `warn`).
/// Safe to call multiple times — `try_init` is a no-op if already initialized.
fn init_rust_tracing() {
    use tracing_subscriber::EnvFilter;
    let filter =
        EnvFilter::try_from_env("BATCHALIGN_RUST_LOG").unwrap_or_else(|_| EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}

/// batchalign_core -- Rust-powered CHAT handling for Batchalign.
#[pymodule]
fn batchalign_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_rust_tracing();
    m.add_class::<ParsedChat>()?;
    m.add_function(wrap_pyfunction!(pyfunctions::parse_and_serialize, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::extract_nlp_words, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::py_dp_align, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::build_chat, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::add_dependent_tiers, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::extract_timed_tiers, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::chat_terminators, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::chat_mor_punct, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::align_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::wer_conform, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::wer_compute, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::wer_metrics, m)?)?;
    m.add_function(wrap_pyfunction!(
        provider_pipeline::run_provider_pipeline,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        provider_pipeline::unwrap_batch_infer_results,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        provider_pipeline::call_batch_infer_provider,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_asr_exec::execute_asr_request_v2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_protocol::dispatch_protocol_message,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_artifacts::find_worker_attachment_by_id,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_artifacts::load_worker_json_attachment,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_artifacts::load_worker_prepared_text_json,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_artifacts::load_worker_prepared_audio_f32le_bytes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_fa_exec::execute_forced_alignment_request_v2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_media_exec::execute_opensmile_request_v2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_media_exec::execute_avqi_request_v2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_media_exec::execute_speaker_request_v2,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        worker_text_results::normalize_text_task_result,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        hk_asr_bridge::clean_funaudio_segment_text,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        hk_asr_bridge::funaudio_segments_to_asr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        hk_asr_bridge::tencent_result_detail_to_asr,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(hk_asr_bridge::aliyun_sentences_to_asr, m)?)?;

    // Cantonese normalization
    m.add_function(wrap_pyfunction!(pyfunctions::normalize_cantonese, m)?)?;
    m.add_function(wrap_pyfunction!(pyfunctions::cantonese_char_tokens, m)?)?;

    // Rev.AI native client
    m.add_function(wrap_pyfunction!(revai::rev_transcribe, m)?)?;
    m.add_function(wrap_pyfunction!(revai::rev_get_timed_words, m)?)?;
    m.add_function(wrap_pyfunction!(revai::rev_submit, m)?)?;
    m.add_function(wrap_pyfunction!(revai::rev_poll, m)?)?;
    m.add_function(wrap_pyfunction!(revai::rev_poll_timed_words, m)?)?;

    // CLI entry point (used by [project.scripts] console command)
    m.add_function(wrap_pyfunction!(cli_entry::cli_main, m)?)?;

    Ok(())
}
