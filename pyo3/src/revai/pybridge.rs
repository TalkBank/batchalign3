//! Rev.AI native HTTP client — PyO3 wrappers.

use batchalign_revai::{RevAiClient, SubmitOptions, extract_timed_words};
use pyo3::prelude::*;
use std::path::Path;

/// Submit audio to Rev.AI, poll until complete, return transcript JSON.
///
/// The transcript JSON has the Rev.AI format:
/// `{"monologues": [{"speaker": N, "elements": [...]}]}`.
///
/// Releases the GIL during all HTTP operations.
#[pyfunction]
#[pyo3(signature = (audio_path, api_key, language, speakers_count=None, skip_postprocessing=false, metadata=None))]
pub(crate) fn rev_transcribe(
    py: Python<'_>,
    audio_path: &str,
    api_key: &str,
    language: &str,
    speakers_count: Option<u32>,
    skip_postprocessing: bool,
    metadata: Option<String>,
) -> PyResult<String> {
    let path = Path::new(audio_path).to_owned();
    let key = api_key.to_owned();
    let opts = SubmitOptions {
        language: language.to_owned(),
        speakers_count,
        skip_postprocessing: Some(skip_postprocessing),
        metadata,
    };

    py.detach(move || {
        let client = RevAiClient::new(&key);
        let transcript = client
            .transcribe_blocking(&path, &opts, 30)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        serde_json::to_string(&transcript)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

/// Submit audio to Rev.AI, return timed words for UTR.
///
/// Returns a JSON array: `[{"word": "...", "start_ms": N, "end_ms": N}, ...]`.
/// Uses fixed 15s poll interval (matching Python UTR behavior).
///
/// Releases the GIL during all HTTP operations.
#[pyfunction]
#[pyo3(signature = (audio_path, api_key, language))]
pub(crate) fn rev_get_timed_words(
    py: Python<'_>,
    audio_path: &str,
    api_key: &str,
    language: &str,
) -> PyResult<String> {
    let path = Path::new(audio_path).to_owned();
    let key = api_key.to_owned();
    let opts = SubmitOptions {
        language: language.to_owned(),
        speakers_count: None,
        skip_postprocessing: None,
        metadata: None,
    };

    py.detach(move || {
        let client = RevAiClient::new(&key);
        let transcript = client
            .transcribe_fixed_poll(&path, &opts, 15)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let timed_words = extract_timed_words(&transcript.transcript);
        serde_json::to_string(&timed_words)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

/// Submit audio to Rev.AI without waiting for the result.
///
/// Returns the Rev.AI job ID string. The caller can later poll for results
/// using [`rev_poll`] or [`rev_poll_timed_words`].
///
/// Releases the GIL during the HTTP upload.
#[pyfunction]
#[pyo3(signature = (audio_path, api_key, language, speakers_count=None, skip_postprocessing=false, metadata=None))]
pub(crate) fn rev_submit(
    py: Python<'_>,
    audio_path: &str,
    api_key: &str,
    language: &str,
    speakers_count: Option<u32>,
    skip_postprocessing: bool,
    metadata: Option<String>,
) -> PyResult<String> {
    let path = Path::new(audio_path).to_owned();
    let key = api_key.to_owned();
    let opts = SubmitOptions {
        language: language.to_owned(),
        speakers_count,
        skip_postprocessing: Some(skip_postprocessing),
        metadata,
    };

    py.detach(move || {
        let client = RevAiClient::new(&key);
        let job = client
            .submit_local_file(&path, &opts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(job.id)
    })
}

/// Poll a previously submitted Rev.AI job and return the transcript JSON.
///
/// `max_poll_secs` caps the exponential backoff interval. Polling starts at
/// 5 seconds, doubles every 3 attempts, capped at `max_poll_secs`.
///
/// Releases the GIL during all HTTP operations.
#[pyfunction]
#[pyo3(signature = (job_id, api_key, max_poll_secs=30))]
pub(crate) fn rev_poll(
    py: Python<'_>,
    job_id: &str,
    api_key: &str,
    max_poll_secs: u64,
) -> PyResult<String> {
    let jid = job_id.to_owned();
    let key = api_key.to_owned();

    py.detach(move || {
        let client = RevAiClient::new(&key);
        let transcript = client
            .poll_and_download(&jid, 5, max_poll_secs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        serde_json::to_string(&transcript)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}

/// Poll a previously submitted Rev.AI job and return timed words JSON.
///
/// Uses a fixed poll interval of `poll_secs` seconds (matching UTR behavior).
/// Returns a JSON array: `[{"word": "...", "start_ms": N, "end_ms": N}, ...]`.
///
/// Releases the GIL during all HTTP operations.
#[pyfunction]
#[pyo3(signature = (job_id, api_key, poll_secs=15))]
pub(crate) fn rev_poll_timed_words(
    py: Python<'_>,
    job_id: &str,
    api_key: &str,
    poll_secs: u64,
) -> PyResult<String> {
    let jid = job_id.to_owned();
    let key = api_key.to_owned();

    py.detach(move || {
        let client = RevAiClient::new(&key);
        let transcript = client
            .poll_and_download(&jid, poll_secs, poll_secs)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let timed_words = extract_timed_words(&transcript.transcript);
        serde_json::to_string(&timed_words)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })
}
