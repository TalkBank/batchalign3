//! Helpers for bridging typed Python callback payloads and responses.

use batchalign_chat_ops::fa::{FaInferItem, FaTimingMode};
use batchalign_chat_ops::morphosyntax::MorphosyntaxBatchItem;
use batchalign_chat_ops::nlp::{FaIndexedTiming, FaRawResponse, FaRawToken};
use batchalign_chat_ops::translate::TranslateResponse;
use batchalign_chat_ops::utseg::{UtsegBatchItem, UtsegResponse};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};

pub(crate) fn translation_payload_to_object<'py>(
    py: Python<'py>,
    text: &str,
    speaker: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let payload = PyDict::new(py);
    payload.set_item("text", text)?;
    payload.set_item("speaker", speaker)?;
    Ok(payload.into_any())
}

pub(crate) fn utseg_payload_to_object<'py>(
    py: Python<'py>,
    words: &[&str],
    text: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let payload = PyDict::new(py);
    payload.set_item("words", PyList::new(py, words)?)?;
    payload.set_item("text", text)?;
    Ok(payload.into_any())
}

pub(crate) fn utseg_batch_payload_to_object<'py>(
    py: Python<'py>,
    payloads: &[&UtsegBatchItem],
) -> PyResult<Bound<'py, PyAny>> {
    let list = PyList::empty(py);
    for item in payloads {
        let obj = PyDict::new(py);
        obj.set_item("words", PyList::new(py, &item.words)?)?;
        obj.set_item("text", &item.text)?;
        list.append(obj)?;
    }
    Ok(list.into_any())
}

pub(crate) fn fa_payload_to_object<'py>(
    py: Python<'py>,
    item: &FaInferItem,
) -> PyResult<Bound<'py, PyAny>> {
    let payload = PyDict::new(py);
    payload.set_item("words", PyList::new(py, &item.words)?)?;
    payload.set_item("word_ids", PyList::new(py, &item.word_ids)?)?;
    payload.set_item(
        "word_utterance_indices",
        PyList::new(py, &item.word_utterance_indices)?,
    )?;
    payload.set_item(
        "word_utterance_word_indices",
        PyList::new(py, &item.word_utterance_word_indices)?,
    )?;
    payload.set_item("audio_start_ms", item.audio_start_ms)?;
    payload.set_item("audio_end_ms", item.audio_end_ms)?;
    payload.set_item(
        "pauses",
        PyBool::new(py, matches!(item.timing_mode, FaTimingMode::WithPauses)),
    )?;
    Ok(payload.into_any())
}

pub(crate) fn parse_translation_response(
    response: &Bound<'_, PyAny>,
) -> PyResult<TranslateResponse> {
    let dict = response.cast::<PyDict>()?;
    let translation = match dict.get_item("translation")? {
        Some(value) => value.extract::<String>()?,
        None => String::new(),
    };
    Ok(TranslateResponse { translation })
}

pub(crate) fn parse_utseg_response(response: &Bound<'_, PyAny>) -> PyResult<UtsegResponse> {
    let dict = response.cast::<PyDict>()?;
    let assignments = match dict.get_item("assignments")? {
        Some(value) => value.extract::<Vec<usize>>()?,
        None => Vec::new(),
    };
    Ok(UtsegResponse { assignments })
}

pub(crate) fn parse_utseg_batch_response(
    response: &Bound<'_, PyAny>,
) -> PyResult<Vec<UtsegResponse>> {
    let list = response.cast::<PyList>()?;
    let mut parsed = Vec::with_capacity(list.len());
    for item in list.iter() {
        parsed.push(parse_utseg_response(&item.into_any())?);
    }
    Ok(parsed)
}

pub(crate) fn parse_fa_response(response: &Bound<'_, PyAny>) -> PyResult<FaRawResponse> {
    let dict = response.cast::<PyDict>()?;
    if let Some(indexed_timings) = dict.get_item("indexed_timings")? {
        let list = indexed_timings.cast::<PyList>()?;
        let mut parsed = Vec::with_capacity(list.len());
        for item in list.iter() {
            if item.is_none() {
                parsed.push(None);
                continue;
            }
            let timing_dict = item.cast::<PyDict>()?;
            let start_ms = timing_dict
                .get_item("start_ms")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing start_ms"))?
                .extract::<u64>()?;
            let end_ms = timing_dict
                .get_item("end_ms")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing end_ms"))?
                .extract::<u64>()?;
            let confidence = match timing_dict.get_item("confidence")? {
                Some(value) if !value.is_none() => Some(value.extract::<f64>()?),
                _ => None,
            };
            parsed.push(Some(FaIndexedTiming {
                start_ms,
                end_ms,
                confidence,
            }));
        }
        return Ok(FaRawResponse::IndexedWordLevel {
            indexed_timings: parsed,
        });
    }

    if let Some(tokens) = dict.get_item("tokens")? {
        let list = tokens.cast::<PyList>()?;
        let mut parsed = Vec::with_capacity(list.len());
        for item in list.iter() {
            let token_dict = item.cast::<PyDict>()?;
            let text = token_dict
                .get_item("text")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing token text"))?
                .extract::<String>()?;
            let time_s = token_dict
                .get_item("time_s")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing token time_s"))?
                .extract::<f64>()?;
            parsed.push(FaRawToken { text, time_s });
        }
        return Ok(FaRawResponse::TokenLevel { tokens: parsed });
    }

    Err(pyo3::exceptions::PyValueError::new_err(
        "FA callback response must contain 'indexed_timings' or 'tokens'",
    ))
}

pub(crate) fn morphosyntax_payload_to_object<'py>(
    py: Python<'py>,
    words: &[&str],
) -> PyResult<Bound<'py, PyAny>> {
    let payload = PyDict::new(py);
    payload.set_item("words", PyList::new(py, words)?)?;
    Ok(payload.into_any())
}

pub(crate) fn morphosyntax_batch_payload_to_object<'py>(
    py: Python<'py>,
    payloads: &[&MorphosyntaxBatchItem],
) -> PyResult<Bound<'py, PyAny>> {
    let list = PyList::empty(py);
    for item in payloads {
        let obj = PyDict::new(py);
        obj.set_item("words", PyList::new(py, &item.words)?)?;
        obj.set_item("terminator", &item.terminator)?;
        obj.set_item("lang", item.lang.as_str())?;
        list.append(obj)?;
    }
    Ok(list.into_any())
}

/// Convert a generic Python value into a JSON value for Rust-side parsing.
///
/// This helper is intentionally shared across multiple PyO3 entry points so
/// the Rust boundary, not Python, decides which primitive shapes are accepted
/// from callback and operation payloads.
pub(crate) fn py_to_json_value(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Ok(v) = value.extract::<bool>() {
        return Ok(serde_json::Value::Bool(v));
    }
    if let Ok(v) = value.extract::<i64>() {
        return Ok(serde_json::Value::Number(v.into()));
    }
    if let Ok(v) = value.extract::<u64>() {
        return Ok(serde_json::Value::Number(v.into()));
    }
    if let Ok(v) = value.extract::<f64>() {
        return serde_json::Number::from_f64(v)
            .map(serde_json::Value::Number)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("invalid float in callback response")
            });
    }
    if let Ok(v) = value.extract::<String>() {
        return Ok(serde_json::Value::String(v));
    }
    if value.hasattr("model_dump")? {
        let kwargs = PyDict::new(value.py());
        kwargs.set_item("mode", "json")?;
        let dumped = value.call_method("model_dump", (), Some(&kwargs))?;
        return py_to_json_value(dumped.as_any());
    }
    if let Ok(list) = value.cast::<PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_json_value(&item.into_any())?);
        }
        return Ok(serde_json::Value::Array(items));
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_to_json_value(&item.into_any())?);
        }
        return Ok(serde_json::Value::Array(items));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut obj = serde_json::Map::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            let key_str = key.extract::<String>()?;
            obj.insert(key_str, py_to_json_value(&item.into_any())?);
        }
        return Ok(serde_json::Value::Object(obj));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "callback response contains unsupported Python type",
    ))
}

pub(crate) fn parse_morphosyntax_response(
    response: &Bound<'_, PyAny>,
) -> PyResult<crate::nlp::UdResponse> {
    let value = py_to_json_value(response)?;
    if let Some(raw_sentences) = value.get("raw_sentences").and_then(|v| v.as_array()) {
        return batchalign_chat_ops::morphosyntax::stanza_raw::parse_raw_stanza_output(
            raw_sentences,
        )
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()));
    }
    serde_json::from_value(value)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

pub(crate) fn parse_morphosyntax_batch_response(
    response: &Bound<'_, PyAny>,
) -> PyResult<Vec<crate::nlp::UdResponse>> {
    let list = response.cast::<PyList>()?;
    let mut parsed = Vec::with_capacity(list.len());
    for item in list.iter() {
        parsed.push(parse_morphosyntax_response(&item.into_any())?);
    }
    Ok(parsed)
}
