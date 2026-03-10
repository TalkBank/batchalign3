//! Rust-owned CHAT pipeline execution for the Python API path.
//!
//! The architectural goal of this module is simple:
//!
//! - Python owns model loading and raw model invocation.
//! - Rust owns CHAT parsing, payload extraction, document mutation,
//!   post-processing, and operation sequencing.
//!
//! The legacy Python-side `pipeline_api_*` helpers grew into a second
//! document-processing layer outside Rust. This module collapses that extra
//! layer by exposing one PyO3 entry point which:
//!
//! 1. parses the CHAT document in Rust,
//! 2. executes a sequence of typed document operations in Rust,
//! 3. calls back into Python only for raw provider batches,
//! 4. serializes the final CHAT document in Rust.
//!
//! Incremental and cache-aware processing are intentionally out of scope for
//! this module. Those flows already live in the Rust server/runtime layer, so
//! they do not need to be preserved in the Python API surface.

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PyModule, PyTuple};
use talkbank_model::WriteChat;

use crate::fa_ops::add_forced_alignment_inner;
use crate::morphosyntax_ops::add_morphosyntax_batched_inner;
use crate::parse::{parse_lenient_pure, parse_strict_pure};
use crate::py_json_bridge::py_to_json_value;
use crate::speaker_ops::add_utterance_timing_inner;
use crate::text_ops::{add_translation_inner, add_utterance_segmentation_batched_inner};

/// One Rust-owned document operation requested by the Python pipeline facade.
///
/// Each variant stores only the operation-local options that still matter once
/// the Python orchestration layer is removed.
enum ProviderPipelineOperation {
    /// Inject translation tiers by batching translation payloads through a
    /// Python provider.
    Translate {
        /// Optional progress callback forwarded back to Python.
        progress_fn: Option<Py<PyAny>>,
    },
    /// Inject `%mor/%gra` annotations using batched provider output.
    Morphosyntax {
        /// Optional progress callback forwarded back to Python.
        progress_fn: Option<Py<PyAny>>,
        /// Whether to skip utterances marked as non-primary-language content.
        skipmultilang: bool,
        /// Whether Stanza retokenization should be mapped back into CHAT.
        retokenize: bool,
    },
    /// Inject forced-alignment timing and generate `%wor`.
    ForcedAlignment {
        /// Optional progress callback forwarded back to Python.
        progress_fn: Option<Py<PyAny>>,
        /// Whether timing should preserve explicit pause structure.
        pauses: bool,
        /// Maximum audio duration per grouped FA window.
        max_group_ms: u64,
        /// Optional total media duration used by FA grouping.
        total_audio_ms: Option<u64>,
    },
    /// Split utterances by batched segmentation assignments.
    UtteranceSegmentation {
        /// Optional progress callback forwarded back to Python.
        progress_fn: Option<Py<PyAny>>,
    },
    /// Inject already-produced timed ASR words.
    UtteranceTiming {
        /// JSON-serialized timed-word payload consumed by the existing Rust
        /// utterance-timing injector.
        timed_words_json: String,
    },
}

#[derive(serde::Deserialize)]
struct BatchInferResultEnvelope {
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(serde::Deserialize)]
struct BatchInferResponseEnvelope {
    results: Vec<BatchInferResultEnvelope>,
}

/// Run a sequence of provider-backed CHAT operations with Rust owning the
/// orchestration loop.
///
/// `provider_batch_fn` must have the signature
/// `(task: str, lang: str, items: list[dict]) -> list[dict | None]`.
///
/// The function only calls back into Python for raw provider batches. Parsing,
/// payload extraction, result injection, and operation ordering remain in
/// Rust. This is the preferred Python API entry point for CHAT-aware work.
#[pyfunction]
#[pyo3(signature = (chat_text, *, lang, provider_batch_fn, operations, lenient=false))]
pub(crate) fn run_provider_pipeline(
    py: Python<'_>,
    chat_text: &str,
    lang: &str,
    provider_batch_fn: &Bound<'_, PyAny>,
    operations: &Bound<'_, PyAny>,
    lenient: bool,
) -> PyResult<String> {
    let mut chat_file = if lenient {
        parse_lenient_pure(chat_text).map_err(pyo3::exceptions::PyValueError::new_err)?
    } else {
        parse_strict_pure(chat_text).map_err(pyo3::exceptions::PyValueError::new_err)?
    };

    let operations = parse_operations(operations)?;
    let provider_batch_fn = provider_batch_fn.clone().unbind();

    for operation in operations {
        run_operation(
            py,
            &mut chat_file,
            lang,
            provider_batch_fn.clone_ref(py),
            operation,
        )?;
    }

    Ok(chat_file.to_chat_string())
}

fn normalize_batch_infer_results(
    task: &str,
    response: &Bound<'_, PyAny>,
) -> PyResult<Vec<serde_json::Value>> {
    let response: BatchInferResponseEnvelope = serde_json::from_value(py_to_json_value(response)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;

    let mut results = Vec::with_capacity(response.results.len());
    let mut errors = Vec::new();

    for (idx, item) in response.results.into_iter().enumerate() {
        if let Some(error) = item.error.filter(|value| !value.is_empty()) {
            errors.push(format!("{idx}: {error}"));
            results.push(serde_json::Value::Null);
            continue;
        }

        match item.result {
            None => results.push(serde_json::Value::Null),
            Some(serde_json::Value::Object(value)) => {
                results.push(serde_json::Value::Object(value))
            }
            Some(_) => {
                return Err(PyTypeError::new_err(format!(
                    "{task} provider returned a non-object result at index {idx}"
                )));
            }
        }
    }

    if !errors.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "{task} provider batch failed for {}",
            errors.join(", ")
        )));
    }

    Ok(results)
}

/// Normalize one Python `BatchInferResponse` into the raw provider-batch shape.
///
/// This keeps batch-infer result unwrapping and validation on the Rust side so
/// `pipeline_api.py` stays a thin adapter instead of regrowing response-control
/// logic in Python.
#[pyfunction]
#[pyo3(signature = (task, response))]
pub(crate) fn unwrap_batch_infer_results(
    task: &str,
    response: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let results = normalize_batch_infer_results(task, response)?;
    serde_json::to_string(&results).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Build a `BatchInferRequest`, call the Python infer host, and normalize the response.
#[pyfunction]
#[pyo3(signature = (task, lang, items, infer_fn))]
pub(crate) fn call_batch_infer_provider(
    py: Python<'_>,
    task: &str,
    lang: &str,
    items: &Bound<'_, PyAny>,
    infer_fn: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let providers = PyModule::import(py, "batchalign.providers")?;
    let infer_task_cls = providers.getattr("InferTask")?;
    let batch_infer_request_cls = providers.getattr("BatchInferRequest")?;
    let task_enum = infer_task_cls.call1((task,))?;

    let kwargs = PyDict::new(py);
    kwargs.set_item("task", task_enum)?;
    kwargs.set_item("lang", lang)?;
    kwargs.set_item("items", items)?;

    let request = batch_infer_request_cls.call((), Some(&kwargs))?;
    let response = infer_fn.call1((request,))?;
    let results = normalize_batch_infer_results(task, &response)?;
    serde_json::to_string(&results).map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Parse the Python `operations` list into a Rust enum sequence.
fn parse_operations(operations: &Bound<'_, PyAny>) -> PyResult<Vec<ProviderPipelineOperation>> {
    let operations = operations.cast::<PyList>()?;
    let mut parsed = Vec::with_capacity(operations.len());
    for operation in operations.iter() {
        parsed.push(parse_operation(&operation.into_any())?);
    }
    Ok(parsed)
}

/// Parse one operation dictionary from the Python facade.
fn parse_operation(operation: &Bound<'_, PyAny>) -> PyResult<ProviderPipelineOperation> {
    let operation = operation.cast::<PyDict>()?;
    let name = required_string_field(operation, "name")?;

    match name.as_str() {
        "translate" => Ok(ProviderPipelineOperation::Translate {
            progress_fn: optional_python_callable(operation, "progress_fn")?,
        }),
        "morphosyntax" => Ok(ProviderPipelineOperation::Morphosyntax {
            progress_fn: optional_python_callable(operation, "progress_fn")?,
            skipmultilang: bool_field(operation, "skipmultilang", false)?,
            retokenize: bool_field(operation, "retokenize", false)?,
        }),
        "fa" => Ok(ProviderPipelineOperation::ForcedAlignment {
            progress_fn: optional_python_callable(operation, "progress_fn")?,
            pauses: bool_field(operation, "pauses", false)?,
            max_group_ms: u64_field(operation, "max_group_ms", 20_000)?,
            total_audio_ms: optional_u64_field(operation, "total_audio_ms")?,
        }),
        "utseg" => Ok(ProviderPipelineOperation::UtteranceSegmentation {
            progress_fn: optional_python_callable(operation, "progress_fn")?,
        }),
        "utr" => Ok(ProviderPipelineOperation::UtteranceTiming {
            timed_words_json: required_json_array_field(operation, "timed_words")?,
        }),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported pipeline operation: {other}"
        ))),
    }
}

/// Execute one parsed operation over the mutable CHAT document.
fn run_operation(
    py: Python<'_>,
    chat_file: &mut talkbank_model::model::ChatFile,
    lang: &str,
    provider_batch_fn: Py<PyAny>,
    operation: ProviderPipelineOperation,
) -> PyResult<()> {
    match operation {
        ProviderPipelineOperation::Translate { progress_fn } => {
            let callback =
                make_single_item_provider_callback(py, provider_batch_fn, "translate", lang)?;
            add_translation_inner(
                py,
                chat_file,
                callback.as_any(),
                progress_fn.as_ref().map(|callback| callback.bind(py)),
            )
        }
        ProviderPipelineOperation::Morphosyntax {
            progress_fn,
            skipmultilang,
            retokenize,
        } => {
            let callback =
                make_batch_provider_callback(py, provider_batch_fn, "morphosyntax", lang)?;
            let policy = batchalign_chat_ops::morphosyntax::MultilingualPolicy::from_skip_flag(
                skipmultilang,
            );
            add_morphosyntax_batched_inner(
                py,
                chat_file,
                lang,
                callback.as_any(),
                progress_fn.as_ref().map(|callback| callback.bind(py)),
                policy,
                retokenize,
            )
        }
        ProviderPipelineOperation::ForcedAlignment {
            progress_fn,
            pauses,
            max_group_ms,
            total_audio_ms,
        } => {
            let callback = make_single_item_provider_callback(py, provider_batch_fn, "fa", lang)?;
            add_forced_alignment_inner(
                py,
                chat_file,
                callback.as_any(),
                progress_fn.as_ref().map(|callback| callback.bind(py)),
                pauses,
                max_group_ms,
                total_audio_ms,
            )
        }
        ProviderPipelineOperation::UtteranceSegmentation { progress_fn } => {
            let callback = make_batch_provider_callback(py, provider_batch_fn, "utseg", lang)?;
            add_utterance_segmentation_batched_inner(
                py,
                chat_file,
                callback.as_any(),
                progress_fn.as_ref().map(|callback| callback.bind(py)),
            )
        }
        ProviderPipelineOperation::UtteranceTiming { timed_words_json } => {
            add_utterance_timing_inner(chat_file, &timed_words_json)
        }
    }
}

/// Build a per-item callback wrapper around the generic Python batch provider.
///
/// Existing Rust document operations such as translation and forced alignment
/// already know how to:
///
/// - extract one payload at a time,
/// - call a Python callback,
/// - parse the callback response,
/// - inject the result back into the AST.
///
/// To avoid reimplementing that logic, we synthesize a callback which adapts
/// the generic batch-provider function to the older single-item callback shape.
fn make_single_item_provider_callback<'py>(
    py: Python<'py>,
    provider_batch_fn: Py<PyAny>,
    task: &'static str,
    lang: &str,
) -> PyResult<Bound<'py, PyCFunction>> {
    let lang = lang.to_string();
    PyCFunction::new_closure(
        py,
        Some(c"batchalign_provider_single"),
        Some(c"Adapt the generic provider batch callback to a single-item callback."),
        move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<Py<PyAny>> {
            let payload = args.get_item(0).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "provider callback expected one payload argument",
                )
            })?;
            let py = args.py();
            let items = PyList::empty(py);
            items.append(payload)?;
            let provider = provider_batch_fn.bind(py);
            let result = provider.call1((task, lang.as_str(), items))?;
            let responses = result.cast::<PyList>()?;
            if responses.len() != 1 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "{task} provider returned {} responses for one payload",
                    responses.len()
                )));
            }
            Ok(responses.get_item(0)?.unbind())
        },
    )
}

/// Build a batched callback wrapper around the generic Python batch provider.
///
/// Morphosyntax and utterance segmentation already batch payload extraction in
/// Rust. This wrapper preserves that efficient path while removing the Python
/// extension orchestration layer.
fn make_batch_provider_callback<'py>(
    py: Python<'py>,
    provider_batch_fn: Py<PyAny>,
    task: &'static str,
    lang: &str,
) -> PyResult<Bound<'py, PyCFunction>> {
    let lang = lang.to_string();
    PyCFunction::new_closure(
        py,
        Some(c"batchalign_provider_batch"),
        Some(
            c"Adapt the generic provider batch callback to the ParsedChat batched callback shape.",
        ),
        move |args: &Bound<'_, PyTuple>, _kwargs| -> PyResult<Py<PyAny>> {
            let payloads = args.get_item(0).map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(
                    "provider callback expected a batched payload argument",
                )
            })?;
            let py = args.py();
            let provider = provider_batch_fn.bind(py);
            let responses = provider.call1((task, lang.as_str(), payloads))?;
            Ok(responses.unbind())
        },
    )
}

/// Return one required string field from an operation dictionary.
fn required_string_field(operation: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    operation
        .get_item(key)?
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "pipeline operation is missing required field {key:?}"
            ))
        })?
        .extract::<String>()
}

/// Return one optional Python callable from an operation dictionary.
fn optional_python_callable(
    operation: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<Py<PyAny>>> {
    let Some(value) = operation.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    if !value.is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "pipeline operation field {key:?} must be callable or None"
        )));
    }
    Ok(Some(value.unbind()))
}

/// Return one optional boolean field from an operation dictionary.
fn bool_field(operation: &Bound<'_, PyDict>, key: &str, default: bool) -> PyResult<bool> {
    match operation.get_item(key)? {
        Some(value) => value.extract::<bool>(),
        None => Ok(default),
    }
}

/// Return one optional unsigned integer field from an operation dictionary.
fn u64_field(operation: &Bound<'_, PyDict>, key: &str, default: u64) -> PyResult<u64> {
    match operation.get_item(key)? {
        Some(value) => value.extract::<u64>(),
        None => Ok(default),
    }
}

/// Return one optional unsigned integer field which may also be `None`.
fn optional_u64_field(operation: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u64>> {
    let Some(value) = operation.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    value.extract::<u64>().map(Some)
}

/// Return one required JSON array field serialized as a string.
///
/// `add_utterance_timing_inner` already expects its payload as JSON text, so
/// this helper preserves that narrow Rust boundary instead of inventing a
/// second representation just for the pipeline executor.
fn required_json_array_field(operation: &Bound<'_, PyDict>, key: &str) -> PyResult<String> {
    let value = operation.get_item(key)?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "pipeline operation is missing required field {key:?}"
        ))
    })?;
    let value = py_to_json_value(&value)?;
    if !matches!(value, serde_json::Value::Array(_)) {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "pipeline operation field {key:?} must be a list"
        )));
    }
    serde_json::to_string(&value).map_err(|error| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "failed to serialize pipeline operation field {key:?}: {error}"
        ))
    })
}
