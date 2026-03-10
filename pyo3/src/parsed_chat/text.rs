//! Translation and utterance segmentation `#[pymethods]` on `ParsedChat`.

use pyo3::prelude::*;

use crate::ParsedChat;
use crate::text_ops::{
    add_translation_inner, add_utterance_segmentation_batched_inner,
    add_utterance_segmentation_inner,
};

#[pymethods]
impl ParsedChat {
    /// Add English translation as a `%xtra` dependent tier via a Python callback.
    #[pyo3(name = "add_translation")]
    #[pyo3(signature = (translation_fn, progress_fn=None))]
    fn py_add_translation(
        &mut self,
        py: Python<'_>,
        translation_fn: &Bound<'_, pyo3::PyAny>,
        progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        self.apply_transactional_mutation(|chat_file| {
            add_translation_inner(py, chat_file, translation_fn, progress_fn)
        })
    }

    /// Split utterances into sub-utterances via a per-utterance Python callback.
    #[pyo3(name = "add_utterance_segmentation")]
    #[pyo3(signature = (segmentation_fn, progress_fn=None))]
    fn py_add_utterance_segmentation(
        &mut self,
        py: Python<'_>,
        segmentation_fn: &Bound<'_, pyo3::PyAny>,
        progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        self.apply_transactional_mutation(|chat_file| {
            add_utterance_segmentation_inner(py, chat_file, segmentation_fn, progress_fn)
        })
    }

    /// Split utterances into sub-utterances via a single batched Python callback.
    #[pyo3(name = "add_utterance_segmentation_batched")]
    #[pyo3(signature = (batch_fn, progress_fn=None))]
    fn py_add_utterance_segmentation_batched(
        &mut self,
        py: Python<'_>,
        batch_fn: &Bound<'_, pyo3::PyAny>,
        progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
    ) -> PyResult<()> {
        self.apply_transactional_mutation(|chat_file| {
            add_utterance_segmentation_batched_inner(py, chat_file, batch_fn, progress_fn)
        })
    }
}
