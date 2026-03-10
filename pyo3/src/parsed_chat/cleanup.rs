//! Disfluency and retrace marker `#[pymethods]` on `ParsedChat`.

use pyo3::prelude::*;

use crate::ParsedChat;
use crate::cleanup_ops::{add_disfluency_markers_inner, add_retrace_markers_inner};

#[pymethods]
impl ParsedChat {
    /// Mark filled pauses and apply word replacements across all utterances.
    #[pyo3(name = "add_disfluency_markers")]
    fn py_add_disfluency_markers(
        &mut self,
        py: Python<'_>,
        filled_pauses_json: talkbank_model::model::Provenance<
            talkbank_model::model::AsrWordsJson,
            String,
        >,
        replacements_json: talkbank_model::model::Provenance<
            talkbank_model::model::AsrWordsJson,
            String,
        >,
    ) -> PyResult<()> {
        let fp_json = filled_pauses_json.data;
        let repl_json = replacements_json.data;
        let inner = &mut self.inner;
        py.detach(|| {
            add_disfluency_markers_inner(inner, &fp_json, &repl_json).map_err(|e| e.to_string())
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Add n-gram retrace markers.
    #[pyo3(name = "add_retrace_markers")]
    fn py_add_retrace_markers(&mut self, py: Python<'_>, lang: talkbank_model::PythonLanguageId) {
        let lang_str = lang.data;
        let inner = &mut self.inner;
        py.detach(|| add_retrace_markers_inner(inner, &lang_str));
    }
}
