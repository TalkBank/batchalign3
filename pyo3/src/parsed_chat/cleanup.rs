//! Disfluency and retrace marker `#[pymethods]` on `ParsedChat`.

use pyo3::prelude::*;

use crate::ParsedChat;
use crate::cleanup_ops::{add_disfluency_markers_inner, add_retrace_markers_inner};
use crate::pytypes::{PythonAsrWordsJson, PythonLanguageId};

#[pymethods]
impl ParsedChat {
    /// Mark filled pauses and apply word replacements across all utterances.
    #[pyo3(name = "add_disfluency_markers")]
    fn py_add_disfluency_markers(
        &mut self,
        py: Python<'_>,
        filled_pauses_json: PythonAsrWordsJson,
        replacements_json: PythonAsrWordsJson,
    ) -> PyResult<()> {
        let fp_json = filled_pauses_json.into_data();
        let repl_json = replacements_json.into_data();
        let inner = &mut self.inner;
        py.detach(|| {
            add_disfluency_markers_inner(inner, &fp_json, &repl_json).map_err(|e| e.to_string())
        })
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Add n-gram retrace markers.
    #[pyo3(name = "add_retrace_markers")]
    fn py_add_retrace_markers(&mut self, py: Python<'_>, lang: PythonLanguageId) {
        let lang_str = lang.into_data();
        let inner = &mut self.inner;
        py.detach(|| add_retrace_markers_inner(inner, &lang_str));
    }
}
