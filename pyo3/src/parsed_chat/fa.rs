//! Forced alignment `#[pymethods]` on `ParsedChat`.

use pyo3::prelude::*;

use crate::ParsedChat;
use crate::fa_ops::add_forced_alignment_inner;

#[pymethods]
impl ParsedChat {
    /// Add word-level timing annotations via a forced-alignment Python callback.
    #[pyo3(name = "add_forced_alignment")]
    #[pyo3(signature = (fa_callback, progress_fn=None, pauses=false, max_group_ms=20000, total_audio_ms=None))]
    #[allow(clippy::too_many_arguments)] // PyO3 boundary — argument count driven by Python API
    fn py_add_forced_alignment(
        &mut self,
        py: Python<'_>,
        fa_callback: &Bound<'_, pyo3::PyAny>,
        progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
        pauses: bool,
        max_group_ms: u64,
        total_audio_ms: Option<u64>,
    ) -> PyResult<()> {
        self.apply_transactional_mutation(|chat_file| {
            add_forced_alignment_inner(
                py,
                chat_file,
                fa_callback,
                progress_fn,
                pauses,
                max_group_ms,
                total_audio_ms,
            )
        })
    }
}
