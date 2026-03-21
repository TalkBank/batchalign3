//! Speaker operations `#[pymethods]` on `ParsedChat`.

use pyo3::prelude::*;

use crate::ParsedChat;
use crate::pytypes::{PythonAsrWordsJson, PythonLanguageId};
use crate::speaker_ops::{add_utterance_timing_inner, reassign_speakers_inner};

#[pymethods]
impl ParsedChat {
    /// Reassign speaker codes based on diarization segments.
    #[pyo3(name = "reassign_speakers")]
    fn py_reassign_speakers(
        &mut self,
        py: Python<'_>,
        segments_json: PythonAsrWordsJson,
        lang: PythonLanguageId,
    ) -> PyResult<()> {
        let seg_json = segments_json.into_data();
        let lang_str = lang.into_data();
        let old_file = std::mem::replace(
            &mut self.inner,
            talkbank_model::model::ChatFile::new(Vec::new()),
        );
        let result = py.detach(|| {
            reassign_speakers_inner(old_file, &seg_json, &lang_str).map_err(|e| e.to_string())
        });
        match result {
            Ok(new_file) => {
                self.inner = new_file;
                Ok(())
            }
            Err(msg) => Err(pyo3::exceptions::PyValueError::new_err(msg)),
        }
    }

    /// Add word-level timing from ASR words.
    ///
    /// If ASR words include stable transcript IDs (`word_id` = `u{n}:w{n}`),
    /// timing is mapped directly by ID. Otherwise, falls back to deterministic
    /// monotonic matching: first constrained to uniquely overlapping utterance
    /// windows when bullets exist; global monotonic fallback is used only when
    /// utterance windows are absent.
    #[pyo3(name = "add_utterance_timing")]
    fn py_add_utterance_timing(
        &mut self,
        py: Python<'_>,
        asr_words_json: PythonAsrWordsJson,
    ) -> PyResult<()> {
        let asr_json = asr_words_json.into_data();
        let inner = &mut self.inner;
        py.detach(|| add_utterance_timing_inner(inner, &asr_json).map_err(|e| e.to_string()))
            .map_err(pyo3::exceptions::PyValueError::new_err)
    }
}
