//! Morphosyntax-related `#[pymethods]` on `ParsedChat`.

use pyo3::prelude::*;

use batchalign_chat_ops::morphosyntax::{
    MultilingualPolicy, extract_payloads_json as extract_morphosyntax_payloads_json,
    inject_from_cache as inject_morphosyntax_from_cache_inner,
};

use crate::ParsedChat;
use crate::morphosyntax_ops::{
    add_morphosyntax_batched_inner, add_morphosyntax_inner, extract_morphosyntax_strings_inner,
};
use crate::pytypes::PythonLanguageId;

#[pymethods]
impl ParsedChat {
    /// Add morphosyntax (%mor/%gra) annotations via a per-utterance Python callback.
    #[pyo3(name = "add_morphosyntax")]
    #[pyo3(signature = (lang, morphosyntax_fn, progress_fn=None, skipmultilang=false, retokenize=false))]
    #[allow(clippy::too_many_arguments)] // PyO3 boundary — argument count driven by Python API
    fn py_add_morphosyntax(
        &mut self,
        py: Python<'_>,
        lang: PythonLanguageId,
        morphosyntax_fn: &Bound<'_, pyo3::PyAny>,
        progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
        skipmultilang: bool,
        retokenize: bool,
    ) -> PyResult<()> {
        let lang_str = lang.into_data();
        let policy = MultilingualPolicy::from_skip_flag(skipmultilang);
        self.apply_transactional_mutation(|chat_file| {
            add_morphosyntax_inner(
                py,
                chat_file,
                &lang_str,
                morphosyntax_fn,
                progress_fn,
                policy,
                retokenize,
            )
        })
    }

    /// Add morphosyntax (%mor/%gra) annotations via a single batched Python callback.
    #[pyo3(name = "add_morphosyntax_batched")]
    #[pyo3(signature = (lang, batch_fn, progress_fn=None, skipmultilang=false, retokenize=false))]
    #[allow(clippy::too_many_arguments)] // PyO3 boundary — argument count driven by Python API
    fn py_add_morphosyntax_batched(
        &mut self,
        py: Python<'_>,
        lang: PythonLanguageId,
        batch_fn: &Bound<'_, pyo3::PyAny>,
        progress_fn: Option<&Bound<'_, pyo3::PyAny>>,
        skipmultilang: bool,
        retokenize: bool,
    ) -> PyResult<()> {
        let lang_str = lang.into_data();
        let policy = MultilingualPolicy::from_skip_flag(skipmultilang);
        self.apply_transactional_mutation(|chat_file| {
            add_morphosyntax_batched_inner(
                py,
                chat_file,
                &lang_str,
                batch_fn,
                progress_fn,
                policy,
                retokenize,
            )
        })
    }

    /// Extract per-utterance word payloads for morphosyntax cache key computation.
    ///
    /// Returns JSON: `[{"line_idx": 0, "words": ["I", "eat"], "lang": "eng"}, ...]`
    #[pyo3(name = "extract_morphosyntax_payloads")]
    #[pyo3(signature = (lang, skipmultilang=false))]
    fn py_extract_morphosyntax_payloads(
        &self,
        py: Python<'_>,
        lang: PythonLanguageId,
        skipmultilang: bool,
    ) -> PyResult<String> {
        let lang_str = lang.into_data();
        let inner = &self.inner;
        let policy = MultilingualPolicy::from_skip_flag(skipmultilang);
        let empty_mwt = std::collections::BTreeMap::new();
        let result =
            py.detach(|| extract_morphosyntax_payloads_json(inner, &lang_str, policy, &empty_mwt));
        result.map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }

    /// Inject cached %mor/%gra strings into specific utterances.
    ///
    /// Input JSON: `[{"line_idx": 0, "mor": "pro:sub|I v|eat .", "gra": "1|2|SUBJ 2|0|ROOT 3|2|PUNCT"}]`
    #[pyo3(name = "inject_morphosyntax_from_cache")]
    fn py_inject_morphosyntax_from_cache(
        &mut self,
        py: Python<'_>,
        data_json: &str,
    ) -> PyResult<()> {
        let data_str = data_json.to_string();
        self.apply_transactional_mutation(|chat_file| {
            py.detach(|| inject_morphosyntax_from_cache_inner(chat_file, &data_str))
                .map_err(pyo3::exceptions::PyRuntimeError::new_err)
        })
    }

    /// Extract final %mor/%gra strings for specified utterances (by line index).
    ///
    /// Input JSON: `[0, 2, 5]` (line indices)
    /// Returns JSON: `[{"line_idx": 0, "mor": "pro:sub|I v|eat .", "gra": "1|2|SUBJ 2|0|ROOT 3|2|PUNCT"}]`
    #[pyo3(name = "extract_morphosyntax_strings")]
    fn py_extract_morphosyntax_strings(
        &self,
        py: Python<'_>,
        line_indices_json: &str,
    ) -> PyResult<String> {
        let indices_str = line_indices_json.to_string();
        let inner = &self.inner;
        let result = py.detach(|| extract_morphosyntax_strings_inner(inner, &indices_str));
        result.map_err(pyo3::exceptions::PyRuntimeError::new_err)
    }
}
