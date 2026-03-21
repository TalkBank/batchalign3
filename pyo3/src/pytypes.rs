//! PyO3-local wrapper types for Python-owned provenance-tagged inputs.
//!
//! These wrappers keep Python extraction logic at the binding boundary instead
//! of inside `talkbank-model`.

use pyo3::prelude::*;

use talkbank_model::model::{
    AsrWordsJson, LanguageId, Provenance, RawChatText, TierDomainMarker, TranscriptJson,
};

pub(crate) struct PyProvenance<M, T = String>(pub(crate) Provenance<M, T>);

impl<M, T> PyProvenance<M, T> {
    pub(crate) fn into_data(self) -> T {
        self.0.data
    }
}

impl<'a, 'py, M, T> FromPyObject<'a, 'py> for PyProvenance<M, T>
where
    T: FromPyObject<'a, 'py>,
{
    type Error = T::Error;

    fn extract(ob: pyo3::Borrowed<'a, 'py, pyo3::PyAny>) -> Result<Self, Self::Error> {
        let data = T::extract(ob)?;
        Ok(Self(Provenance::new(data)))
    }
}

pub(crate) type PythonChatText = PyProvenance<RawChatText, String>;
pub(crate) type PythonTranscriptJson = PyProvenance<TranscriptJson, String>;
pub(crate) type PythonAsrWordsJson = PyProvenance<AsrWordsJson, String>;
pub(crate) type PythonLanguageId = PyProvenance<LanguageId, String>;
pub(crate) type PythonTierDomain = PyProvenance<TierDomainMarker, String>;
