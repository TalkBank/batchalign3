//! Metadata extraction from CHAT headers.

use pyo3::PyResult;

pub(crate) fn serialize_extracted_words(
    extracted: &[crate::extract::ExtractedUtterance],
) -> String {
    #[derive(serde::Serialize)]
    struct SerializedWord<'a> {
        text: &'a str,
        raw_text: &'a str,
        utterance_word_index: usize,
        word_id: String,
        form_type: &'a Option<talkbank_model::model::FormType>,
        lang_marker: bool,
        provenance: &'a str,
    }

    #[derive(serde::Serialize)]
    struct SerializedUtterance<'a> {
        speaker: &'a str,
        utterance_index: usize,
        words: Vec<SerializedWord<'a>>,
    }

    let utterances: Vec<SerializedUtterance> = extracted
        .iter()
        .map(|utt| SerializedUtterance {
            speaker: utt.speaker.as_str(),
            utterance_index: utt.utterance_index.raw(),
            words: utt
                .words
                .iter()
                .map(|w| SerializedWord {
                    text: w.text.as_str(),
                    raw_text: w.raw_text.as_str(),
                    utterance_word_index: w.utterance_word_index.raw(),
                    word_id: format!("u{}:w{}", utt.utterance_index, w.utterance_word_index),
                    form_type: &w.form_type,
                    lang_marker: w.lang.is_some(),
                    provenance: "chat_original",
                })
                .collect(),
        })
        .collect();

    serde_json::to_string(&utterances).unwrap_or_else(|e| {
        tracing::error!(error = %e, "failed to serialize extracted words");
        "[]".to_string()
    })
}

/// Pure-Rust version of extract_metadata (no PyResult, for use in allow_threads).
pub(crate) fn extract_metadata_from_chat_file_pure(
    chat_file: &talkbank_model::model::ChatFile,
) -> Result<String, String> {
    extract_metadata_from_chat_file(chat_file).map_err(|e| e.to_string())
}

pub(crate) fn extract_metadata_from_chat_file(
    chat_file: &talkbank_model::model::ChatFile,
) -> PyResult<String> {
    use talkbank_model::model::MediaType;

    #[derive(serde::Serialize)]
    struct MetadataJson<'a> {
        langs: Vec<&'a str>,
        media_name: Option<&'a str>,
        media_type: Option<String>,
    }

    let langs: Vec<&str> = chat_file
        .languages
        .iter()
        .map(|lang| lang.as_str())
        .collect();
    let media_name = chat_file
        .media
        .as_ref()
        .map(|media| media.filename.as_str());
    let media_type = chat_file
        .media
        .as_ref()
        .map(|media| match &media.media_type {
            MediaType::Audio => "audio".to_string(),
            MediaType::Video => "video".to_string(),
            MediaType::Missing => "missing".to_string(),
            MediaType::Unsupported(v) => v.clone(),
        });

    serde_json::to_string(&MetadataJson {
        langs,
        media_name,
        media_type,
    })
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("metadata JSON failed: {e}")))
}
