//! Build a CHAT file from a JSON transcript description.
//!
//! Delegates to `batchalign_chat_ops::build_chat` for the actual construction.
//! This module provides the PyO3-facing wrapper that accepts `PythonTranscriptJson`.

/// Build a CHAT file from a JSON transcript description.
///
/// Delegates to [`batchalign_chat_ops::build_chat::build_chat_from_json`].
pub(crate) fn build_chat_inner(
    transcript_json: talkbank_model::PythonTranscriptJson,
) -> Result<talkbank_model::model::ChatFile, String> {
    batchalign_chat_ops::build_chat::build_chat_from_json(&transcript_json.data)
}
