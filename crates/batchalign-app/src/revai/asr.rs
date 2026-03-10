//! Rust-owned Rev.AI ASR inference for server-mode transcription.
//!
//! This path exists so `transcribe` and `benchmark` do not need to route the
//! Rev.AI provider through the Python worker at all. The only engines that
//! should stay in Python are the ones that genuinely require Python-hosted
//! model libraries.

use std::path::Path;

use batchalign_revai::{RevAiClient, SubmitOptions, Transcript};

use crate::api::{LanguageCode3, NumSpeakers};
use crate::error::ServerError;
use crate::transcribe::{AsrResponse, AsrToken};

use super::{RevAiLanguageHint, load_revai_api_key};

/// Run Rev.AI ASR directly from Rust and map the transcript into the shared
/// `AsrResponse` domain used by the transcribe pipeline.
pub(crate) async fn infer_revai_asr(
    audio_path: &Path,
    lang: &LanguageCode3,
    num_speakers: NumSpeakers,
    rev_job_id: Option<&str>,
) -> Result<AsrResponse, ServerError> {
    let api_key =
        load_revai_api_key().map_err(|error| ServerError::Validation(error.to_string()))?;
    let audio_path = audio_path.to_path_buf();
    let lang = lang.clone();
    let rev_job_id = rev_job_id.map(str::to_string);

    tokio::task::spawn_blocking(move || {
        let transcript = fetch_revai_transcript(
            &api_key,
            &audio_path,
            &lang,
            num_speakers,
            rev_job_id.as_deref(),
        )
        .map_err(|error| ServerError::Validation(error.to_string()))?;
        Ok(transcript_to_asr_response(&transcript, &lang))
    })
    .await
    .map_err(|error| ServerError::Validation(format!("Rev.AI task join error: {error}")))?
}

/// Fetch a Rev.AI transcript either by polling an existing submitted job or by
/// submitting one local audio file and waiting for completion.
pub(super) fn fetch_revai_transcript(
    api_key: &super::RevAiApiKey,
    audio_path: &Path,
    lang: &LanguageCode3,
    num_speakers: NumSpeakers,
    rev_job_id: Option<&str>,
) -> batchalign_revai::Result<Transcript> {
    let client = RevAiClient::new(api_key.as_str());
    if let Some(job_id) = rev_job_id {
        return client.poll_and_download(job_id, 5, 30);
    }

    let lang_hint = RevAiLanguageHint::from(lang);
    let speakers_count = match lang_hint.as_str() {
        "en" | "es" => Some(num_speakers.0),
        _ => None,
    };
    let metadata = audio_path
        .file_stem()
        .map(|stem| format!("batchalign3_{}", stem.to_string_lossy()));
    let options = SubmitOptions {
        language: lang_hint.as_str().to_string(),
        speakers_count,
        skip_postprocessing: Some(false),
        metadata,
    };
    client.transcribe_blocking(audio_path, &options, 30)
}

fn transcript_to_asr_response(transcript: &Transcript, lang: &LanguageCode3) -> AsrResponse {
    let mut tokens = Vec::new();

    for monologue in &transcript.monologues {
        let speaker = monologue.speaker.to_string();
        for element in &monologue.elements {
            if element.element_type != "text" {
                continue;
            }

            let text = element.value.trim();
            if text.is_empty() {
                continue;
            }

            tokens.push(AsrToken {
                text: text.to_string(),
                start_s: element.ts,
                end_s: element.end_ts,
                speaker: Some(speaker.clone()),
                confidence: element.confidence,
            });
        }
    }

    AsrResponse {
        tokens,
        lang: lang.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::transcript_to_asr_response;
    use crate::api::LanguageCode3;
    use batchalign_revai::Transcript;

    #[test]
    fn transcript_projection_keeps_only_text_elements() {
        let transcript: Transcript = serde_json::from_str(
            r#"{
            "monologues": [{
                "speaker": 3,
                "elements": [
                    {"type": "text", "value": "hello", "ts": 0.5, "end_ts": 0.9, "confidence": 0.75},
                    {"type": "punct", "value": ","},
                    {"type": "text", "value": "world", "ts": 1.0, "end_ts": 1.4}
                ]
            }]
        }"#,
        )
        .unwrap();

        let response = transcript_to_asr_response(&transcript, &LanguageCode3::from("eng"));
        assert_eq!(response.lang, "eng");
        assert_eq!(response.tokens.len(), 2);
        assert_eq!(response.tokens[0].text, "hello");
        assert_eq!(response.tokens[0].speaker.as_deref(), Some("3"));
        assert_eq!(response.tokens[0].confidence, Some(0.75));
        assert_eq!(response.tokens[1].text, "world");
    }
}
