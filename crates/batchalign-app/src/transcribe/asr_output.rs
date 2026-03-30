//! ASR response conversion, participant ID generation, and CHAT helpers.

use batchalign_chat_ops::asr_postprocess::{
    self, AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs,
    SpeakerIndex,
};
use batchalign_chat_ops::build_chat::{self, TranscriptDescription};
use batchalign_chat_ops::serialize::to_chat_string;
use tracing::warn;

use crate::error::ServerError;

use super::types::{AsrResponse, TranscribeOptions};

/// Convert flat ASR tokens (with speaker labels) into speaker-grouped monologues.
///
/// Groups consecutive tokens by speaker. Adjacent tokens with the same speaker
/// are combined into a single monologue. Speaker changes create new monologues.
pub(crate) fn convert_asr_response(response: &AsrResponse) -> AsrOutput {
    if let Some(monologues) = &response.source_monologues {
        return AsrOutput {
            monologues: monologues.clone(),
        };
    }

    if response.tokens.is_empty() {
        return AsrOutput {
            monologues: Vec::new(),
        };
    }

    let mut monologues: Vec<AsrMonologue> = Vec::new();
    let mut current_speaker: Option<SpeakerIndex> = None;
    let mut current_elements: Vec<AsrElement> = Vec::new();

    for token in &response.tokens {
        let speaker_idx = SpeakerIndex(
            token
                .speaker
                .as_deref()
                .and_then(parse_speaker_label)
                .unwrap_or(0),
        );

        if current_speaker != Some(speaker_idx) {
            // Flush previous monologue
            if let Some(spk) = current_speaker
                && !current_elements.is_empty()
            {
                monologues.push(AsrMonologue {
                    speaker: spk,
                    elements: std::mem::take(&mut current_elements),
                });
            }
            current_speaker = Some(speaker_idx);
        }

        current_elements.push(AsrElement {
            value: AsrRawText::new(token.text.clone()),
            ts: AsrTimestampSecs(token.start_s.map(|s| s.0).unwrap_or(0.0)),
            end_ts: AsrTimestampSecs(token.end_s.map(|s| s.0).unwrap_or(0.0)),
            kind: AsrElementKind::Text,
        });
    }

    // Flush last monologue
    if let Some(spk) = current_speaker
        && !current_elements.is_empty()
    {
        monologues.push(AsrMonologue {
            speaker: spk,
            elements: current_elements,
        });
    }

    AsrOutput { monologues }
}

/// Return `true` when the ASR response already carries usable speaker labels.
pub(crate) fn response_has_speaker_labels(response: &AsrResponse) -> bool {
    response.tokens.iter().any(|token| {
        token
            .speaker
            .as_deref()
            .and_then(parse_speaker_label)
            .is_some()
    })
}

pub(super) fn parse_speaker_label(label: &str) -> Option<usize> {
    let trimmed = label.trim();
    trimmed.parse::<usize>().ok().or_else(|| {
        trimmed
            .rsplit('_')
            .next()
            .and_then(|suffix| suffix.parse().ok())
    })
}

/// Generate participant IDs from speaker indices.
///
/// Uses standard CHAT speaker codes: PAR, INV, CHI, etc.
pub(crate) fn generate_participant_ids(
    utterances: &[asr_postprocess::Utterance],
    num_speakers: usize,
) -> Vec<String> {
    let mut max_speaker = 0usize;
    for utt in utterances {
        let s = utt.speaker.as_usize();
        if s > max_speaker {
            max_speaker = s;
        }
    }
    generate_standard_participant_ids((max_speaker + 1).max(num_speakers))
}

pub(crate) fn generate_standard_participant_ids(count: usize) -> Vec<String> {
    const STANDARD_CODES: [&str; 8] = ["PAR", "INV", "CHI", "MOT", "FAT", "SIS", "BRO", "GRM"];

    (0..count)
        .map(|index| {
            if index < STANDARD_CODES.len() {
                STANDARD_CODES[index].to_string()
            } else {
                format!("SP{index}")
            }
        })
        .collect()
}

pub(crate) fn build_empty_chat_text(opts: &TranscribeOptions) -> Result<String, ServerError> {
    warn!(audio_path = %opts.media_name.as_deref().unwrap_or("<unknown>"), "ASR returned no tokens");
    let desc = TranscriptDescription {
        langs: vec![opts.lang.to_string()],
        participants: vec![build_chat::ParticipantDesc {
            id: "PAR".to_string(),
            name: None,
            role: None,
            corpus: None,
        }],
        media_name: opts.media_name.clone(),
        media_type: Some("audio".to_string()),
        utterances: vec![],
        write_wor: opts.write_wor,
    };
    let chat_file = build_chat::build_chat(&desc)
        .map_err(|e| ServerError::Validation(format!("Failed to build empty CHAT: {e}")))?;
    Ok(to_chat_string(&chat_file))
}

pub(crate) fn insert_transcribe_comment(chat_text: &str, opts: &TranscribeOptions) -> String {
    let comment = format!(
        "@Comment:\tBatchalign {}, ASR Engine {}. Unchecked output of ASR model.\n",
        env!("CARGO_PKG_VERSION"),
        opts.backend.comment_engine_name()
    );

    if let Some(pos) = chat_text.find("\n*") {
        let insert_at = pos + 1;
        let mut out = String::with_capacity(chat_text.len() + comment.len());
        out.push_str(&chat_text[..insert_at]);
        out.push_str(&comment);
        out.push_str(&chat_text[insert_at..]);
        return out;
    }

    if let Some(pos) = chat_text.find("\n@End") {
        let insert_at = pos + 1;
        let mut out = String::with_capacity(chat_text.len() + comment.len());
        out.push_str(&chat_text[..insert_at]);
        out.push_str(&comment);
        out.push_str(&chat_text[insert_at..]);
        return out;
    }

    let mut out = chat_text.to_owned();
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(&comment);
    out
}
