//! Server-side transcribe orchestrator.
//!
//! Owns the full audio-to-CHAT lifecycle:
//! raw ASR inference → Rust normalization → post-processing → CHAT assembly
//! → optional utseg → optional morphosyntax.
//!
//! Raw ASR inference comes from one of two typed backends:
//! - the Rust-owned Rev.AI client for `asr_engine=rev`
//! - the Python worker V2 `execute_v2(task="asr")` path for every other
//!   Python-hosted ASR engine
//!
//! All post-processing (compound merging, number expansion, retokenization) and
//! CHAT construction happen in Rust.
//!
//! # Call path
//!
//! `runner::dispatch_transcribe_infer()`
//! → [`process_transcribe`] (per audio file)
//! → selected ASR backend → raw provider payload
//! → Rust normalization into shared timing/token records
//! → `batchalign_chat_ops::asr_postprocess::process_raw_asr()` → utterances
//! → `batchalign_chat_ops::build_chat::build_chat()` → ChatFile
//! → optional `crate::utseg::process_utseg()` → re-segmented CHAT
//! → optional `crate::morphosyntax::process_morphosyntax()` → morphotagged CHAT

use std::path::Path;

use crate::api::{LanguageCode3, NumSpeakers};
use crate::revai::infer_revai_asr;
use crate::types::worker_v2::{AsrBackendV2, SpeakerBackendV2, SpeakerSegmentV2};
use crate::worker::artifacts_v2::PreparedArtifactRuntimeV2;
use crate::worker::asr_request_v2::{
    AsrBuildInputV2, AsrInputSourceV2, PreparedAsrRequestIdsV2, build_asr_request_v2,
};
use crate::worker::asr_result_v2::parse_asr_response_v2;
use crate::worker::pool::WorkerPool;
use crate::worker::speaker_request_v2::{
    PreparedSpeakerRequestIdsV2, SpeakerBuildInputV2, build_speaker_request_v2,
};
use crate::worker::speaker_result_v2::parse_speaker_result_v2;
use batchalign_chat_ops::asr_postprocess::{self, AsrElement, AsrMonologue, AsrOutput};
use batchalign_chat_ops::build_chat::{self, TranscriptDescription};
use batchalign_chat_ops::serialize::to_chat_string;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::pipeline::transcribe::run_transcribe_pipeline;
use crate::runner::util::ProgressSender;

// ---------------------------------------------------------------------------
// ASR response types (match Python inference/asr.py models)
// ---------------------------------------------------------------------------

/// A single raw ASR output token from the selected ASR backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrToken {
    /// Word text.
    pub text: String,
    /// Start time in seconds.
    pub start_s: Option<f64>,
    /// End time in seconds.
    pub end_s: Option<f64>,
    /// Speaker label (e.g. "0", "1") from diarization.
    pub speaker: Option<String>,
    /// Confidence score (0.0–1.0).
    pub confidence: Option<f64>,
}

/// Shared ASR inference response consumed by the Rust transcribe pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResponse {
    /// Raw tokens with timestamps and speaker labels.
    pub tokens: Vec<AsrToken>,
    /// Language code.
    #[serde(default = "default_lang")]
    pub lang: String,
}

fn default_lang() -> String {
    "eng".to_string()
}

/// Which runtime boundary owns raw ASR inference for one command execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AsrBackend {
    /// Use the Rust-owned Rev.AI client directly from the server.
    RustRevAi,
    /// Use a Python worker path selected by a typed worker-mode value.
    Worker(AsrWorkerMode),
}

/// Concrete Python-worker ASR execution mode selected by the Rust control
/// plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AsrWorkerMode {
    /// Local Whisper via worker protocol V2 prepared-audio requests.
    LocalWhisperV2,
    /// Tencent ASR via worker protocol V2 provider-media requests.
    HkTencentV2,
    /// Aliyun ASR via worker protocol V2 provider-media requests.
    HkAliyunV2,
    /// FunAudio ASR via worker protocol V2 provider-media requests.
    HkFunaudioV2,
}

impl AsrWorkerMode {
    /// Select the concrete worker-side execution mode from the command option
    /// string.
    fn from_engine_name(engine_name: &str) -> Self {
        match engine_name {
            "tencent" => Self::HkTencentV2,
            "aliyun" => Self::HkAliyunV2,
            "funaudio" => Self::HkFunaudioV2,
            _ => Self::LocalWhisperV2,
        }
    }

    /// Return the corresponding live V2 backend.
    fn as_v2_backend(self) -> AsrBackendV2 {
        match self {
            Self::LocalWhisperV2 => AsrBackendV2::LocalWhisper,
            Self::HkTencentV2 => AsrBackendV2::HkTencent,
            Self::HkAliyunV2 => AsrBackendV2::HkAliyun,
            Self::HkFunaudioV2 => AsrBackendV2::HkFunaudio,
        }
    }
}

impl AsrBackend {
    /// Select the runtime boundary from the configured ASR engine string.
    pub(crate) fn from_engine_name(engine_name: &str) -> Self {
        if engine_name == "rev" {
            Self::RustRevAi
        } else {
            Self::Worker(AsrWorkerMode::from_engine_name(engine_name))
        }
    }
}

impl AsrBackend {
    fn comment_engine_name(self) -> &'static str {
        match self {
            Self::RustRevAi => "rev",
            Self::Worker(AsrWorkerMode::LocalWhisperV2) => "whisper",
            Self::Worker(AsrWorkerMode::HkTencentV2) => "tencent",
            Self::Worker(AsrWorkerMode::HkAliyunV2) => "aliyun",
            Self::Worker(AsrWorkerMode::HkFunaudioV2) => "funaudio",
        }
    }
}

/// Options controlling the transcribe pipeline.
#[derive(Clone)]
pub struct TranscribeOptions {
    /// Which runtime boundary owns raw ASR inference.
    pub(crate) backend: AsrBackend,
    /// Whether the command requested diarized speaker attribution.
    pub diarize: bool,
    /// Concrete speaker backend selected by Rust when dedicated diarization is needed.
    pub speaker_backend: Option<SpeakerBackendV2>,
    /// Language code (ISO 639-3).
    pub lang: String,
    /// Expected number of speakers for diarization.
    pub num_speakers: usize,
    /// Whether to run utterance segmentation after CHAT assembly.
    pub with_utseg: bool,
    /// Whether to run morphosyntax after CHAT assembly.
    pub with_morphosyntax: bool,
    /// Whether to override the cache for utseg/morphosyntax.
    pub override_cache: bool,
    /// Whether to generate `%wor` tiers in the transcribe output.
    ///
    /// Defaults to `false` (BA2 parity: `--wor` was opt-in for transcribe).
    pub write_wor: bool,
    /// Media filename for the @Media header.
    pub media_name: Option<String>,
    /// Rev.AI pre-submitted job ID (from preflight).
    pub rev_job_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Process a single audio file through the transcribe pipeline.
///
/// Returns the final serialized CHAT text.
///
/// # Pipeline stages
///
/// 1. **ASR inference** — invoke the selected ASR backend, get raw tokens
/// 2. **Post-processing** — compound merging, number expansion, retokenization
/// 3. **CHAT assembly** — build `ChatFile` AST from utterances
/// 4. **Utterance segmentation** (optional) — BERT-based re-segmentation
/// 5. **Morphosyntax** (optional) — POS/dependency tagging
pub(crate) async fn process_transcribe(
    audio_path: &Path,
    services: PipelineServices<'_>,
    opts: &TranscribeOptions,
    progress: Option<&ProgressSender>,
) -> Result<String, ServerError> {
    run_transcribe_pipeline(audio_path, services, opts, progress).await
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parameters for ASR worker inference.
pub(crate) struct AsrInferParams<'a> {
    /// Which runtime boundary owns raw ASR inference.
    pub backend: AsrBackend,
    /// Audio file to transcribe.
    pub audio_path: &'a Path,
    /// Language code (ISO 639-3).
    pub lang: &'a LanguageCode3,
    /// Expected number of speakers for diarization.
    pub num_speakers: NumSpeakers,
    /// Rev.AI pre-submitted job ID (from preflight).
    pub rev_job_id: Option<&'a str>,
}

/// Parameters for dedicated speaker-diarization inference.
pub(crate) struct SpeakerInferParams<'a> {
    /// Audio file to diarize.
    pub audio_path: &'a Path,
    /// Language code used for worker checkout/bootstrap.
    pub lang: &'a LanguageCode3,
    /// Expected number of speakers when known.
    pub expected_speakers: NumSpeakers,
    /// Dedicated diarization backend chosen by Rust.
    pub backend: SpeakerBackendV2,
}

/// Call the Python worker for ASR inference on a single audio file.
pub(crate) async fn infer_asr(
    pool: &WorkerPool,
    params: &AsrInferParams<'_>,
) -> Result<AsrResponse, ServerError> {
    match params.backend {
        AsrBackend::RustRevAi => {
            infer_revai_asr(
                params.audio_path,
                params.lang,
                params.num_speakers,
                params.rev_job_id,
            )
            .await
        }
        AsrBackend::Worker(worker_mode) => infer_asr_via_worker_v2(pool, params, worker_mode).await,
    }
}

/// Call the live V2 Python worker path for ASR inference on a single audio
/// file and normalize its typed result into the shared Rust ASR response
/// shape.
async fn infer_asr_via_worker_v2(
    pool: &WorkerPool,
    params: &AsrInferParams<'_>,
    worker_mode: AsrWorkerMode,
) -> Result<AsrResponse, ServerError> {
    let artifacts = PreparedArtifactRuntimeV2::new("asr_v2").map_err(|error| {
        ServerError::Validation(format!("failed to create ASR V2 artifact runtime: {error}"))
    })?;
    let request = build_asr_request_v2(
        artifacts.store(),
        AsrBuildInputV2 {
            ids: &PreparedAsrRequestIdsV2::new("asr-v2-request", "asr-v2-audio"),
            input: match worker_mode {
                AsrWorkerMode::LocalWhisperV2 => AsrInputSourceV2::PreparedAudio {
                    audio_path: params.audio_path,
                },
                AsrWorkerMode::HkTencentV2
                | AsrWorkerMode::HkAliyunV2
                | AsrWorkerMode::HkFunaudioV2 => AsrInputSourceV2::ProviderMedia {
                    media_path: params.audio_path,
                    num_speakers: params.num_speakers,
                },
            },
            lang: params.lang,
            backend: worker_mode.as_v2_backend(),
        },
    )
    .await
    .map_err(|error| {
        ServerError::Validation(format!(
            "failed to build worker protocol V2 ASR request: {error}"
        ))
    })?;

    let response = pool
        .dispatch_execute_v2(params.lang, &request)
        .await
        .map_err(ServerError::Worker)?;

    parse_asr_response_v2(&response, params.lang)
        .map_err(|error| ServerError::Validation(format!("ASR V2 response parse failed: {error}")))
}

/// Call the live V2 Python worker path for dedicated speaker diarization on a
/// single audio file.
pub(crate) async fn infer_speaker(
    pool: &WorkerPool,
    params: &SpeakerInferParams<'_>,
) -> Result<Vec<SpeakerSegmentV2>, ServerError> {
    let artifacts = PreparedArtifactRuntimeV2::new("speaker_v2").map_err(|error| {
        ServerError::Validation(format!(
            "failed to create speaker V2 artifact runtime: {error}"
        ))
    })?;
    let request = build_speaker_request_v2(
        artifacts.store(),
        SpeakerBuildInputV2 {
            ids: &PreparedSpeakerRequestIdsV2::new("speaker-v2-request", "speaker-v2-audio"),
            audio_path: params.audio_path,
            backend: params.backend,
            expected_speakers: Some(params.expected_speakers),
        },
    )
    .await
    .map_err(|error| {
        ServerError::Validation(format!(
            "failed to build worker protocol V2 speaker request: {error}"
        ))
    })?;

    let response = pool
        .dispatch_execute_v2(params.lang, &request)
        .await
        .map_err(ServerError::Worker)?;

    parse_speaker_result_v2(&response)
        .map(|result| result.segments.clone())
        .map_err(|error| {
            ServerError::Validation(format!("speaker V2 response parse failed: {error}"))
        })
}

/// Convert flat ASR tokens (with speaker labels) into speaker-grouped monologues.
///
/// Groups consecutive tokens by speaker. Adjacent tokens with the same speaker
/// are combined into a single monologue. Speaker changes create new monologues.
pub(crate) fn convert_asr_response(response: &AsrResponse, use_speaker_labels: bool) -> AsrOutput {
    if response.tokens.is_empty() {
        return AsrOutput {
            monologues: Vec::new(),
        };
    }

    let mut monologues: Vec<AsrMonologue> = Vec::new();
    let mut current_speaker: Option<usize> = None;
    let mut current_elements: Vec<AsrElement> = Vec::new();

    for token in &response.tokens {
        let speaker_idx = if use_speaker_labels {
            token
                .speaker
                .as_deref()
                .and_then(parse_speaker_label)
                .unwrap_or(0)
        } else {
            0
        };

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
            value: token.text.clone(),
            ts: token.start_s.unwrap_or(0.0),
            end_ts: token.end_s.unwrap_or(0.0),
            r#type: "text".to_string(),
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

fn parse_speaker_label(label: &str) -> Option<usize> {
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
        if utt.speaker > max_speaker {
            max_speaker = utt.speaker;
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
        langs: vec![opts.lang.clone()],
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
    Ok(insert_transcribe_comment(&to_chat_string(&chat_file), opts))
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use batchalign_chat_ops::build_chat::{self, TranscriptDescription};

    #[test]
    fn asr_backend_mapping_distinguishes_live_v2_worker_modes() {
        assert_eq!(AsrBackend::from_engine_name("rev"), AsrBackend::RustRevAi);
        assert_eq!(
            AsrBackend::from_engine_name("tencent"),
            AsrBackend::Worker(AsrWorkerMode::HkTencentV2)
        );
        assert_eq!(
            AsrBackend::from_engine_name("aliyun"),
            AsrBackend::Worker(AsrWorkerMode::HkAliyunV2)
        );
        assert_eq!(
            AsrBackend::from_engine_name("funaudio"),
            AsrBackend::Worker(AsrWorkerMode::HkFunaudioV2)
        );
        assert_eq!(
            AsrBackend::from_engine_name("whisper_oai"),
            AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2)
        );
    }

    fn sample_transcribe_options(backend: AsrBackend) -> TranscribeOptions {
        TranscribeOptions {
            backend,
            diarize: false,
            speaker_backend: None,
            lang: "fra".into(),
            num_speakers: 1,
            with_utseg: false,
            with_morphosyntax: false,
            override_cache: false,
            write_wor: false,
            media_name: Some("sample".into()),
            rev_job_id: None,
        }
    }

    #[test]
    fn build_empty_chat_text_includes_transcribe_comment() {
        let text =
            build_empty_chat_text(&sample_transcribe_options(AsrBackend::RustRevAi)).unwrap();

        assert!(text.contains(&format!(
            "@Comment:\tBatchalign {}, ASR Engine rev.",
            env!("CARGO_PKG_VERSION")
        )));
    }

    #[test]
    fn insert_transcribe_comment_inserts_header_before_utterances() {
        let desc = TranscriptDescription {
            langs: vec!["fra".into()],
            participants: vec![build_chat::ParticipantDesc {
                id: "PAR".into(),
                name: None,
                role: None,
                corpus: None,
            }],
            media_name: Some("sample".into()),
            media_type: Some("audio".into()),
            utterances: vec![build_chat::UtteranceDesc {
                speaker: "PAR".into(),
                words: Some(vec![build_chat::WordDesc {
                    text: "bonjour".into(),
                    start_ms: Some(0),
                    end_ms: Some(500),
                }]),
                text: None,
                start_ms: None,
                end_ms: None,
            }],
            write_wor: false,
        };
        let chat_file = build_chat::build_chat(&desc).unwrap();
        let text = to_chat_string(&chat_file);

        let text = insert_transcribe_comment(
            &text,
            &sample_transcribe_options(AsrBackend::Worker(AsrWorkerMode::LocalWhisperV2)),
        );

        let comment_pos = text
            .find(&format!(
                "@Comment:\tBatchalign {}, ASR Engine whisper.",
                env!("CARGO_PKG_VERSION")
            ))
            .unwrap();
        let utterance_pos = text.find("\n*PAR:").unwrap();
        assert!(comment_pos < utterance_pos);
    }

    #[test]
    fn test_convert_asr_response_groups_by_speaker() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "hello".into(),
                    start_s: Some(0.0),
                    end_s: Some(0.5),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "world".into(),
                    start_s: Some(0.5),
                    end_s: Some(1.0),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "hi".into(),
                    start_s: Some(1.0),
                    end_s: Some(1.5),
                    speaker: Some("1".into()),
                    confidence: None,
                },
            ],
            lang: "eng".to_string(),
        };

        let output = convert_asr_response(&response, true);
        assert_eq!(output.monologues.len(), 2);
        assert_eq!(output.monologues[0].speaker, 0);
        assert_eq!(output.monologues[0].elements.len(), 2);
        assert_eq!(output.monologues[1].speaker, 1);
        assert_eq!(output.monologues[1].elements.len(), 1);
    }

    #[test]
    fn test_convert_asr_response_handles_speaker_change_and_back() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "a".into(),
                    start_s: Some(0.0),
                    end_s: Some(0.3),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "b".into(),
                    start_s: Some(0.3),
                    end_s: Some(0.6),
                    speaker: Some("1".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "c".into(),
                    start_s: Some(0.6),
                    end_s: Some(0.9),
                    speaker: Some("0".into()),
                    confidence: None,
                },
            ],
            lang: "eng".to_string(),
        };

        let output = convert_asr_response(&response, true);
        assert_eq!(output.monologues.len(), 3);
        assert_eq!(output.monologues[0].speaker, 0);
        assert_eq!(output.monologues[1].speaker, 1);
        assert_eq!(output.monologues[2].speaker, 0);
    }

    #[test]
    fn test_convert_asr_response_empty() {
        let response = AsrResponse {
            tokens: vec![],
            lang: "eng".to_string(),
        };
        let output = convert_asr_response(&response, true);
        assert!(output.monologues.is_empty());
    }

    #[test]
    fn test_convert_asr_response_no_speaker_defaults_to_zero() {
        let response = AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(0.0),
                end_s: Some(0.5),
                speaker: None,
                confidence: None,
            }],
            lang: "eng".to_string(),
        };

        let output = convert_asr_response(&response, true);
        assert_eq!(output.monologues.len(), 1);
        assert_eq!(output.monologues[0].speaker, 0);
    }

    #[test]
    fn test_convert_asr_response_ignores_labels_when_diarization_disabled() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "hello".into(),
                    start_s: Some(0.0),
                    end_s: Some(0.5),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "world".into(),
                    start_s: Some(0.5),
                    end_s: Some(1.0),
                    speaker: Some("1".into()),
                    confidence: None,
                },
            ],
            lang: "eng".to_string(),
        };

        let output = convert_asr_response(&response, false);
        assert_eq!(output.monologues.len(), 1);
        assert_eq!(output.monologues[0].speaker, 0);
        assert_eq!(output.monologues[0].elements.len(), 2);
    }

    #[test]
    fn test_parse_speaker_label_accepts_suffix_format() {
        assert_eq!(parse_speaker_label("1"), Some(1));
        assert_eq!(parse_speaker_label("SPEAKER_2"), Some(2));
        assert_eq!(parse_speaker_label("not-a-speaker"), None);
    }

    #[test]
    fn test_response_has_speaker_labels_detects_numeric_suffixes() {
        let response = AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(0.0),
                end_s: Some(0.5),
                speaker: Some("SPEAKER_1".into()),
                confidence: None,
            }],
            lang: "eng".to_string(),
        };

        assert!(response_has_speaker_labels(&response));
    }

    #[test]
    fn test_generate_participant_ids() {
        let utterances = vec![
            asr_postprocess::Utterance {
                speaker: 0,
                words: vec![],
            },
            asr_postprocess::Utterance {
                speaker: 1,
                words: vec![],
            },
        ];
        let ids = generate_participant_ids(&utterances, 2);
        assert_eq!(ids, vec!["PAR", "INV"]);
    }

    #[test]
    fn test_generate_participant_ids_many_speakers() {
        let utterances = vec![asr_postprocess::Utterance {
            speaker: 9,
            words: vec![],
        }];
        let ids = generate_participant_ids(&utterances, 10);
        assert_eq!(ids.len(), 10);
        assert_eq!(ids[0], "PAR");
        assert_eq!(ids[8], "SP8");
        assert_eq!(ids[9], "SP9");
    }

    #[test]
    fn test_generate_standard_participant_ids_uses_chat_defaults_then_sp() {
        let ids = generate_standard_participant_ids(5);
        assert_eq!(ids, vec!["PAR", "INV", "CHI", "MOT", "FAT"]);
    }
}
