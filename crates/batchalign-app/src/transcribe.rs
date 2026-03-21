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

use crate::api::{
    DurationSeconds, LanguageCode3, LanguageSpec, NumSpeakers, RevAiJobId, WorkerLanguage,
};
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
use batchalign_chat_ops::asr_postprocess::{
    self, AsrElement, AsrElementKind, AsrMonologue, AsrOutput, AsrRawText, AsrTimestampSecs,
    SpeakerIndex,
};
use batchalign_chat_ops::build_chat::{self, TranscriptDescription};
use batchalign_chat_ops::serialize::to_chat_string;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::error::ServerError;
use crate::pipeline::PipelineServices;
use crate::runner::util::ProgressSender;
use crate::workflow::PerFileWorkflow;
use crate::workflow::transcribe::{TranscribeWorkflow, TranscribeWorkflowRequest};

// ---------------------------------------------------------------------------
// ASR response types (match Python inference/asr.py models)
// ---------------------------------------------------------------------------

/// A single raw ASR output token from the selected ASR backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrToken {
    /// Word text.
    pub text: String,
    /// Start time in seconds.
    pub start_s: Option<DurationSeconds>,
    /// End time in seconds.
    pub end_s: Option<DurationSeconds>,
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
    pub lang: LanguageCode3,
}

fn default_lang() -> LanguageCode3 {
    LanguageCode3::eng()
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
    /// Language specification — `Auto` for ASR auto-detect, or a resolved code.
    ///
    /// The type system enforces that post-ASR stages (utseg, morphotag) must
    /// resolve `Auto` to a concrete language before calling NLP workers.
    pub lang: LanguageSpec,
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
    pub rev_job_id: Option<RevAiJobId>,
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
    debug_dir: Option<&Path>,
) -> Result<String, ServerError> {
    TranscribeWorkflow
        .run(TranscribeWorkflowRequest {
            audio_path,
            services,
            options: opts,
            progress,
            debug_dir,
        })
        .await
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
    /// Language specification for ASR dispatch. May be `Auto` — the GPU
    /// worker and ASR engine handle auto-detect internally.
    pub lang: &'a LanguageSpec,
    /// Expected number of speakers for diarization.
    pub num_speakers: NumSpeakers,
    /// Rev.AI pre-submitted job ID (from preflight).
    pub rev_job_id: Option<&'a str>,
}

/// Parameters for dedicated speaker-diarization inference.
pub(crate) struct SpeakerInferParams<'a> {
    /// Audio file to diarize.
    pub audio_path: &'a Path,
    /// Language specification for worker dispatch. May be `Auto`.
    pub lang: &'a LanguageSpec,
    /// Expected number of speakers when known.
    pub expected_speakers: NumSpeakers,
    /// Dedicated diarization backend chosen by Rust.
    pub backend: SpeakerBackendV2,
}

fn asr_worker_languages(lang: &LanguageSpec) -> (WorkerLanguage, LanguageCode3) {
    (
        lang.to_worker_language(),
        // Documented default: Auto language → eng for CHAT header construction.
        // The real detected language will replace this if ASR auto-detection succeeds.
        lang.as_resolved()
            .cloned()
            .unwrap_or_else(LanguageCode3::eng),
    )
}

/// Call the Python worker for ASR inference on a single audio file.
pub(crate) async fn infer_asr(
    pool: &WorkerPool,
    params: &AsrInferParams<'_>,
) -> Result<AsrResponse, ServerError> {
    let (worker_lang, fallback_lang) = asr_worker_languages(params.lang);

    match params.backend {
        AsrBackend::RustRevAi => {
            // Rev.AI path receives the full LanguageSpec so it can pass
            // "auto" to Rev.AI and read the detected language from the job.
            infer_revai_asr(
                params.audio_path,
                params.lang,
                params.num_speakers,
                params.rev_job_id,
            )
            .await
        }
        AsrBackend::Worker(worker_mode) => {
            infer_asr_via_worker_v2(pool, params, worker_mode, &worker_lang, &fallback_lang).await
        }
    }
}

/// Call the live V2 Python worker path for ASR inference on a single audio
/// file and normalize its typed result into the shared Rust ASR response
/// shape.
async fn infer_asr_via_worker_v2(
    pool: &WorkerPool,
    params: &AsrInferParams<'_>,
    worker_mode: AsrWorkerMode,
    worker_lang: &WorkerLanguage,
    fallback_lang: &LanguageCode3,
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
            lang: worker_lang,
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
        .dispatch_execute_v2(worker_lang, &request)
        .await
        .map_err(ServerError::Worker)?;

    parse_asr_response_v2(&response, fallback_lang)
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

    // Documented default: Auto language → eng for worker dispatch.
    // Workers need a concrete language for model selection.
    let pool_lang = params
        .lang
        .as_resolved()
        .cloned()
        .unwrap_or_else(LanguageCode3::eng);
    let response = pool
        .dispatch_execute_v2(&pool_lang, &request)
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
///
/// Speaker labels from the ASR engine are **always** used when present,
/// matching batchalign2's `process_generation()` which unconditionally reads
/// `utterance["speaker"]` from Rev.AI monologues. The `--diarization` CLI flag
/// controls only whether a *dedicated* speaker model (Pyannote/NeMo) runs as a
/// separate pipeline stage — it does not suppress labels the ASR engine already
/// provides.
pub(crate) fn convert_asr_response(response: &AsrResponse) -> AsrOutput {
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

    #[test]
    fn asr_auto_uses_auto_worker_language_with_resolved_fallback() {
        let (worker_lang, fallback_lang) = asr_worker_languages(&LanguageSpec::Auto);
        assert_eq!(worker_lang, WorkerLanguage::Auto);
        assert_eq!(fallback_lang, LanguageCode3::eng());
    }

    #[test]
    fn asr_resolved_language_preserves_worker_and_fallback_values() {
        let lang = LanguageSpec::Resolved(LanguageCode3::fra());
        let (worker_lang, fallback_lang) = asr_worker_languages(&lang);
        assert_eq!(worker_lang, WorkerLanguage::from(LanguageCode3::fra()));
        assert_eq!(fallback_lang, LanguageCode3::fra());
    }

    fn sample_transcribe_options(backend: AsrBackend) -> TranscribeOptions {
        TranscribeOptions {
            backend,
            diarize: false,
            speaker_backend: None,
            lang: LanguageCode3::fra().into(),
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
                    text: asr_postprocess::ChatWordText::new("bonjour"),
                    start_ms: Some(0),
                    end_ms: Some(500),
                    ..Default::default()
                }]),
                text: None,
                start_ms: None,
                end_ms: None,
                lang: None,
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
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.5)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "world".into(),
                    start_s: Some(DurationSeconds(0.5)),
                    end_s: Some(DurationSeconds(1.0)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "hi".into(),
                    start_s: Some(DurationSeconds(1.0)),
                    end_s: Some(DurationSeconds(1.5)),
                    speaker: Some("1".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
        };

        let output = convert_asr_response(&response);
        assert_eq!(output.monologues.len(), 2);
        assert_eq!(output.monologues[0].speaker, SpeakerIndex(0));
        assert_eq!(output.monologues[0].elements.len(), 2);
        assert_eq!(output.monologues[1].speaker, SpeakerIndex(1));
        assert_eq!(output.monologues[1].elements.len(), 1);
    }

    #[test]
    fn test_convert_asr_response_handles_speaker_change_and_back() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "a".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.3)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "b".into(),
                    start_s: Some(DurationSeconds(0.3)),
                    end_s: Some(DurationSeconds(0.6)),
                    speaker: Some("1".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "c".into(),
                    start_s: Some(DurationSeconds(0.6)),
                    end_s: Some(DurationSeconds(0.9)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
        };

        let output = convert_asr_response(&response);
        assert_eq!(output.monologues.len(), 3);
        assert_eq!(output.monologues[0].speaker, 0);
        assert_eq!(output.monologues[1].speaker, 1);
        assert_eq!(output.monologues[2].speaker, 0);
    }

    #[test]
    fn test_convert_asr_response_empty() {
        let response = AsrResponse {
            tokens: vec![],
            lang: LanguageCode3::eng(),
        };
        let output = convert_asr_response(&response);
        assert!(output.monologues.is_empty());
    }

    #[test]
    fn test_convert_asr_response_no_speaker_defaults_to_zero() {
        let response = AsrResponse {
            tokens: vec![AsrToken {
                text: "hello".into(),
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: None,
                confidence: None,
            }],
            lang: LanguageCode3::eng(),
        };

        let output = convert_asr_response(&response);
        assert_eq!(output.monologues.len(), 1);
        assert_eq!(output.monologues[0].speaker, 0);
    }

    /// Regression test for Brian's bug report (2026-03-18): bare
    /// `batchalign3 transcribe` with no `--diarization` flag must still
    /// produce multi-speaker output when the ASR engine (Rev.AI) returns
    /// speaker-labeled monologues.
    ///
    /// In batchalign2, `process_generation()` unconditionally reads
    /// `utterance["speaker"]` from Rev.AI monologues. The `--diarize` flag
    /// only controls whether a *separate* Pyannote stage runs. BA3 must
    /// match this: speaker labels from the ASR engine are always used.
    #[test]
    fn test_convert_asr_response_always_uses_speaker_labels() {
        let response = AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "hello".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.5)),
                    speaker: Some("0".into()),
                    confidence: None,
                },
                AsrToken {
                    text: "world".into(),
                    start_s: Some(DurationSeconds(0.5)),
                    end_s: Some(DurationSeconds(1.0)),
                    speaker: Some("1".into()),
                    confidence: None,
                },
            ],
            lang: LanguageCode3::eng(),
        };

        // Speaker labels must be respected regardless of any diarization flag.
        // Previously this test asserted the opposite (1 monologue, speaker 0),
        // which enshrined the bug.
        let output = convert_asr_response(&response);
        assert_eq!(
            output.monologues.len(),
            2,
            "each speaker change must start a new monologue"
        );
        assert_eq!(output.monologues[0].speaker, 0);
        assert_eq!(output.monologues[0].elements.len(), 1);
        assert_eq!(output.monologues[1].speaker, 1);
        assert_eq!(output.monologues[1].elements.len(), 1);
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
                start_s: Some(DurationSeconds(0.0)),
                end_s: Some(DurationSeconds(0.5)),
                speaker: Some("SPEAKER_1".into()),
                confidence: None,
            }],
            lang: LanguageCode3::eng(),
        };

        assert!(response_has_speaker_labels(&response));
    }

    #[test]
    fn test_generate_participant_ids() {
        let utterances = vec![
            asr_postprocess::Utterance {
                speaker: SpeakerIndex(0),
                words: vec![],
                lang: None,
            },
            asr_postprocess::Utterance {
                speaker: SpeakerIndex(1),
                words: vec![],
                lang: None,
            },
        ];
        let ids = generate_participant_ids(&utterances, 2);
        assert_eq!(ids, vec!["PAR", "INV"]);
    }

    #[test]
    fn test_generate_participant_ids_many_speakers() {
        let utterances = vec![asr_postprocess::Utterance {
            speaker: SpeakerIndex(9),
            words: vec![],
            lang: None,
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

    // -----------------------------------------------------------------------
    // Canned-response integration tests
    //
    // Exercise the full conversion chain with realistic ASR payloads:
    //   AsrResponse → convert_asr_response() → process_raw_asr()
    //   → generate_participant_ids() → transcript_from_asr_utterances()
    //   → build_chat() → to_chat_string()
    //
    // These catch bugs that unit tests on individual stages miss — the same
    // class of bugs that echo-worker integration tests failed to expose.
    // -----------------------------------------------------------------------

    /// Build a realistic canned Rev.AI-style response: 2 speakers, ~20 tokens
    /// each, with timing and speaker labels. Simulates a short interview.
    fn canned_revai_two_speaker_response() -> AsrResponse {
        AsrResponse {
            tokens: vec![
                // Speaker 0 — first turn
                AsrToken {
                    text: "so".into(),
                    start_s: Some(DurationSeconds(0.24)),
                    end_s: Some(DurationSeconds(0.42)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "tell".into(),
                    start_s: Some(DurationSeconds(0.42)),
                    end_s: Some(DurationSeconds(0.60)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "me".into(),
                    start_s: Some(DurationSeconds(0.60)),
                    end_s: Some(DurationSeconds(0.72)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "about".into(),
                    start_s: Some(DurationSeconds(0.72)),
                    end_s: Some(DurationSeconds(0.96)),
                    speaker: Some("0".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "your".into(),
                    start_s: Some(DurationSeconds(0.96)),
                    end_s: Some(DurationSeconds(1.14)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "experience".into(),
                    start_s: Some(DurationSeconds(1.14)),
                    end_s: Some(DurationSeconds(1.68)),
                    speaker: Some("0".into()),
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "with".into(),
                    start_s: Some(DurationSeconds(1.68)),
                    end_s: Some(DurationSeconds(1.86)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "the".into(),
                    start_s: Some(DurationSeconds(1.86)),
                    end_s: Some(DurationSeconds(1.98)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "program.".into(),
                    start_s: Some(DurationSeconds(1.98)),
                    end_s: Some(DurationSeconds(2.52)),
                    speaker: Some("0".into()),
                    confidence: Some(0.95),
                },
                // Speaker 1 — response
                AsrToken {
                    text: "well".into(),
                    start_s: Some(DurationSeconds(3.00)),
                    end_s: Some(DurationSeconds(3.24)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "I".into(),
                    start_s: Some(DurationSeconds(3.24)),
                    end_s: Some(DurationSeconds(3.36)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "started".into(),
                    start_s: Some(DurationSeconds(3.36)),
                    end_s: Some(DurationSeconds(3.72)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "about".into(),
                    start_s: Some(DurationSeconds(3.72)),
                    end_s: Some(DurationSeconds(3.96)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "3".into(),
                    start_s: Some(DurationSeconds(3.96)),
                    end_s: Some(DurationSeconds(4.14)),
                    speaker: Some("1".into()),
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "years".into(),
                    start_s: Some(DurationSeconds(4.14)),
                    end_s: Some(DurationSeconds(4.38)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "ago.".into(),
                    start_s: Some(DurationSeconds(4.38)),
                    end_s: Some(DurationSeconds(4.68)),
                    speaker: Some("1".into()),
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "it".into(),
                    start_s: Some(DurationSeconds(4.80)),
                    end_s: Some(DurationSeconds(4.92)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "was".into(),
                    start_s: Some(DurationSeconds(4.92)),
                    end_s: Some(DurationSeconds(5.10)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "really".into(),
                    start_s: Some(DurationSeconds(5.10)),
                    end_s: Some(DurationSeconds(5.40)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "helpful".into(),
                    start_s: Some(DurationSeconds(5.40)),
                    end_s: Some(DurationSeconds(5.82)),
                    speaker: Some("1".into()),
                    confidence: Some(0.96),
                },
                // Speaker 0 — follow-up
                AsrToken {
                    text: "that".into(),
                    start_s: Some(DurationSeconds(6.00)),
                    end_s: Some(DurationSeconds(6.18)),
                    speaker: Some("0".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "sounds".into(),
                    start_s: Some(DurationSeconds(6.18)),
                    end_s: Some(DurationSeconds(6.48)),
                    speaker: Some("0".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "great".into(),
                    start_s: Some(DurationSeconds(6.48)),
                    end_s: Some(DurationSeconds(6.78)),
                    speaker: Some("0".into()),
                    confidence: Some(0.99),
                },
                // Speaker 1 — closing
                AsrToken {
                    text: "yeah".into(),
                    start_s: Some(DurationSeconds(7.00)),
                    end_s: Some(DurationSeconds(7.24)),
                    speaker: Some("1".into()),
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "I".into(),
                    start_s: Some(DurationSeconds(7.24)),
                    end_s: Some(DurationSeconds(7.36)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
                AsrToken {
                    text: "would".into(),
                    start_s: Some(DurationSeconds(7.36)),
                    end_s: Some(DurationSeconds(7.56)),
                    speaker: Some("1".into()),
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "recommend".into(),
                    start_s: Some(DurationSeconds(7.56)),
                    end_s: Some(DurationSeconds(8.04)),
                    speaker: Some("1".into()),
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "it".into(),
                    start_s: Some(DurationSeconds(8.04)),
                    end_s: Some(DurationSeconds(8.16)),
                    speaker: Some("1".into()),
                    confidence: Some(0.99),
                },
            ],
            lang: LanguageCode3::eng(),
        }
    }

    /// Build a canned Whisper-style response: no speaker labels, single
    /// contiguous stream of tokens with timing.
    fn canned_whisper_no_speaker_response() -> AsrResponse {
        AsrResponse {
            tokens: vec![
                AsrToken {
                    text: "the".into(),
                    start_s: Some(DurationSeconds(0.0)),
                    end_s: Some(DurationSeconds(0.18)),
                    speaker: None,
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "quick".into(),
                    start_s: Some(DurationSeconds(0.18)),
                    end_s: Some(DurationSeconds(0.42)),
                    speaker: None,
                    confidence: Some(0.93),
                },
                AsrToken {
                    text: "brown".into(),
                    start_s: Some(DurationSeconds(0.42)),
                    end_s: Some(DurationSeconds(0.66)),
                    speaker: None,
                    confidence: Some(0.94),
                },
                AsrToken {
                    text: "fox".into(),
                    start_s: Some(DurationSeconds(0.66)),
                    end_s: Some(DurationSeconds(0.90)),
                    speaker: None,
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "jumps".into(),
                    start_s: Some(DurationSeconds(0.90)),
                    end_s: Some(DurationSeconds(1.20)),
                    speaker: None,
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "over".into(),
                    start_s: Some(DurationSeconds(1.20)),
                    end_s: Some(DurationSeconds(1.44)),
                    speaker: None,
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "the".into(),
                    start_s: Some(DurationSeconds(1.44)),
                    end_s: Some(DurationSeconds(1.56)),
                    speaker: None,
                    confidence: Some(0.98),
                },
                AsrToken {
                    text: "lazy".into(),
                    start_s: Some(DurationSeconds(1.56)),
                    end_s: Some(DurationSeconds(1.86)),
                    speaker: None,
                    confidence: Some(0.94),
                },
                AsrToken {
                    text: "dog.".into(),
                    start_s: Some(DurationSeconds(1.86)),
                    end_s: Some(DurationSeconds(2.22)),
                    speaker: None,
                    confidence: Some(0.96),
                },
                AsrToken {
                    text: "then".into(),
                    start_s: Some(DurationSeconds(2.40)),
                    end_s: Some(DurationSeconds(2.58)),
                    speaker: None,
                    confidence: Some(0.93),
                },
                AsrToken {
                    text: "it".into(),
                    start_s: Some(DurationSeconds(2.58)),
                    end_s: Some(DurationSeconds(2.70)),
                    speaker: None,
                    confidence: Some(0.97),
                },
                AsrToken {
                    text: "sat".into(),
                    start_s: Some(DurationSeconds(2.70)),
                    end_s: Some(DurationSeconds(2.94)),
                    speaker: None,
                    confidence: Some(0.95),
                },
                AsrToken {
                    text: "down".into(),
                    start_s: Some(DurationSeconds(2.94)),
                    end_s: Some(DurationSeconds(3.18)),
                    speaker: None,
                    confidence: Some(0.96),
                },
            ],
            lang: LanguageCode3::eng(),
        }
    }

    /// Run the full canned-response conversion chain and return CHAT text.
    ///
    /// Mirrors the pipeline stages in `pipeline/transcribe.rs`:
    /// `convert_asr_response` → `process_raw_asr` → `generate_participant_ids`
    /// → `transcript_from_asr_utterances` → `build_chat` → `to_chat_string`.
    fn run_canned_response_to_chat(
        response: &AsrResponse,
        num_speakers: usize,
        media_name: Option<&str>,
    ) -> String {
        let asr_output = convert_asr_response(response);
        let utterances = asr_postprocess::process_raw_asr(&asr_output, &response.lang);
        let participant_ids = generate_participant_ids(&utterances, num_speakers);
        let desc = build_chat::transcript_from_asr_utterances(
            &utterances,
            &participant_ids,
            &[response.lang.to_string()],
            media_name,
            false,
        );
        let chat_file = build_chat::build_chat(&desc).expect("build_chat must succeed");
        to_chat_string(&chat_file)
    }

    /// Full pipeline test: canned Rev.AI 2-speaker response produces valid
    /// multi-speaker CHAT with correct headers and timing.
    #[test]
    fn canned_revai_response_produces_multi_speaker_chat() {
        let response = canned_revai_two_speaker_response();
        let chat = run_canned_response_to_chat(&response, 2, Some("interview.mp3"));

        // Must have 2 @Participants entries
        let participants_line = chat
            .lines()
            .find(|l| l.starts_with("@Participants:"))
            .expect("@Participants header missing");
        assert!(
            participants_line.contains("PAR") && participants_line.contains("INV"),
            "expected PAR and INV in @Participants, got: {participants_line}"
        );

        // Must have 2 @ID lines
        let id_count = chat.lines().filter(|l| l.starts_with("@ID:")).count();
        assert_eq!(id_count, 2, "expected 2 @ID lines, got {id_count}");

        // Must have utterances from both speakers
        let par_count = chat.lines().filter(|l| l.starts_with("*PAR:")).count();
        let inv_count = chat.lines().filter(|l| l.starts_with("*INV:")).count();
        assert!(
            par_count >= 1,
            "expected at least 1 *PAR utterance, got {par_count}"
        );
        assert!(
            inv_count >= 1,
            "expected at least 1 *INV utterance, got {inv_count}"
        );

        // Timing bullets must be present (the \x15 delimiters)
        assert!(
            chat.contains('\x15'),
            "timing bullets missing from output CHAT"
        );

        // @Media header
        assert!(
            chat.contains("@Media:\tinterview, audio"),
            "expected @Media header with stripped extension"
        );

        // Must reparse cleanly
        let (_parsed, errors) = batchalign_chat_ops::parse::parse_lenient(&chat);
        assert!(
            errors.is_empty(),
            "generated CHAT must reparse cleanly: {errors:?}"
        );
    }

    /// Full pipeline test: canned Whisper response (no speaker labels) produces
    /// single-speaker CHAT with exactly 1 participant.
    #[test]
    fn canned_whisper_response_produces_single_speaker_chat() {
        let response = canned_whisper_no_speaker_response();
        let chat = run_canned_response_to_chat(&response, 1, Some("recording.wav"));

        // Must have exactly 1 participant
        let id_count = chat.lines().filter(|l| l.starts_with("@ID:")).count();
        assert_eq!(
            id_count, 1,
            "expected 1 @ID line for single-speaker, got {id_count}"
        );

        // All utterances must be from PAR (speaker 0)
        let non_par_utts: Vec<&str> = chat
            .lines()
            .filter(|l| l.starts_with('*') && !l.starts_with("*PAR:"))
            .collect();
        assert!(
            non_par_utts.is_empty(),
            "all utterances should be *PAR for single-speaker, found: {non_par_utts:?}"
        );

        // Must have at least 1 utterance
        let par_count = chat.lines().filter(|l| l.starts_with("*PAR:")).count();
        assert!(
            par_count >= 1,
            "expected at least 1 *PAR utterance, got {par_count}"
        );

        // Timing bullets must be present
        assert!(
            chat.contains('\x15'),
            "timing bullets missing from single-speaker output"
        );

        // Must reparse cleanly
        let (_parsed, errors) = batchalign_chat_ops::parse::parse_lenient(&chat);
        assert!(
            errors.is_empty(),
            "generated CHAT must reparse cleanly: {errors:?}"
        );
    }

    /// Regression test: Rev.AI response with speaker labels must produce
    /// multi-speaker output regardless of the diarization flag.
    ///
    /// This is the end-to-end version of the
    /// `test_convert_asr_response_always_uses_speaker_labels` unit test.
    /// It exercises the full chain through CHAT serialization to catch
    /// any stage that might collapse speakers.
    #[test]
    fn canned_revai_speaker_labels_produce_multi_speaker_regardless_of_diarize_flag() {
        let response = canned_revai_two_speaker_response();

        // The pipeline does not consult opts.diarize during
        // convert_asr_response → process_raw_asr → build_chat. Verify this
        // by running the same canned data through the conversion chain.
        let chat = run_canned_response_to_chat(&response, 2, Some("test.mp3"));

        // Count distinct speaker codes in utterance lines
        let speaker_codes: std::collections::BTreeSet<&str> = chat
            .lines()
            .filter(|l| l.starts_with('*'))
            .filter_map(|l| l.split(':').next())
            .map(|code| code.trim_start_matches('*'))
            .collect();
        assert!(
            speaker_codes.len() >= 2,
            "Rev.AI response with speaker labels must produce at least 2 distinct speakers \
             in the output CHAT, but only found: {speaker_codes:?}. \
             This was Brian's bug report: speaker labels from ASR must always be used."
        );
    }

    /// Whisper response (no speaker labels) should produce single-speaker
    /// output even when num_speakers > 1 — without dedicated diarization,
    /// Whisper tokens all default to speaker 0.
    #[test]
    fn canned_whisper_no_labels_stays_single_speaker_even_with_high_num_speakers() {
        let response = canned_whisper_no_speaker_response();
        // Pass num_speakers=3 — but since there are no labels, all tokens
        // map to speaker 0 and only PAR appears in the output.
        let chat = run_canned_response_to_chat(&response, 3, None);

        let speaker_codes: std::collections::BTreeSet<&str> = chat
            .lines()
            .filter(|l| l.starts_with('*'))
            .filter_map(|l| l.split(':').next())
            .map(|code| code.trim_start_matches('*'))
            .collect();
        assert_eq!(
            speaker_codes.len(),
            1,
            "Whisper response without speaker labels should produce exactly 1 speaker, got: {speaker_codes:?}"
        );
        assert!(speaker_codes.contains("PAR"), "sole speaker should be PAR");
    }

    /// Verify that number expansion works end-to-end in the canned Rev.AI
    /// response (the token "3" should become "three" in the output).
    #[test]
    fn canned_revai_response_expands_numbers() {
        let response = canned_revai_two_speaker_response();
        let chat = run_canned_response_to_chat(&response, 2, None);

        assert!(
            chat.contains("three"),
            "number '3' in canned response should be expanded to 'three' in CHAT output"
        );
        // The raw digit should not appear as a standalone word
        let has_raw_digit = chat
            .lines()
            .any(|l| l.starts_with('*') && l.split_whitespace().any(|w| w == "3"));
        assert!(
            !has_raw_digit,
            "raw digit '3' should not appear as a standalone word in utterance lines"
        );
    }

    /// Verify that embedded sentence-ending punctuation in canned responses
    /// (e.g. "program." or "ago.") splits correctly into utterance boundaries.
    #[test]
    fn canned_revai_response_splits_on_embedded_periods() {
        let response = canned_revai_two_speaker_response();
        let asr_output = convert_asr_response(&response);
        let utterances = asr_postprocess::process_raw_asr(&asr_output, &response.lang);

        // "program." and "ago." should create utterance boundaries, so we
        // expect more than 2 utterances from the 4-turn conversation.
        assert!(
            utterances.len() >= 3,
            "expected at least 3 utterances from embedded-period splitting, got {}",
            utterances.len()
        );

        // Every utterance must end with a terminator
        for (i, utt) in utterances.iter().enumerate() {
            let last = utt.words.last().expect("utterance should have words");
            assert!(
                matches!(last.text.as_str(), "." | "?" | "!"),
                "utterance {i} should end with a terminator, got: {:?}",
                last.text
            );
        }
    }
}
