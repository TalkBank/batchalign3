//! Worker protocol V2 schema types.
//!
//! These types define the next worker boundary described in
//! `book/src/developer/worker-protocol-v2.md`. Unlike the current
//! JSON-lines protocol in [`super::worker`], this schema is intentionally
//! staged for migration:
//!
//! - the types are drift-tested against Python
//! - canonical fixtures live under `tests/fixtures/worker_protocol_v2/`
//! - production code now dispatches FA, ASR, and speaker requests through
//!   these typed envelopes, while the remaining tasks are still staged
//!
//! The design goal is to keep Python as a thin model host while Rust owns
//! preprocessing, postprocessing, document semantics, and cache policy.
//!
//! ## Timing field validation contract
//!
//! Several response structs carry floating-point or integer timing fields
//! (`start_s`, `end_s`, `time_s`, `start_ms`, `end_ms`).  On the Python side,
//! Pydantic V2 models in `_types_v2.py` enforce upstream validation:
//! non-finite values (NaN, ±Inf) are rejected, and reversed ranges
//! (`start > end`) are rejected via `@model_validator`.  Rust deserializes
//! these fields permissively — `serde_json` will accept any valid JSON number
//! — because the Python worker has already sanitised the data before it
//! reaches the wire.  If a new producer is added that bypasses Python
//! validation, Rust-side checks must be added to the affected structs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{DurationMs, DurationSeconds, EngineVersion, LanguageCode3, NumSpeakers};
use crate::worker::WorkerPid;

string_id!(
    /// Stable identifier for one V2 protocol request/response pair.
    pub WorkerRequestIdV2
);

string_id!(
    /// Stable identifier for one prepared worker artifact.
    pub WorkerArtifactIdV2
);

string_id!(
    /// Filesystem path to a prepared worker artifact.
    pub WorkerArtifactPathV2
);

numeric_id!(
    /// Worker protocol major version.
    pub WorkerProtocolVersionV2(u16) [Eq]
);

numeric_id!(
    /// Audio sample rate in Hz carried by prepared artifacts.
    pub SampleRateHzV2(u32) [Eq]
);

numeric_id!(
    /// Number of channels in a prepared audio artifact.
    pub ChannelCountV2(u16) [Eq]
);

numeric_id!(
    /// Number of audio frames in a prepared artifact.
    pub FrameCountV2(u64) [Eq]
);

numeric_id!(
    /// Byte offset inside a prepared artifact file.
    pub ByteOffsetV2(u64) [Eq]
);

numeric_id!(
    /// Byte length inside a prepared artifact file.
    pub ByteLengthV2(u64) [Eq]
);

/// Worker role selected during the protocol handshake.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkerKindV2 {
    /// Stateless inference worker process.
    Infer,
}

/// High-level V2 task family.
#[derive(
    Debug,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    Hash,
    PartialOrd,
    Ord,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum InferenceTaskV2 {
    /// Morphosyntax tagging.
    Morphosyntax,
    /// Utterance segmentation.
    Utseg,
    /// Machine translation.
    Translate,
    /// Coreference annotation.
    Coref,
    /// Automatic speech recognition.
    Asr,
    /// Forced alignment.
    ForcedAlignment,
    /// Speaker diarization.
    Speaker,
    /// OpenSMILE feature extraction.
    Opensmile,
    /// AVQI feature extraction.
    Avqi,
}

/// ASR backend selected by the Rust control plane.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AsrBackendV2 {
    /// Local Whisper runtime hosted in Python.
    LocalWhisper,
    /// Tencent Cantonese ASR provider.
    HkTencent,
    /// Aliyun Cantonese ASR provider.
    HkAliyun,
    /// FunASR Cantonese provider.
    HkFunaudio,
    /// Rev.AI provider.
    Revai,
}

/// Forced-alignment backend selected by the Rust control plane.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FaBackendV2 {
    /// Whisper token-timestamp alignment.
    Whisper,
    /// MMS Wave2Vec forced alignment.
    Wave2vec,
    /// Cantonese Wave2Vec forced alignment.
    Wav2vecCanto,
}

/// Speaker diarization backend selected by Rust.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerBackendV2 {
    /// Pyannote diarization backend.
    Pyannote,
    /// NeMo diarization backend.
    Nemo,
}

/// Small artifact-kind vocabulary advertised in task capabilities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkerAttachmentKindV2 {
    /// File-backed prepared PCM audio.
    PreparedAudio,
    /// File-backed prepared text/JSON.
    PreparedText,
    /// Inline JSON attachment carried inside the envelope.
    InlineJson,
    /// Provider-local media path that Rust still has not replaced.
    ProviderMedia,
    /// Previously submitted provider job identifier.
    SubmittedJob,
}

/// PCM encoding used for prepared audio artifacts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PreparedAudioEncodingV2 {
    /// Little-endian float32 PCM frames.
    PcmF32le,
}

/// Encoding used for prepared text artifacts.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PreparedTextEncodingV2 {
    /// UTF-8 JSON text stored on disk.
    Utf8Json,
}

/// Error category for protocol-level failures.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolErrorCodeV2 {
    /// Worker/runtime does not understand the requested protocol version.
    UnsupportedProtocol,
    /// Request payload shape was invalid for the task.
    InvalidPayload,
    /// Required attachment was not supplied.
    MissingAttachment,
    /// Attachment existed logically but could not be read.
    AttachmentUnreadable,
    /// Model or SDK runtime for the task is unavailable.
    ModelUnavailable,
    /// Runtime failed while executing the task.
    RuntimeFailure,
}

/// Text-joining mode for forced-alignment payloads.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FaTextModeV2 {
    /// Join words with spaces before model invocation.
    SpaceJoined,
    /// Join words as character stream.
    CharJoined,
}

/// Runtime information returned during the V2 handshake.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct WorkerRuntimeInfoV2 {
    /// Python runtime version used by the worker.
    pub python_version: String,
    /// Whether the runtime is free-threaded.
    pub free_threaded: bool,
}

/// Initial V2 handshake request sent by Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct HelloRequestV2 {
    /// Requested protocol version.
    pub protocol_version: WorkerProtocolVersionV2,
    /// Worker role the parent process expects.
    pub worker_kind: WorkerKindV2,
}

/// Initial V2 handshake response sent by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct HelloResponseV2 {
    /// Agreed protocol version.
    pub protocol_version: WorkerProtocolVersionV2,
    /// OS process id of the worker.
    pub worker_pid: WorkerPid,
    /// Runtime metadata needed by the pool.
    pub runtime: WorkerRuntimeInfoV2,
}

/// Request for task capability metadata.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CapabilitiesRequestV2 {
    /// Correlation id for the capability lookup.
    pub request_id: WorkerRequestIdV2,
}

/// One task capability advertised by a V2 worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct TaskCapabilityV2 {
    /// Task family supported by the worker.
    pub task: InferenceTaskV2,
    /// Attachment/input kinds the task can consume.
    pub accepted_inputs: Vec<WorkerAttachmentKindV2>,
    /// Whether the task can emit progress events.
    pub supports_progress_events: bool,
}

/// Response describing task capabilities for the worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CapabilitiesResponseV2 {
    /// Correlation id that matches the request.
    pub request_id: WorkerRequestIdV2,
    /// Task capabilities advertised by the runtime.
    pub tasks: Vec<TaskCapabilityV2>,
    /// Engine version strings keyed by task name.
    pub engine_versions: BTreeMap<String, EngineVersion>,
}

/// File-backed prepared audio artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct PreparedAudioRefV2 {
    /// Stable artifact id referenced by request payloads.
    pub id: WorkerArtifactIdV2,
    /// Filesystem path to the prepared artifact.
    pub path: WorkerArtifactPathV2,
    /// PCM encoding for the prepared audio.
    pub encoding: PreparedAudioEncodingV2,
    /// Number of channels in the artifact view.
    pub channels: ChannelCountV2,
    /// Sample rate in Hz.
    pub sample_rate_hz: SampleRateHzV2,
    /// Number of frames in the view.
    pub frame_count: FrameCountV2,
    /// Byte offset inside the artifact file.
    pub byte_offset: ByteOffsetV2,
    /// Byte length inside the artifact file.
    pub byte_len: ByteLengthV2,
}

/// File-backed prepared text artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct PreparedTextRefV2 {
    /// Stable artifact id referenced by request payloads.
    pub id: WorkerArtifactIdV2,
    /// Filesystem path to the prepared artifact.
    pub path: WorkerArtifactPathV2,
    /// Encoding used by the file content.
    pub encoding: PreparedTextEncodingV2,
    /// Byte offset inside the artifact file.
    pub byte_offset: ByteOffsetV2,
    /// Byte length inside the artifact file.
    pub byte_len: ByteLengthV2,
}

/// Small inline JSON attachment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct InlineJsonRefV2 {
    /// Stable artifact id referenced by request payloads.
    pub id: WorkerArtifactIdV2,
    /// Inline JSON payload carried with the envelope.
    pub value: serde_json::Value,
}

/// Prepared artifact reference carried alongside one execute request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ArtifactRefV2 {
    /// Prepared PCM audio view.
    PreparedAudio(PreparedAudioRefV2),
    /// Prepared UTF-8 JSON or text view.
    PreparedText(PreparedTextRefV2),
    /// Small inline JSON attachment.
    InlineJson(InlineJsonRefV2),
}

/// Request-time reference to a prepared audio artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct PreparedAudioInputV2 {
    /// Artifact id of the audio descriptor included in `attachments`.
    pub audio_ref_id: WorkerArtifactIdV2,
}

/// Temporary cloud-provider media input retained during migration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ProviderMediaInputV2 {
    /// Media file path readable by the worker host.
    pub media_path: WorkerArtifactPathV2,
    /// Expected number of speakers for diarization-aware providers.
    pub num_speakers: NumSpeakers,
}

/// Previously submitted provider job id.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SubmittedJobInputV2 {
    /// Provider job identifier to poll.
    pub provider_job_id: WorkerArtifactIdV2,
}

/// ASR input variants for V2.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AsrInputV2 {
    /// Local prepared audio path.
    PreparedAudio(PreparedAudioInputV2),
    /// Provider-local media path.
    ProviderMedia(ProviderMediaInputV2),
    /// Previously submitted provider job.
    SubmittedJob(SubmittedJobInputV2),
}

/// V2 ASR request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct AsrRequestV2 {
    /// Input language for the transcript.
    pub lang: LanguageCode3,
    /// Backend selected by Rust.
    pub backend: AsrBackendV2,
    /// Backend-specific input transport.
    pub input: AsrInputV2,
}

/// V2 forced-alignment request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ForcedAlignmentRequestV2 {
    /// Backend selected by Rust.
    pub backend: FaBackendV2,
    /// Reference to the prepared text/JSON payload for the word arrays.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Reference to the prepared audio span.
    pub audio_ref_id: WorkerArtifactIdV2,
    /// Text shaping mode requested by Rust.
    pub text_mode: FaTextModeV2,
    /// Whether pause markers should be preserved.
    pub pauses: bool,
}

/// V2 morphosyntax request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct MorphosyntaxRequestV2 {
    /// Primary language routed by Rust.
    pub lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of utterance items frozen into the prepared batch payload.
    pub item_count: u32,
}

/// V2 utterance-segmentation request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UtsegRequestV2 {
    /// Primary language routed by Rust.
    pub lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of utterance items frozen into the prepared batch payload.
    pub item_count: u32,
}

/// V2 translation request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct TranslateRequestV2 {
    /// Source language determined by Rust.
    pub source_lang: LanguageCode3,
    /// Target language requested by Rust.
    pub target_lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of utterance items frozen into the prepared batch payload.
    pub item_count: u32,
}

/// V2 coreference request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefRequestV2 {
    /// Primary language routed by Rust.
    pub lang: LanguageCode3,
    /// Reference to the prepared text batch payload.
    pub payload_ref_id: WorkerArtifactIdV2,
    /// Number of document items frozen into the prepared batch payload.
    pub item_count: u32,
}

/// V2 speaker diarization request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerRequestV2 {
    /// Backend selected by Rust.
    pub backend: SpeakerBackendV2,
    /// Input transport for the speaker runtime.
    pub input: SpeakerInputV2,
    /// Expected number of speakers when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_speakers: Option<NumSpeakers>,
}

/// V2 openSMILE request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct OpenSmileRequestV2 {
    /// Reference to the prepared audio attachment.
    pub audio_ref_id: WorkerArtifactIdV2,
    /// Requested openSMILE feature-set name.
    pub feature_set: String,
    /// Requested openSMILE feature-level name.
    pub feature_level: String,
}

/// V2 AVQI request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct AvqiRequestV2 {
    /// Reference to the prepared continuous-speech audio attachment.
    pub cs_audio_ref_id: WorkerArtifactIdV2,
    /// Reference to the prepared sustained-vowel audio attachment.
    pub sv_audio_ref_id: WorkerArtifactIdV2,
}

/// Prepared-audio speaker input owned by Rust.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerPreparedAudioInputV2 {
    /// Artifact id of the prepared mono PCM audio view.
    pub audio_ref_id: WorkerArtifactIdV2,
}

/// Current input variants for speaker diarization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeakerInputV2 {
    /// Prepared mono PCM audio owned by Rust.
    PreparedAudio(SpeakerPreparedAudioInputV2),
}

/// Typed execute payload carried by one V2 request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskRequestV2 {
    /// Automatic speech recognition request.
    Asr(AsrRequestV2),
    /// Forced-alignment request.
    ForcedAlignment(ForcedAlignmentRequestV2),
    /// Morphosyntax request.
    Morphosyntax(MorphosyntaxRequestV2),
    /// Utterance-segmentation request.
    Utseg(UtsegRequestV2),
    /// Translation request.
    Translate(TranslateRequestV2),
    /// Coreference request.
    Coref(CorefRequestV2),
    /// Speaker diarization request.
    Speaker(SpeakerRequestV2),
    /// OpenSMILE request.
    Opensmile(OpenSmileRequestV2),
    /// AVQI request.
    Avqi(AvqiRequestV2),
}

/// One top-level V2 execution request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct ExecuteRequestV2 {
    /// Correlation id for the request.
    pub request_id: WorkerRequestIdV2,
    /// Task family being executed.
    pub task: InferenceTaskV2,
    /// Typed task payload.
    pub payload: TaskRequestV2,
    /// Prepared artifacts attached to the request.
    pub attachments: Vec<ArtifactRefV2>,
}

impl ExecuteRequestV2 {
    /// Return the timeout budget this request should receive on the worker
    /// transport.
    pub fn timeout_seconds(&self) -> u64 {
        self.payload.timeout_seconds()
    }

    /// Return the timeout with optional config overrides for audio and
    /// analysis tasks.
    pub fn timeout_seconds_with_config(
        &self,
        audio_timeout_s: u64,
        analysis_timeout_s: u64,
    ) -> u64 {
        self.payload
            .timeout_seconds_with_config(audio_timeout_s, analysis_timeout_s)
    }
}

impl TaskRequestV2 {
    /// Return the timeout budget this task family should receive on the worker
    /// transport.
    pub fn timeout_seconds(&self) -> u64 {
        self.timeout_seconds_with_config(0, 0)
    }

    /// Return the timeout with optional config overrides.
    ///
    /// When `audio_timeout_s` or `analysis_timeout_s` is 0, the built-in
    /// defaults (1800 and 120) are used.
    pub fn timeout_seconds_with_config(
        &self,
        audio_timeout_s: u64,
        analysis_timeout_s: u64,
    ) -> u64 {
        match self {
            Self::Morphosyntax(request) => batched_text_timeout_seconds(request.item_count),
            Self::Utseg(request) => batched_text_timeout_seconds(request.item_count),
            Self::Translate(request) => batched_text_timeout_seconds(request.item_count),
            Self::Coref(request) => batched_text_timeout_seconds(request.item_count),
            // Audio-based tasks can process files of arbitrary length.
            // A 30-minute recording can take 5+ minutes for Whisper inference;
            // a 2-hour file can take 20+ minutes.  Use a generous ceiling.
            Self::Asr(_) | Self::ForcedAlignment(_) | Self::Speaker(_) => {
                if audio_timeout_s > 0 {
                    audio_timeout_s
                } else {
                    1800
                }
            }
            // Lightweight audio analysis — 120s is sufficient.
            Self::Opensmile(_) | Self::Avqi(_) => {
                if analysis_timeout_s > 0 {
                    analysis_timeout_s
                } else {
                    120
                }
            }
        }
    }
}

/// Return the timeout budget for one batched text-inference request.
fn batched_text_timeout_seconds(item_count: u32) -> u64 {
    u64::from(item_count).saturating_mul(5).max(120)
}

/// One raw Whisper chunk span returned by Python.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperChunkSpanV2 {
    /// Surface text for the chunk.
    pub text: String,
    /// Start timestamp in seconds.
    pub start_s: DurationSeconds,
    /// End timestamp in seconds.
    pub end_s: DurationSeconds,
}

/// One ASR result built from raw Whisper chunks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperChunkResultV2 {
    /// Transcript language.
    pub lang: LanguageCode3,
    /// Concatenated transcript text.
    pub text: String,
    /// Raw chunk spans.
    pub chunks: Vec<WhisperChunkSpanV2>,
}

/// Stable vocabulary for one monologue element returned by an ASR provider.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AsrElementKindV2 {
    /// Lexical content that should become transcript tokens.
    Text,
    /// Punctuation emitted by the provider.
    Punctuation,
}

/// One raw ASR element inside a speaker monologue.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct AsrElementV2 {
    /// Surface token or punctuation value.
    pub value: String,
    /// Start timestamp in seconds when the provider exposes one.
    #[serde(default)]
    pub start_s: Option<DurationSeconds>,
    /// End timestamp in seconds when the provider exposes one.
    #[serde(default)]
    pub end_s: Option<DurationSeconds>,
    /// Stable element kind selected by the worker adapter.
    pub kind: AsrElementKindV2,
    /// Optional model/provider confidence score.
    #[serde(default)]
    pub confidence: Option<f64>,
}

/// One speaker-attributed monologue returned by a provider ASR backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct AsrMonologueV2 {
    /// Stable speaker label chosen by the worker adapter.
    pub speaker: String,
    /// Ordered elements inside the monologue.
    pub elements: Vec<AsrElementV2>,
}

/// One ASR result built from provider-shaped speaker monologues.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct MonologueAsrResultV2 {
    /// Transcript language reported by the provider adapter.
    pub lang: LanguageCode3,
    /// Speaker-grouped ASR output.
    pub monologues: Vec<AsrMonologueV2>,
}

/// One raw Whisper forced-alignment token span returned by Python.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperTokenTimingV2 {
    /// Surface token text returned by the FA runtime.
    pub text: String,
    /// Token onset timestamp in seconds.
    pub time_s: DurationSeconds,
}

/// Forced-alignment token response returned before Rust token-to-word
/// reconciliation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct WhisperTokenTimingResultV2 {
    /// Raw token timings in model order.
    pub tokens: Vec<WhisperTokenTimingV2>,
}

/// One word-level timing result.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct IndexedWordTimingV2 {
    /// Start time in milliseconds.
    pub start_ms: DurationMs,
    /// End time in milliseconds.
    pub end_ms: DurationMs,
    /// Optional model confidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

/// Forced-alignment indexed response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct IndexedWordTimingResultV2 {
    /// Indexed timing results aligned to the request words.
    pub indexed_timings: Vec<Option<IndexedWordTimingV2>>,
}

/// One morphosyntax item result returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct MorphosyntaxItemResultV2 {
    /// Raw Stanza `doc.to_dict()` sentence arrays when inference succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_sentences: Option<Vec<serde_json::Value>>,
    /// Optional per-item runtime error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Batched morphosyntax response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct MorphosyntaxResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<MorphosyntaxItemResultV2>,
}

/// One utseg item result returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UtsegItemResultV2 {
    /// Raw constituency trees when inference succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trees: Option<Vec<String>>,
    /// Optional per-item runtime error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Batched utterance-segmentation response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct UtsegResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<UtsegItemResultV2>,
}

/// One translation item result returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct TranslationItemResultV2 {
    /// Raw model translation when inference succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_translation: Option<String>,
    /// Optional per-item runtime error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Batched translation response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct TranslationResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<TranslationItemResultV2>,
}

/// One structured coreference chain reference returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefChainRefV2 {
    /// Chain identifier assigned by the coreference runtime.
    pub chain_id: usize,
    /// Whether the current word starts this mention.
    pub is_start: bool,
    /// Whether the current word ends this mention.
    pub is_end: bool,
}

/// One per-sentence coreference annotation returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefAnnotationV2 {
    /// Sentence index inside the corresponding document item.
    pub sentence_idx: usize,
    /// Per-word chain references parallel to the sentence words.
    pub words: Vec<Vec<CorefChainRefV2>>,
}

/// One coreference item result returned by Python.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefItemResultV2 {
    /// Structured sparse annotations when inference succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<CorefAnnotationV2>>,
    /// Optional per-item runtime error.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Batched coreference response payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct CorefResultV2 {
    /// Item results aligned to the prepared batch payload order.
    pub items: Vec<CorefItemResultV2>,
}

/// One raw speaker diarization segment returned by Python.
///
/// Timing fields validated upstream by Python Pydantic models (see module docs).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerSegmentV2 {
    /// Segment start in milliseconds.
    pub start_ms: DurationMs,
    /// Segment end in milliseconds.
    pub end_ms: DurationMs,
    /// Stable speaker label chosen by the model adapter.
    pub speaker: String,
}

/// Raw speaker diarization output returned by the model host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct SpeakerResultV2 {
    /// Ordered diarization segments.
    pub segments: Vec<SpeakerSegmentV2>,
}

/// Raw openSMILE output returned by the model host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct OpenSmileResultV2 {
    /// Requested feature-set name.
    pub feature_set: String,
    /// Requested feature-level name.
    pub feature_level: String,
    /// Number of extracted feature columns.
    pub num_features: u64,
    /// Number of result rows/segments.
    pub duration_segments: u64,
    /// Source audio identifier echoed by the worker.
    pub audio_file: String,
    /// Tabular feature rows keyed by feature name.
    pub rows: Vec<std::collections::BTreeMap<String, f64>>,
    /// Whether the underlying runtime succeeded.
    pub success: bool,
    /// Optional runtime error detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Raw AVQI output returned by the model host.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct AvqiResultV2 {
    /// AVQI score.
    pub avqi: f64,
    /// Cepstral Peak Prominence Smoothed.
    pub cpps: f64,
    /// Harmonics-to-noise ratio.
    pub hnr: f64,
    /// Local shimmer percentage.
    pub shimmer_local: f64,
    /// Local shimmer in dB.
    pub shimmer_local_db: f64,
    /// LTAS slope.
    pub slope: f64,
    /// LTAS tilt.
    pub tilt: f64,
    /// Continuous-speech file label echoed by the worker.
    pub cs_file: String,
    /// Sustained-vowel file label echoed by the worker.
    pub sv_file: String,
    /// Whether the runtime succeeded.
    pub success: bool,
    /// Optional runtime error detail.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Typed execute result payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TaskResultV2 {
    /// Whisper chunk output.
    WhisperChunkResult(WhisperChunkResultV2),
    /// Provider-shaped speaker monologue output.
    MonologueAsrResult(MonologueAsrResultV2),
    /// Raw Whisper FA token timings.
    WhisperTokenTimingResult(WhisperTokenTimingResultV2),
    /// Forced-alignment indexed timings.
    IndexedWordTimingResult(IndexedWordTimingResultV2),
    /// Batched morphosyntax result.
    MorphosyntaxResult(MorphosyntaxResultV2),
    /// Batched utterance-segmentation result.
    UtsegResult(UtsegResultV2),
    /// Batched translation result.
    TranslationResult(TranslationResultV2),
    /// Batched coreference result.
    CorefResult(CorefResultV2),
    /// Raw speaker diarization result.
    SpeakerResult(SpeakerResultV2),
    /// Raw openSMILE result.
    OpensmileResult(OpenSmileResultV2),
    /// Raw AVQI result.
    AvqiResult(AvqiResultV2),
}

/// Top-level execute outcome.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecuteOutcomeV2 {
    /// Request completed successfully.
    Success,
    /// Request failed at the protocol/runtime boundary.
    Error {
        /// Stable protocol error category.
        code: ProtocolErrorCodeV2,
        /// Human-readable detail for logs and tests.
        message: String,
    },
}

/// Top-level V2 execute response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
pub struct ExecuteResponseV2 {
    /// Correlation id for the request.
    pub request_id: WorkerRequestIdV2,
    /// Success or typed protocol/runtime error.
    pub outcome: ExecuteOutcomeV2,
    /// Typed task result when execution succeeded.
    #[serde(default)]
    pub result: Option<TaskResultV2>,
    /// Execution time in seconds.
    pub elapsed_s: DurationSeconds,
}

/// Progress event emitted by long-running V2 tasks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ProgressEventV2 {
    /// Correlation id of the request being updated.
    pub request_id: WorkerRequestIdV2,
    /// Completed units.
    pub completed: u32,
    /// Total units expected.
    pub total: u32,
    /// Stable progress stage label.
    pub stage: String,
}

/// Shutdown request sent to a V2 worker.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
pub struct ShutdownRequestV2 {
    /// Correlation id for the shutdown message.
    pub request_id: WorkerRequestIdV2,
}
