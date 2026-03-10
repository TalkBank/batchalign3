"""Typed schema models for worker protocol V2.

These models mirror `crates/batchalign-app/src/types/worker_v2.rs`. The live
worker loop now uses them for forced alignment, ASR, and speaker requests. The
shared fixture suite still matters because it keeps the remaining staged task
families and the cross-language contract honest.

- define the canonical V2 protocol shape in Python
- validate canonical fixture files shared with Rust
- prevent drift while the remaining task migrations are still staged

The design goal is to make Python a thin model host. Request and response
models therefore describe model-ready inputs plus prepared-artifact references,
not CLI commands or document-processing workflows.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field, FiniteFloat, StringConstraints, model_validator

from batchalign.inference._domain_types import LanguageCode, NumSpeakers
from batchalign.worker._types import WorkerJSONValue

WorkerRequestIdV2: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""Stable identifier for one V2 protocol request/response pair."""

WorkerArtifactIdV2: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""Stable identifier for one prepared worker artifact."""

WorkerArtifactPathV2: TypeAlias = Annotated[str, StringConstraints(min_length=1)]
"""Filesystem path to a prepared worker artifact."""

ProtocolVersionV2: TypeAlias = Annotated[int, Field(ge=2)]
"""Worker protocol major version."""

FiniteNonNegativeFloat: TypeAlias = Annotated[FiniteFloat, Field(ge=0)]
"""Finite floating-point value constrained to be non-negative."""


class WorkerKindV2(str, Enum):
    """Worker role selected during the V2 handshake."""

    INFER = "infer"


class InferenceTaskV2(str, Enum):
    """High-level V2 task family."""

    MORPHOSYNTAX = "morphosyntax"
    UTSEG = "utseg"
    TRANSLATE = "translate"
    COREF = "coref"
    ASR = "asr"
    FORCED_ALIGNMENT = "forced_alignment"
    SPEAKER = "speaker"
    OPENSMILE = "opensmile"
    AVQI = "avqi"


class AsrBackendV2(str, Enum):
    """ASR backend selected by Rust."""

    LOCAL_WHISPER = "local_whisper"
    HK_TENCENT = "hk_tencent"
    HK_ALIYUN = "hk_aliyun"
    HK_FUNAUDIO = "hk_funaudio"
    REVAI = "revai"


class FaBackendV2(str, Enum):
    """Forced-alignment backend selected by Rust."""

    WHISPER = "whisper"
    WAVE2VEC = "wave2vec"
    WAV2VEC_CANTO = "wav2vec_canto"


class SpeakerBackendV2(str, Enum):
    """Speaker backend selected by Rust."""

    PYANNOTE = "pyannote"
    NEMO = "nemo"


class WorkerAttachmentKindV2(str, Enum):
    """Small attachment vocabulary advertised in capabilities."""

    PREPARED_AUDIO = "prepared_audio"
    PREPARED_TEXT = "prepared_text"
    INLINE_JSON = "inline_json"
    PROVIDER_MEDIA = "provider_media"
    SUBMITTED_JOB = "submitted_job"


class PreparedAudioEncodingV2(str, Enum):
    """PCM encoding used for prepared audio artifacts."""

    PCM_F32LE = "pcm_f32le"


class PreparedTextEncodingV2(str, Enum):
    """Encoding used for prepared text artifacts."""

    UTF8_JSON = "utf8_json"


class ProtocolErrorCodeV2(str, Enum):
    """Error category for protocol-level failures."""

    UNSUPPORTED_PROTOCOL = "unsupported_protocol"
    INVALID_PAYLOAD = "invalid_payload"
    MISSING_ATTACHMENT = "missing_attachment"
    ATTACHMENT_UNREADABLE = "attachment_unreadable"
    MODEL_UNAVAILABLE = "model_unavailable"
    RUNTIME_FAILURE = "runtime_failure"


class FaTextModeV2(str, Enum):
    """Text-joining mode for forced-alignment payloads."""

    SPACE_JOINED = "space_joined"
    CHAR_JOINED = "char_joined"


class WorkerRuntimeInfoV2(BaseModel):
    """Runtime information returned during the V2 handshake."""

    python_version: str
    free_threaded: bool


class HelloRequestV2(BaseModel):
    """Initial handshake request sent by Rust."""

    protocol_version: ProtocolVersionV2
    worker_kind: WorkerKindV2


class HelloResponseV2(BaseModel):
    """Initial handshake response sent by Python."""

    protocol_version: ProtocolVersionV2
    worker_pid: int
    runtime: WorkerRuntimeInfoV2


class CapabilitiesRequestV2(BaseModel):
    """Request for task capability metadata."""

    request_id: WorkerRequestIdV2


class TaskCapabilityV2(BaseModel):
    """One task capability advertised by a V2 worker."""

    task: InferenceTaskV2
    accepted_inputs: list[WorkerAttachmentKindV2]
    supports_progress_events: bool


class CapabilitiesResponseV2(BaseModel):
    """Response describing task capabilities for the worker."""

    request_id: WorkerRequestIdV2
    tasks: list[TaskCapabilityV2]
    engine_versions: dict[str, str]


class PreparedAudioRefV2(BaseModel):
    """File-backed prepared audio artifact."""

    kind: Literal["prepared_audio"] = "prepared_audio"
    id: WorkerArtifactIdV2
    path: WorkerArtifactPathV2
    encoding: PreparedAudioEncodingV2
    channels: int = Field(ge=1)
    sample_rate_hz: int = Field(ge=1)
    frame_count: int = Field(ge=0)
    byte_offset: int = Field(ge=0)
    byte_len: int = Field(ge=0)


class PreparedTextRefV2(BaseModel):
    """File-backed prepared text artifact."""

    kind: Literal["prepared_text"] = "prepared_text"
    id: WorkerArtifactIdV2
    path: WorkerArtifactPathV2
    encoding: PreparedTextEncodingV2
    byte_offset: int = Field(ge=0)
    byte_len: int = Field(ge=0)


class InlineJsonRefV2(BaseModel):
    """Small inline JSON attachment."""

    kind: Literal["inline_json"] = "inline_json"
    id: WorkerArtifactIdV2
    value: WorkerJSONValue


ArtifactRefV2: TypeAlias = Annotated[
    PreparedAudioRefV2 | PreparedTextRefV2 | InlineJsonRefV2,
    Field(discriminator="kind"),
]
"""Prepared artifact descriptor carried alongside an execute request."""


class PreparedAudioInputV2(BaseModel):
    """Reference to a prepared audio attachment."""

    audio_ref_id: WorkerArtifactIdV2


class ProviderMediaInputV2(BaseModel):
    """Temporary cloud-provider media input retained during migration."""

    media_path: WorkerArtifactPathV2
    num_speakers: NumSpeakers


class SubmittedJobInputV2(BaseModel):
    """Previously submitted provider job id."""

    provider_job_id: WorkerArtifactIdV2


class PreparedAudioAsrInputV2(BaseModel):
    """ASR request input that references prepared audio."""

    kind: Literal["prepared_audio"] = "prepared_audio"
    data: PreparedAudioInputV2


class ProviderMediaAsrInputV2(BaseModel):
    """ASR request input that keeps a provider-local media path."""

    kind: Literal["provider_media"] = "provider_media"
    data: ProviderMediaInputV2


class SubmittedJobAsrInputV2(BaseModel):
    """ASR request input that polls an existing provider job."""

    kind: Literal["submitted_job"] = "submitted_job"
    data: SubmittedJobInputV2


AsrInputV2: TypeAlias = Annotated[
    PreparedAudioAsrInputV2 | ProviderMediaAsrInputV2 | SubmittedJobAsrInputV2,
    Field(discriminator="kind"),
]
"""Backend-specific ASR input transport."""


class AsrRequestV2(BaseModel):
    """V2 ASR request payload."""

    lang: LanguageCode
    backend: AsrBackendV2
    input: AsrInputV2


class ForcedAlignmentRequestV2(BaseModel):
    """V2 forced-alignment request payload."""

    backend: FaBackendV2
    payload_ref_id: WorkerArtifactIdV2
    audio_ref_id: WorkerArtifactIdV2
    text_mode: FaTextModeV2
    pauses: bool


class MorphosyntaxRequestV2(BaseModel):
    """V2 morphosyntax request payload."""

    lang: LanguageCode
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)


class UtsegRequestV2(BaseModel):
    """V2 utterance-segmentation request payload."""

    lang: LanguageCode
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)


class TranslateRequestV2(BaseModel):
    """V2 translation request payload."""

    source_lang: LanguageCode
    target_lang: LanguageCode
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)


class CorefRequestV2(BaseModel):
    """V2 coreference request payload."""

    lang: LanguageCode
    payload_ref_id: WorkerArtifactIdV2
    item_count: int = Field(ge=0)


class SpeakerRequestV2(BaseModel):
    """V2 speaker diarization request payload."""

    backend: SpeakerBackendV2
    input: "SpeakerInputV2"
    expected_speakers: NumSpeakers | None = None


class OpenSmileRequestV2(BaseModel):
    """V2 openSMILE request payload."""

    audio_ref_id: WorkerArtifactIdV2
    feature_set: str = "eGeMAPSv02"
    feature_level: str = "functionals"


class AvqiRequestV2(BaseModel):
    """V2 AVQI request payload."""

    cs_audio_ref_id: WorkerArtifactIdV2
    sv_audio_ref_id: WorkerArtifactIdV2


class SpeakerPreparedAudioInputV2(BaseModel):
    """Prepared-audio speaker input owned by Rust."""

    audio_ref_id: WorkerArtifactIdV2


class SpeakerPreparedAudioRefInputV2(BaseModel):
    """Tagged wrapper for prepared-audio speaker requests."""

    kind: Literal["prepared_audio"] = "prepared_audio"
    data: SpeakerPreparedAudioInputV2


SpeakerInputV2 = SpeakerPreparedAudioRefInputV2
"""Speaker input transport for the live V2 boundary."""


class AsrTaskRequestV2(BaseModel):
    """Tagged wrapper for an ASR execute payload."""

    kind: Literal["asr"] = "asr"
    data: AsrRequestV2


class ForcedAlignmentTaskRequestV2(BaseModel):
    """Tagged wrapper for a forced-alignment execute payload."""

    kind: Literal["forced_alignment"] = "forced_alignment"
    data: ForcedAlignmentRequestV2


class MorphosyntaxTaskRequestV2(BaseModel):
    """Tagged wrapper for a morphosyntax execute payload."""

    kind: Literal["morphosyntax"] = "morphosyntax"
    data: MorphosyntaxRequestV2


class UtsegTaskRequestV2(BaseModel):
    """Tagged wrapper for an utterance-segmentation execute payload."""

    kind: Literal["utseg"] = "utseg"
    data: UtsegRequestV2


class TranslateTaskRequestV2(BaseModel):
    """Tagged wrapper for a translation execute payload."""

    kind: Literal["translate"] = "translate"
    data: TranslateRequestV2


class CorefTaskRequestV2(BaseModel):
    """Tagged wrapper for a coreference execute payload."""

    kind: Literal["coref"] = "coref"
    data: CorefRequestV2


class SpeakerTaskRequestV2(BaseModel):
    """Tagged wrapper for a speaker execute payload."""

    kind: Literal["speaker"] = "speaker"
    data: SpeakerRequestV2


class OpenSmileTaskRequestV2(BaseModel):
    """Tagged wrapper for an openSMILE execute payload."""

    kind: Literal["opensmile"] = "opensmile"
    data: OpenSmileRequestV2


class AvqiTaskRequestV2(BaseModel):
    """Tagged wrapper for an AVQI execute payload."""

    kind: Literal["avqi"] = "avqi"
    data: AvqiRequestV2


TaskRequestV2: TypeAlias = Annotated[
    AsrTaskRequestV2
    | ForcedAlignmentTaskRequestV2
    | MorphosyntaxTaskRequestV2
    | UtsegTaskRequestV2
    | TranslateTaskRequestV2
    | CorefTaskRequestV2
    | SpeakerTaskRequestV2
    | OpenSmileTaskRequestV2
    | AvqiTaskRequestV2,
    Field(discriminator="kind"),
]
"""Typed execute payload carried by one V2 request."""


class ExecuteRequestV2(BaseModel):
    """One top-level V2 execution request."""

    request_id: WorkerRequestIdV2
    task: InferenceTaskV2
    payload: TaskRequestV2
    attachments: list[ArtifactRefV2]


class WhisperChunkSpanV2(BaseModel):
    """One raw Whisper chunk span returned by Python."""

    text: str
    start_s: FiniteNonNegativeFloat
    end_s: FiniteNonNegativeFloat

    @model_validator(mode="after")
    def _validate_range(self) -> WhisperChunkSpanV2:
        if self.end_s < self.start_s:
            raise ValueError("Whisper chunk end_s must be >= start_s")
        return self


class WhisperChunkResultPayloadV2(BaseModel):
    """Raw Whisper chunk output returned by Python."""

    lang: LanguageCode
    text: str
    chunks: list[WhisperChunkSpanV2]


class AsrElementKindV2(str, Enum):
    """Stable vocabulary for one monologue element returned by ASR."""

    TEXT = "text"
    PUNCTUATION = "punctuation"


class AsrElementV2(BaseModel):
    """One raw ASR element inside a speaker monologue."""

    value: str
    start_s: FiniteNonNegativeFloat | None = None
    end_s: FiniteNonNegativeFloat | None = None
    kind: AsrElementKindV2
    confidence: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> AsrElementV2:
        if self.start_s is not None and self.end_s is not None and self.end_s < self.start_s:
            raise ValueError("ASR element end_s must be >= start_s")
        return self


class AsrMonologueV2(BaseModel):
    """One speaker-attributed monologue returned by a provider backend."""

    speaker: str
    elements: list[AsrElementV2]


class MonologueAsrResultPayloadV2(BaseModel):
    """Provider-shaped ASR output returned as speaker monologues."""

    lang: LanguageCode
    monologues: list[AsrMonologueV2]


class WhisperTokenTimingV2(BaseModel):
    """One raw Whisper forced-alignment token onset."""

    text: str
    time_s: FiniteNonNegativeFloat


class WhisperTokenTimingResultPayloadV2(BaseModel):
    """Raw Whisper forced-alignment token output."""

    tokens: list[WhisperTokenTimingV2]


class IndexedWordTimingV2(BaseModel):
    """One word-level timing result."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    confidence: FiniteFloat | None = None

    @model_validator(mode="after")
    def _validate_range(self) -> IndexedWordTimingV2:
        if self.end_ms < self.start_ms:
            raise ValueError("Indexed word timing end_ms must be >= start_ms")
        return self


class IndexedWordTimingResultPayloadV2(BaseModel):
    """Forced-alignment indexed timing output."""

    indexed_timings: list[IndexedWordTimingV2 | None]


class MorphosyntaxItemResultV2(BaseModel):
    """One morphosyntax item result returned by Python."""

    raw_sentences: list[WorkerJSONValue] | None = None
    error: str | None = None


class MorphosyntaxResultPayloadV2(BaseModel):
    """Batched morphosyntax response payload."""

    items: list[MorphosyntaxItemResultV2]


class UtsegItemResultV2(BaseModel):
    """One utterance-segmentation item result returned by Python."""

    trees: list[str] | None = None
    error: str | None = None


class UtsegResultPayloadV2(BaseModel):
    """Batched utterance-segmentation response payload."""

    items: list[UtsegItemResultV2]


class TranslationItemResultV2(BaseModel):
    """One translation item result returned by Python."""

    raw_translation: str | None = None
    error: str | None = None


class TranslationResultPayloadV2(BaseModel):
    """Batched translation response payload."""

    items: list[TranslationItemResultV2]


class CorefChainRefV2(BaseModel):
    """One structured coreference chain reference returned by Python."""

    chain_id: int = Field(ge=0)
    is_start: bool
    is_end: bool


class CorefAnnotationV2(BaseModel):
    """One per-sentence coreference annotation returned by Python."""

    sentence_idx: int = Field(ge=0)
    words: list[list[CorefChainRefV2]]


class CorefItemResultV2(BaseModel):
    """One coreference item result returned by Python."""

    annotations: list[CorefAnnotationV2] | None = None
    error: str | None = None


class CorefResultPayloadV2(BaseModel):
    """Batched coreference response payload."""

    items: list[CorefItemResultV2]


class SpeakerSegmentV2(BaseModel):
    """One raw speaker diarization segment returned by Python."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    speaker: str

    @model_validator(mode="after")
    def _validate_range(self) -> SpeakerSegmentV2:
        if self.end_ms < self.start_ms:
            raise ValueError("Speaker segment end_ms must be >= start_ms")
        return self


class SpeakerResultPayloadV2(BaseModel):
    """Raw speaker diarization output returned by the model host."""

    segments: list[SpeakerSegmentV2]


class OpenSmileResultPayloadV2(BaseModel):
    """Raw openSMILE tabular output returned by the model host."""

    feature_set: str
    feature_level: str
    num_features: int = Field(ge=0)
    duration_segments: int = Field(ge=0)
    audio_file: str
    rows: list[dict[str, FiniteFloat]]
    success: bool
    error: str | None = None


class AvqiResultPayloadV2(BaseModel):
    """Raw AVQI metrics returned by the model host."""

    avqi: FiniteFloat
    cpps: FiniteFloat
    hnr: FiniteFloat
    shimmer_local: FiniteFloat
    shimmer_local_db: FiniteFloat
    slope: FiniteFloat
    tilt: FiniteFloat
    cs_file: str
    sv_file: str
    success: bool
    error: str | None = None


class WhisperChunkResultV2(BaseModel):
    """Tagged wrapper for raw Whisper chunk output."""

    kind: Literal["whisper_chunk_result"] = "whisper_chunk_result"
    data: WhisperChunkResultPayloadV2


class MonologueAsrResultV2(BaseModel):
    """Tagged wrapper for provider-shaped speaker-monologue output."""

    kind: Literal["monologue_asr_result"] = "monologue_asr_result"
    data: MonologueAsrResultPayloadV2


class WhisperTokenTimingResultV2(BaseModel):
    """Tagged wrapper for raw Whisper FA token output."""

    kind: Literal["whisper_token_timing_result"] = "whisper_token_timing_result"
    data: WhisperTokenTimingResultPayloadV2


class IndexedWordTimingResultV2(BaseModel):
    """Tagged wrapper for indexed alignment output."""

    kind: Literal["indexed_word_timing_result"] = "indexed_word_timing_result"
    data: IndexedWordTimingResultPayloadV2


class MorphosyntaxResultV2(BaseModel):
    """Tagged wrapper for batched morphosyntax output."""

    kind: Literal["morphosyntax_result"] = "morphosyntax_result"
    data: MorphosyntaxResultPayloadV2


class UtsegResultV2(BaseModel):
    """Tagged wrapper for batched utterance-segmentation output."""

    kind: Literal["utseg_result"] = "utseg_result"
    data: UtsegResultPayloadV2


class TranslationResultV2(BaseModel):
    """Tagged wrapper for translation output."""

    kind: Literal["translation_result"] = "translation_result"
    data: TranslationResultPayloadV2


class CorefResultV2(BaseModel):
    """Tagged wrapper for batched coreference output."""

    kind: Literal["coref_result"] = "coref_result"
    data: CorefResultPayloadV2


class SpeakerResultV2(BaseModel):
    """Tagged wrapper for raw speaker diarization output."""

    kind: Literal["speaker_result"] = "speaker_result"
    data: SpeakerResultPayloadV2


class OpenSmileResultV2(BaseModel):
    """Tagged wrapper for raw openSMILE output."""

    kind: Literal["opensmile_result"] = "opensmile_result"
    data: OpenSmileResultPayloadV2


class AvqiResultV2(BaseModel):
    """Tagged wrapper for raw AVQI output."""

    kind: Literal["avqi_result"] = "avqi_result"
    data: AvqiResultPayloadV2


TaskResultV2: TypeAlias = Annotated[
    WhisperChunkResultV2
    | MonologueAsrResultV2
    | WhisperTokenTimingResultV2
    | IndexedWordTimingResultV2
    | MorphosyntaxResultV2
    | UtsegResultV2
    | TranslationResultV2
    | CorefResultV2
    | SpeakerResultV2
    | OpenSmileResultV2
    | AvqiResultV2,
    Field(discriminator="kind"),
]
"""Typed execute result payload."""


class ExecuteSuccessV2(BaseModel):
    """Successful execute outcome."""

    kind: Literal["success"] = "success"


class ExecuteErrorV2(BaseModel):
    """Protocol/runtime failure outcome."""

    kind: Literal["error"] = "error"
    code: ProtocolErrorCodeV2
    message: str


ExecuteOutcomeV2: TypeAlias = Annotated[
    ExecuteSuccessV2 | ExecuteErrorV2,
    Field(discriminator="kind"),
]
"""Top-level execute outcome."""


class ExecuteResponseV2(BaseModel):
    """Top-level V2 execute response."""

    request_id: WorkerRequestIdV2
    outcome: ExecuteOutcomeV2
    result: TaskResultV2 | None = None
    elapsed_s: FiniteNonNegativeFloat


class ProgressEventV2(BaseModel):
    """Progress event emitted by long-running V2 tasks."""

    request_id: WorkerRequestIdV2
    completed: int = Field(ge=0)
    total: int = Field(ge=0)
    stage: str


class ShutdownRequestV2(BaseModel):
    """Shutdown request sent to a V2 worker."""

    request_id: WorkerRequestIdV2
