"""Type stubs for the batchalign_core Rust worker runtime extension.

This stub exposes type information for mypy to check Python code
that calls into the Rust worker runtime.
"""

from collections.abc import Callable


# ---------------------------------------------------------------------------
# Worker protocol dispatch
# ---------------------------------------------------------------------------

def dispatch_protocol_message(
    message: object,
    *,
    health_fn: Callable[..., object],
    capabilities_fn: Callable[..., object],
    infer_fn: Callable[..., object],
    batch_infer_fn: Callable[..., object],
    execute_v2_fn: Callable[..., object],
    infer_request_model: object,
    batch_infer_request_model: object,
    execute_v2_request_model: object,
    validation_error_type: object,
) -> tuple[dict[str, object], bool]: ...


# ---------------------------------------------------------------------------
# Worker V2 execution
# ---------------------------------------------------------------------------

def execute_asr_request_v2(
    request: object,
    local_whisper_runner: Callable[..., object] | None = None,
    hk_tencent_runner: Callable[..., object] | None = None,
    hk_aliyun_runner: Callable[..., object] | None = None,
    hk_funaudio_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_forced_alignment_request_v2(
    request: object,
    whisper_runner: Callable[..., object] | None = None,
    wave2vec_runner: Callable[..., object] | None = None,
    canto_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_opensmile_request_v2(
    request: object,
    prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_avqi_request_v2(
    request: object,
    prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...

def execute_speaker_request_v2(
    request: object,
    pyannote_prepared_audio_runner: Callable[..., object] | None = None,
    nemo_prepared_audio_runner: Callable[..., object] | None = None,
) -> str: ...

def normalize_text_task_result(
    task: str,
    response: object,
    expected_count: int,
) -> str: ...

def align_tokens(
    original_words: list[str],
    stanza_tokens: list[str],
    alpha2: str = "",
) -> list[str | tuple[str, bool]]: ...


# ---------------------------------------------------------------------------
# Worker artifact loaders
# ---------------------------------------------------------------------------

def find_worker_attachment_by_id(
    attachments: object,
    artifact_id: str,
) -> str: ...

def load_worker_json_attachment(
    attachments: object,
    artifact_id: str,
) -> str: ...

def load_worker_prepared_text_json(
    attachment: object,
) -> str: ...

def load_worker_prepared_audio_f32le_bytes(
    attachment: object,
) -> bytes: ...


# ---------------------------------------------------------------------------
# HK/Cantonese ASR bridges
# ---------------------------------------------------------------------------

def clean_funaudio_segment_text(text: str) -> str: ...

def funaudio_segments_to_asr(segments: object, lang: str) -> str: ...

def tencent_result_detail_to_asr(result_detail: object, lang: str) -> str: ...

def aliyun_sentences_to_asr(sentences: object, lang: str) -> str: ...

def normalize_cantonese(text: str) -> str: ...

def cantonese_char_tokens(text: str) -> list[str]: ...


# ---------------------------------------------------------------------------
# Rev.AI native client
# ---------------------------------------------------------------------------

def rev_transcribe(
    audio_path: str,
    api_key: str,
    language: str,
    speakers_count: int | None = None,
    skip_postprocessing: bool = False,
    metadata: str | None = None,
) -> str: ...

def rev_get_timed_words(
    audio_path: str,
    api_key: str,
    language: str,
) -> str: ...

def rev_submit(
    audio_path: str,
    api_key: str,
    language: str,
    speakers_count: int | None = None,
    skip_postprocessing: bool = False,
    metadata: str | None = None,
) -> str: ...

def rev_poll(
    job_id: str,
    api_key: str,
    max_poll_secs: int = 30,
) -> str: ...

def rev_poll_timed_words(
    job_id: str,
    api_key: str,
    poll_secs: int = 15,
) -> str: ...


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli_main() -> None: ...
