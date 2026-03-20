"""Tests for the staged worker-protocol V2 forced-alignment executor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from batchalign.worker._fa_v2 import (
    ForcedAlignmentExecutionHostV2,
    execute_forced_alignment_request_v2,
)
from batchalign.worker._types_v2 import (
    ExecuteErrorV2,
    ExecuteRequestV2,
    ExecuteSuccessV2,
    FaBackendV2,
    FaTextModeV2,
    ForcedAlignmentRequestV2,
    IndexedWordTimingResultV2,
    InferenceTaskV2,
    PreparedAudioEncodingV2,
    PreparedAudioRefV2,
    PreparedTextEncodingV2,
    PreparedTextRefV2,
    ProtocolErrorCodeV2,
    WhisperTokenTimingResultV2,
)


def _write_pcm_f32le(path: Path, samples: np.ndarray) -> None:
    """Write little-endian float32 PCM test data to disk."""

    path.write_bytes(samples.astype("<f4").tobytes())


def _write_payload(path: Path) -> None:
    """Write a minimal prepared FA payload for staged V2 executor tests."""

    path.write_text(
        json.dumps(
            {
                "words": ["hello", "world"],
                "word_ids": ["u0:w0", "u0:w1"],
                "word_utterance_indices": [0, 0],
                "word_utterance_word_indices": [0, 1],
            }
        ),
        encoding="utf-8",
    )


def _make_request(tmp_path: Path, *, backend: FaBackendV2, text_mode: FaTextModeV2) -> ExecuteRequestV2:
    """Create one staged V2 FA execute request with prepared artifacts."""

    payload_path = tmp_path / "payload.json"
    audio_path = tmp_path / "audio.pcm"
    _write_payload(payload_path)
    _write_pcm_f32le(audio_path, np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))

    return ExecuteRequestV2(
        request_id="req-fa-stage-1",
        task=InferenceTaskV2.FORCED_ALIGNMENT,
        payload=ForcedAlignmentRequestV2(
            backend=backend,
            payload_ref_id="payload-ref-1",
            audio_ref_id="audio-ref-1",
            text_mode=text_mode,
            pauses=True,
        ),
        attachments=[
            PreparedTextRefV2(
                id="payload-ref-1",
                path=str(payload_path),
                encoding=PreparedTextEncodingV2.UTF8_JSON,
                byte_offset=0,
                byte_len=payload_path.stat().st_size,
            ),
            PreparedAudioRefV2(
                id="audio-ref-1",
                path=str(audio_path),
                encoding=PreparedAudioEncodingV2.PCM_F32LE,
                channels=1,
                sample_rate_hz=16000,
                frame_count=4,
                byte_offset=0,
                byte_len=16,
            ),
        ],
    )


def test_executes_whisper_fa_v2_request(tmp_path: Path) -> None:
    """The staged V2 executor should return raw Whisper token timing output."""

    captured: dict[str, object] = {}

    def whisper_runner(audio: np.ndarray, text: str, pauses: bool) -> list[tuple[str, float]]:
        captured["shape"] = audio.shape
        captured["text"] = text
        captured["pauses"] = pauses
        return [("hello", 0.12), ("world", 0.38)]

    response = execute_forced_alignment_request_v2(
        _make_request(tmp_path, backend=FaBackendV2.WHISPER, text_mode=FaTextModeV2.SPACE_JOINED),
        ForcedAlignmentExecutionHostV2(whisper_runner=whisper_runner),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, WhisperTokenTimingResultV2)
    assert response.result.tokens[0].text == "hello"
    assert response.result.tokens[1].time_s == 0.38
    assert captured == {"shape": (4,), "text": "hello world", "pauses": True}


def test_executes_wave2vec_fa_v2_request(tmp_path: Path) -> None:
    """The staged V2 executor should project indexed word timings for wave2vec."""

    def wave2vec_runner(audio: np.ndarray, words: list[str]) -> list[tuple[str, tuple[int, int]]]:
        assert audio.shape == (4,)
        assert words == ["hello", "world"]
        return [("hello", (10, 40)), ("world", (40, 90))]

    response = execute_forced_alignment_request_v2(
        _make_request(tmp_path, backend=FaBackendV2.WAVE2VEC, text_mode=FaTextModeV2.SPACE_JOINED),
        ForcedAlignmentExecutionHostV2(wave2vec_runner=wave2vec_runner),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, IndexedWordTimingResultV2)
    assert response.result.indexed_timings[0].start_ms == 10
    assert response.result.indexed_timings[1].end_ms == 90


def test_executes_cantonese_wave2vec_fa_v2_request(tmp_path: Path) -> None:
    """The FA executor should preserve the internal Cantonese callback contract."""

    captured: dict[str, object] = {}

    def canto_runner(audio, payload, request):
        captured["shape"] = audio.shape
        captured["words"] = payload.words
        captured["text_mode"] = request.text_mode
        return [(payload.words[0], (50, 120)), (payload.words[1], (130, 220))]

    response = execute_forced_alignment_request_v2(
        _make_request(tmp_path, backend=FaBackendV2.WAV2VEC_CANTO, text_mode=FaTextModeV2.CHAR_JOINED),
        ForcedAlignmentExecutionHostV2(canto_runner=canto_runner),
    )

    assert isinstance(response.outcome, ExecuteSuccessV2)
    assert isinstance(response.result, IndexedWordTimingResultV2)
    assert response.result.indexed_timings[0].start_ms == 50
    assert response.result.indexed_timings[1].end_ms == 220
    assert captured == {
        "shape": (4,),
        "words": ["hello", "world"],
        "text_mode": FaTextModeV2.CHAR_JOINED,
    }


def test_returns_missing_attachment_error_for_invalid_request(tmp_path: Path) -> None:
    """Missing prepared artifacts should become typed protocol errors."""

    request = _make_request(tmp_path, backend=FaBackendV2.WHISPER, text_mode=FaTextModeV2.SPACE_JOINED)
    request.attachments = request.attachments[:1]

    response = execute_forced_alignment_request_v2(
        request,
        ForcedAlignmentExecutionHostV2(whisper_runner=lambda *_args: []),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MISSING_ATTACHMENT
    assert response.result is None


def test_invalid_numeric_attachment_becomes_invalid_payload_even_if_validation_is_bypassed(
    tmp_path: Path,
) -> None:
    """Rust should reject bad FA prepared-audio numerics even when Python is bypassed."""

    request = _make_request(tmp_path, backend=FaBackendV2.WHISPER, text_mode=FaTextModeV2.SPACE_JOINED)
    raw_attachment = request.attachments[1].model_dump()
    raw_attachment["sample_rate_hz"] = 0
    bad_attachment = PreparedAudioRefV2.model_construct(**raw_attachment)
    bad_request = ExecuteRequestV2.model_construct(
        request_id=request.request_id,
        task=request.task,
        payload=request.payload,
        attachments=[request.attachments[0], bad_attachment],
    )

    response = execute_forced_alignment_request_v2(
        bad_request,
        ForcedAlignmentExecutionHostV2(whisper_runner=lambda *_args: []),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.INVALID_PAYLOAD
    assert "positive sample_rate_hz" in response.outcome.message
    assert response.result is None


def test_returns_model_unavailable_for_unwired_cantonese_backend(tmp_path: Path) -> None:
    """Backends without an installed host should fail explicitly."""

    response = execute_forced_alignment_request_v2(
        _make_request(
            tmp_path,
            backend=FaBackendV2.WAV2VEC_CANTO,
            text_mode=FaTextModeV2.CHAR_JOINED,
        ),
        ForcedAlignmentExecutionHostV2(),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.MODEL_UNAVAILABLE
    assert "Cantonese FA host" in response.outcome.message


def test_invalid_whisper_fa_host_output_becomes_runtime_failure(tmp_path: Path) -> None:
    """Malformed Whisper FA host output should be classified as runtime failure."""

    response = execute_forced_alignment_request_v2(
        _make_request(tmp_path, backend=FaBackendV2.WHISPER, text_mode=FaTextModeV2.SPACE_JOINED),
        ForcedAlignmentExecutionHostV2(
            whisper_runner=lambda _audio, _text, _pauses: [("hello", float("nan"))]
        ),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.RUNTIME_FAILURE
    assert "invalid forced-alignment host output" in response.outcome.message
    assert response.result is None


def test_invalid_wave2vec_fa_host_output_becomes_runtime_failure(tmp_path: Path) -> None:
    """Malformed wave2vec FA host output should be classified as runtime failure."""

    response = execute_forced_alignment_request_v2(
        _make_request(tmp_path, backend=FaBackendV2.WAVE2VEC, text_mode=FaTextModeV2.SPACE_JOINED),
        ForcedAlignmentExecutionHostV2(
            wave2vec_runner=lambda _audio, _words: [("hello", (40, 10))]
        ),
    )

    assert isinstance(response.outcome, ExecuteErrorV2)
    assert response.outcome.code is ProtocolErrorCodeV2.RUNTIME_FAILURE
    assert "invalid forced-alignment host output" in response.outcome.message
    assert response.result is None
