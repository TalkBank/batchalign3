"""Tests for typed model handles replacing object/tuple/monkey-patch patterns.

These tests verify the structural contracts of WhisperASRHandle,
WhisperFAHandle, and Wave2VecFAHandle without loading real ML models.
"""

from __future__ import annotations

from batchalign.inference.audio import bind_whisper_token_timestamp_extractor
from batchalign.inference.types import (
    Wave2VecFAHandle,
    WhisperASRHandle,
    WhisperFAHandle,
)


class TestWhisperASRHandle:
    """WhisperASRHandle replaces monkey-patched _ba_* attributes."""

    def test_stores_metadata(self) -> None:
        handle = WhisperASRHandle(
            pipe=lambda *a, **kw: {"chunks": []},
            config="fake_config",
            lang="english",
            sample_rate=16000,
        )
        assert handle.config == "fake_config"
        assert handle.lang == "english"
        assert handle.sample_rate == 16000

    def test_callable_forwards_to_pipe(self) -> None:
        calls: list[tuple[str, dict[str, int | dict[str, str]]]] = []

        def fake_pipe(audio: str, **kwargs: int | dict[str, str]) -> dict[str, list[dict[str, str | tuple[float, float]]]]:
            calls.append((audio, kwargs))
            return {"chunks": [{"text": "hello", "timestamp": (0.0, 1.0)}]}

        handle = WhisperASRHandle(
            pipe=fake_pipe,
            config="cfg",
            lang="english",
            sample_rate=16000,
        )
        result = handle("audio_data", batch_size=1, generate_kwargs={"task": "transcribe"})
        assert len(calls) == 1
        assert result["chunks"][0]["text"] == "hello"  # type: ignore[index]

    def test_gen_kwargs_normal_language(self) -> None:
        handle = WhisperASRHandle(
            pipe=None,
            config="my_config",
            lang="english",
            sample_rate=16000,
        )
        kw = handle.gen_kwargs("english")
        assert kw["task"] == "transcribe"
        assert kw["language"] == "english"
        assert kw["generation_config"] == "my_config"

    def test_gen_kwargs_cantonese(self) -> None:
        handle = WhisperASRHandle(
            pipe=None,
            config="my_config",
            lang="Cantonese",
            sample_rate=16000,
        )
        kw = handle.gen_kwargs("Cantonese")
        assert "task" not in kw
        assert "language" not in kw
        assert kw["generation_config"] == "my_config"


class TestWhisperFAHandle:
    """WhisperFAHandle replaces (model, processor, sample_rate) tuple."""

    def test_stores_fields(self) -> None:
        handle = WhisperFAHandle(
            model="fake_model",
            processor="fake_processor",
            sample_rate=16000,
        )
        assert handle.model == "fake_model"
        assert handle.processor == "fake_processor"
        assert handle.sample_rate == 16000

    def test_no_tuple_unpacking_needed(self) -> None:
        """Typed handle eliminates the tuple[object, ...] cast pattern."""
        handle = WhisperFAHandle(model="m", processor="p", sample_rate=22050)
        # Previously: _bundle: tuple[object, ...] = model_bundle  # type: ignore
        # Now: just access handle.model, handle.processor, handle.sample_rate
        assert isinstance(handle.sample_rate, int)

    def test_timestamp_extractor_binding_stays_instance_local(self) -> None:
        """The Whisper workaround should patch one model instance, not a class."""

        class _FakeModel:
            """Small test double for an instance-bound Whisper override."""

            marker = "fake-model"

        model = _FakeModel()
        bind_whisper_token_timestamp_extractor(model)  # type: ignore[arg-type]

        bound = model._extract_token_timestamps  # type: ignore[attr-defined]
        assert bound.__self__ is model


class TestWave2VecFAHandle:
    """Wave2VecFAHandle replaces (model, sample_rate) tuple."""

    def test_stores_fields(self) -> None:
        handle = Wave2VecFAHandle(model="fake_model", sample_rate=16000)
        assert handle.model == "fake_model"
        assert handle.sample_rate == 16000

    def test_no_tuple_unpacking_needed(self) -> None:
        handle = Wave2VecFAHandle(model="m", sample_rate=44100)
        assert isinstance(handle.sample_rate, int)
