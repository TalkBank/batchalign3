"""Tests for typed model handles — behavioral contracts only.

Verifies non-trivial logic in WhisperASRHandle (gen_kwargs branching,
callable forwarding) and WhisperFAHandle (instance-local monkey-patching).
Field-storage tests removed: dataclass fields storing what was passed in
is tested by Python itself, not us.
"""

from __future__ import annotations

from batchalign.inference.audio import bind_whisper_token_timestamp_extractor
from batchalign.inference.types import (
    WhisperASRHandle,
)


class TestWhisperASRHandle:
    """WhisperASRHandle replaces monkey-patched _ba_* attributes."""

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

    def test_gen_kwargs_auto_omits_language(self) -> None:
        """When lang is ``"auto"``, Whisper should auto-detect — no ``language`` key."""
        handle = WhisperASRHandle(
            pipe=None,
            config="my_config",
            lang="auto",
            sample_rate=16000,
        )
        kw = handle.gen_kwargs("auto")
        assert "language" not in kw, "auto-detect must omit 'language' so Whisper detects it"
        assert kw["generation_config"] == "my_config"
        assert kw["repetition_penalty"] == 1.001

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

    def test_timestamp_extractor_binding_stays_instance_local(self) -> None:
        """The Whisper workaround should patch one model instance, not a class."""

        class _FakeModel:
            """Small test double for an instance-bound Whisper override."""

            marker = "fake-model"

        model = _FakeModel()
        bind_whisper_token_timestamp_extractor(model)  # type: ignore[arg-type]

        bound = model._extract_token_timestamps  # type: ignore[attr-defined]
        assert bound.__self__ is model
