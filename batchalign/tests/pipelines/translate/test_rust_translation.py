"""Tests for Rust-based translation pipeline.

Tests the batchalign_core.ParsedChat handle-based translation entry point and
the Python callback adapter.

Uses fake translation functions (no network calls).
"""

from __future__ import annotations
from typing import Any

import json

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if batchalign_core is not built
# ---------------------------------------------------------------------------

batchalign_core = pytest.importorskip("batchalign_core")


# ---------------------------------------------------------------------------
# Helper: run translation via ParsedChat handle
# ---------------------------------------------------------------------------

def _run_translation(chat_text: str, callback, **kwargs) -> str:
    """Run translation via ParsedChat handle and return serialized CHAT."""
    handle = batchalign_core.ParsedChat.parse(chat_text)
    handle.add_translation(callback, **kwargs)
    return handle.serialize()


# ---------------------------------------------------------------------------
# Shared CHAT content for tests
# ---------------------------------------------------------------------------

SIMPLE_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
"""

MULTI_UTT_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\tI eat cookies .
*MOT:\tgood job .
@End
"""

EXISTING_TRANSLATION_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
%xtra:\told translation
@End
"""


# ---------------------------------------------------------------------------
# Test: ParsedChat.add_translation with a fake callback
# ---------------------------------------------------------------------------

class TestAddTranslationDirect:
    """Test the Rust add_translation method directly with fake callbacks."""

    def test_simple_callback(self) -> None:
        """Basic: fake callback returns translation, Rust injects %xtra tier."""
        def fake_callback(payload: dict[str, Any]) -> dict[str, str]:
            text = payload["text"]
            assert "I" in text or "eat" in text or "cookies" in text
            return {"translation": "I eat cookies"}

        result = _run_translation(SIMPLE_CHAT, fake_callback)
        assert "%xtra:" in result
        assert "I eat cookies" in result

    def test_callback_receives_text_and_speaker(self) -> None:
        """The callback payload includes text and speaker."""
        payloads: list[dict[str, Any]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, str]:
            payloads.append(payload)
            return {"translation": "translated"}

        _run_translation(SIMPLE_CHAT, capture_callback)
        assert len(payloads) == 1
        assert "text" in payloads[0]
        assert "speaker" in payloads[0]
        assert payloads[0]["speaker"] == "CHI"

    def test_multi_utterance(self) -> None:
        """Each utterance triggers a separate callback invocation."""
        calls: list[str] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, str]:
            calls.append(payload["text"])
            return {"translation": f"translated: {payload['text']}"}

        result = _run_translation(MULTI_UTT_CHAT, capture_callback)
        assert len(calls) == 2
        # Both utterances should have %xtra tiers
        lines = result.split("\n")
        xtra_lines = [l for l in lines if l.startswith("%xtra:")]
        assert len(xtra_lines) == 2

    def test_empty_translation_no_tier(self) -> None:
        """Empty translation in callback response -> no %xtra tier added."""
        def empty_callback(payload: dict[str, Any]) -> dict[str, str]:
            return {"translation": ""}

        result = _run_translation(SIMPLE_CHAT, empty_callback)
        assert "%xtra:" not in result

    def test_existing_translation_replaced(self) -> None:
        """Pre-existing %xtra tier is replaced with new translation."""
        def fake_callback(payload: dict[str, Any]) -> dict[str, str]:
            return {"translation": "new translation"}

        result = _run_translation(
            EXISTING_TRANSLATION_CHAT, fake_callback,
        )
        assert "new translation" in result
        assert "old translation" not in result

    def test_progress_callback(self) -> None:
        """Progress callback receives (completed, total) per utterance."""
        progress: list[tuple[int, int]] = []

        def fake_callback(payload: dict[str, Any]) -> dict[str, str]:
            return {"translation": "translated"}

        def progress_fn(completed: int, total: int) -> None:
            progress.append((completed, total))

        _run_translation(
            MULTI_UTT_CHAT, fake_callback, progress_fn=progress_fn,
        )
        assert len(progress) == 2
        assert progress[0] == (1, 2)
        assert progress[1] == (2, 2)

    def test_preserves_chat_structure(self) -> None:
        """Rust round-trip preserves the CHAT header and speaker lines."""
        def fake_callback(payload: dict[str, Any]) -> dict[str, str]:
            return {"translation": "translated"}

        result = _run_translation(SIMPLE_CHAT, fake_callback)
        assert "@UTF8" in result
        assert "@Begin" in result
        assert "@Languages:" in result
        assert "@End" in result
        assert "*CHI:" in result
        assert "I eat cookies" in result

    def test_callbacks_pass_python_objects(self) -> None:
        """Callbacks receive a dict payload and return a dict response."""
        seen_types: list[type[Any]] = []

        def typed_callback(payload: dict[str, Any]) -> dict[str, str]:
            seen_types.append(type(payload))
            assert payload["speaker"] == "CHI"
            return {"translation": f"typed: {payload['text']}"}

        result = _run_translation(SIMPLE_CHAT, typed_callback)
        assert seen_types == [dict]
        assert "typed: I eat cookies" in result

    def test_callback_failure_preserves_original_chat(self) -> None:
        """A failing callback should not leave the ParsedChat handle partially mutated."""

        handle = batchalign_core.ParsedChat.parse(MULTI_UTT_CHAT)
        original = handle.serialize()
        calls = 0

        def flaky_callback(payload: dict[str, Any]) -> dict[str, str]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("translation boom")
            return {"translation": f"translated: {payload['text']}"}

        with pytest.raises(Exception, match="translation boom"):
            handle.add_translation(flaky_callback)

        assert calls == 2
        assert handle.serialize() == original
