"""Tests for Rust-based utterance segmentation pipeline.

Tests the ParsedChat handle methods for utterance segmentation and
the Python callback adapter.

Uses fake segmentation callbacks (no Stanza needed).
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
# Helpers: run utterance segmentation via ParsedChat handle
# ---------------------------------------------------------------------------

def _run_utseg(chat_text: str, callback, **kwargs) -> str:
    """Run utterance segmentation via ParsedChat handle and return serialized CHAT."""
    handle = batchalign_core.ParsedChat.parse(chat_text)
    handle.add_utterance_segmentation(callback, **kwargs)
    return handle.serialize()


def _run_utseg_batched(chat_text: str, batch_fn, **kwargs) -> str:
    """Run batched utterance segmentation via ParsedChat handle and return serialized CHAT."""
    handle = batchalign_core.ParsedChat.parse(chat_text)
    handle.add_utterance_segmentation_batched(batch_fn, **kwargs)
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

LONG_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies and he likes cake .
@End
"""

MULTI_UTT_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\tI eat cookies and he likes cake .
*MOT:\tgood job .
@End
"""


# ---------------------------------------------------------------------------
# Test: ParsedChat.add_utterance_segmentation with fake callbacks
# ---------------------------------------------------------------------------

class TestAddUtteranceSegmentationDirect:
    """Test the ParsedChat add_utterance_segmentation method directly."""

    def test_no_split(self) -> None:
        """Callback returns all-same-group -> utterance unchanged."""
        def no_split_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            return {"assignments": [0] * len(payload["words"])}

        result = _run_utseg(SIMPLE_CHAT, no_split_callback)
        # Should have exactly one utterance line
        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*CHI:")]
        assert len(utt_lines) == 1
        assert "I eat cookies" in utt_lines[0]

    def test_split_two_groups(self) -> None:
        """Callback splits into two groups -> two utterances produced."""
        def split_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            words = payload["words"]
            if len(words) == 7:
                return {"assignments": [0, 0, 0, 1, 1, 1, 1]}
            return {"assignments": [0] * len(words)}

        result = _run_utseg(LONG_CHAT, split_callback)
        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*CHI:")]
        assert len(utt_lines) == 2
        assert "I eat cookies" in utt_lines[0]
        assert "he likes cake" in utt_lines[1]

    def test_callback_receives_words_and_text(self) -> None:
        """Callback payload includes words list and text string."""
        payloads: list[dict[str, Any]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            payloads.append(payload)
            words = payloads[-1]["words"]
            return {"assignments": [0] * len(words)}

        _run_utseg(SIMPLE_CHAT, capture_callback)
        assert len(payloads) == 1
        assert "words" in payloads[0]
        assert "text" in payloads[0]
        assert payloads[0]["words"] == ["I", "eat", "cookies"]

    def test_multi_utterance_separate_callbacks(self) -> None:
        """Each utterance triggers its own callback invocation."""
        calls: list[list[str]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            calls.append(payload["words"])
            return {"assignments": [0] * len(payload["words"])}

        _run_utseg(MULTI_UTT_CHAT, capture_callback)
        assert len(calls) == 2
        assert calls[0] == ["I", "eat", "cookies", "and", "he", "likes", "cake"]
        assert calls[1] == ["good", "job"]

    def test_progress_callback(self) -> None:
        """Progress callback receives (completed, total) per utterance."""
        progress: list[tuple[int, int]] = []

        def fake_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            return {"assignments": [0] * len(payload["words"])}

        def progress_fn(completed: int, total: int) -> None:
            progress.append((completed, total))

        _run_utseg(MULTI_UTT_CHAT, fake_callback, progress_fn=progress_fn)
        assert len(progress) == 2
        assert progress[0] == (1, 2)
        assert progress[1] == (2, 2)

    def test_preserves_chat_structure(self) -> None:
        """Rust round-trip preserves the CHAT header."""
        def no_split(payload: dict[str, Any]) -> dict[str, list[int]]:
            return {"assignments": [0] * len(payload["words"])}

        result = _run_utseg(SIMPLE_CHAT, no_split)
        assert "@UTF8" in result
        assert "@Begin" in result
        assert "@Languages:" in result
        assert "@End" in result
        assert "*CHI:" in result

    def test_split_preserves_speaker(self) -> None:
        """Split utterances retain the same speaker code."""
        def split_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            words = payload["words"]
            if len(words) == 7:
                return {"assignments": [0, 0, 0, 1, 1, 1, 1]}
            return {"assignments": [0] * len(words)}

        result = _run_utseg(LONG_CHAT, split_callback)
        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*")]
        # Both split utterances should be from CHI
        for line in utt_lines:
            assert line.startswith("*CHI:")

    def test_split_all_get_terminators(self) -> None:
        """Each split utterance ends with a terminator (period)."""
        def split_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            words = payload["words"]
            if len(words) == 7:
                return {"assignments": [0, 0, 0, 1, 1, 1, 1]}
            return {"assignments": [0] * len(words)}

        result = _run_utseg(LONG_CHAT, split_callback)
        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*CHI:")]
        assert len(utt_lines) == 2
        for line in utt_lines:
            assert line.rstrip().endswith(".")

    def test_callbacks_pass_python_objects(self) -> None:
        """Callbacks receive a dict payload and accept dict response."""
        seen_types: list[type[Any]] = []

        def typed_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            seen_types.append(type(payload))
            return {"assignments": [0] * len(payload["words"])}

        result = _run_utseg(SIMPLE_CHAT, typed_callback)
        assert seen_types == [dict]
        assert "*CHI:" in result

    def test_callback_failure_preserves_original_chat(self) -> None:
        """A failing callback should not leave the ParsedChat handle partially segmented."""

        handle = batchalign_core.ParsedChat.parse(MULTI_UTT_CHAT)
        original = handle.serialize()
        calls = 0

        def flaky_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("utseg boom")
            return {"assignments": [0] * len(payload["words"])}

        with pytest.raises(Exception, match="utseg boom"):
            handle.add_utterance_segmentation(flaky_callback)

        assert calls == 2
        assert handle.serialize() == original


# ---------------------------------------------------------------------------
# Test: End-to-end Rust + callback integration
# ---------------------------------------------------------------------------

class TestRustEndToEnd:
    """Full round-trip: CHAT text -> Rust -> callback -> CHAT."""

    def test_round_trip_split(self) -> None:
        """Split callback produces multiple utterances via Rust orchestrator."""
        def split_callback(payload: dict[str, Any]) -> dict[str, list[int]]:
            words = payload["words"]
            if len(words) == 7:
                return {"assignments": [0, 0, 0, 1, 1, 1, 1]}
            return {"assignments": [0] * len(words)}

        result = _run_utseg(LONG_CHAT, split_callback)
        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*CHI:")]
        assert len(utt_lines) == 2


# ---------------------------------------------------------------------------
# Test: Batched utterance segmentation (ParsedChat.add_utterance_segmentation_batched)
# ---------------------------------------------------------------------------

class TestAddUtteranceSegmentationBatched:
    """Test the batched utterance segmentation via ParsedChat handle."""

    def test_batch_receives_object_array(self) -> None:
        """Batch callback receives a list of typed payload objects."""
        received: list[list[dict[str, Any]]] = []

        def batch_callback(items: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
            received.append(items)
            return [
                {"assignments": [0] * len(item["words"])} for item in items
            ]

        _run_utseg_batched(MULTI_UTT_CHAT, batch_callback)
        assert len(received) == 1  # single call
        assert len(received[0]) == 2  # two utterances
        assert received[0][0]["words"] == ["I", "eat", "cookies", "and", "he", "likes", "cake"]
        assert received[0][1]["words"] == ["good", "job"]

    def test_batch_splits_utterances(self) -> None:
        """Batch callback results that split utterances are applied."""
        def batch_callback(items: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
            results = []
            for item in items:
                words = item["words"]
                if len(words) == 7:
                    results.append({"assignments": [0, 0, 0, 1, 1, 1, 1]})
                else:
                    results.append({"assignments": [0] * len(words)})
            return results

        result = _run_utseg_batched(MULTI_UTT_CHAT, batch_callback)
        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*")]
        # CHI utterance split into 2, MOT stays 1 = 3 total
        assert len(utt_lines) == 3

    def test_batch_response_length_mismatch_raises(self) -> None:
        """Wrong response count raises an error."""
        def bad_callback(items: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
            return [{"assignments": [0]}]  # always 1

        with pytest.raises(Exception, match="length mismatch"):
            _run_utseg_batched(MULTI_UTT_CHAT, bad_callback)

    def test_batch_progress(self) -> None:
        """Progress callback fires during rebuild phase."""
        progress: list[tuple[int, int]] = []

        def batch_callback(items: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
            return [
                {"assignments": [0] * len(item["words"])} for item in items
            ]

        def progress_fn(completed: int, total: int) -> None:
            progress.append((completed, total))

        _run_utseg_batched(
            MULTI_UTT_CHAT, batch_callback, progress_fn=progress_fn,
        )
        assert len(progress) == 2
        assert progress[-1] == (2, 2)

    def test_batch_no_split_preserves_structure(self) -> None:
        """No-split batch result preserves CHAT structure."""
        def batch_callback(items: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
            return [
                {"assignments": [0] * len(item["words"])} for item in items
            ]

        result = _run_utseg_batched(SIMPLE_CHAT, batch_callback)
        assert "@UTF8" in result
        assert "@Begin" in result
        assert "*CHI:" in result
        assert "I eat cookies" in result
        assert "@End" in result

    def test_batch_progress_failure_preserves_original_chat(self) -> None:
        """A failing batched progress hook should not leave partial splits behind."""

        handle = batchalign_core.ParsedChat.parse(MULTI_UTT_CHAT)
        original = handle.serialize()

        def batch_callback(items: list[dict[str, Any]]) -> list[dict[str, list[int]]]:
            return [{"assignments": [0] * len(item["words"])} for item in items]

        def progress_fn(_completed: int, _total: int) -> None:
            raise RuntimeError("batched utseg progress boom")

        with pytest.raises(Exception, match="batched utseg progress boom"):
            handle.add_utterance_segmentation_batched(
                batch_callback,
                progress_fn=progress_fn,
            )

        assert handle.serialize() == original
