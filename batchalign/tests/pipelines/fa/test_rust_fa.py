"""Tests for Rust-based forced alignment pipeline.

Tests the ParsedChat handle-based forced alignment API.
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
# Shared CHAT content for tests
# ---------------------------------------------------------------------------

TIMED_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@Media:\ttest, audio
*CHI:\thello world . \x150_5000\x15
@End
"""

TWO_UTT_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child, MOT Mother
@ID:\teng|test|CHI||female|||Target_Child|||
@ID:\teng|test|MOT|||||Mother|||
@Media:\ttest, audio
*CHI:\thello world . \x150_5000\x15
*MOT:\tgood job . \x155000_10000\x15
@End
"""

UNTIMED_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@Media:\ttest, audio
*CHI:\thello world .
@End
"""

RETRACE_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@Media:\ttest, audio
*CHI:\t<I want> [/] I need cookie . \x150_8000\x15
@End
"""

LONG_SPLIT_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@Media:\ttest, audio
*CHI:\thello . \x150_10000\x15
*CHI:\tworld . \x1510000_25000\x15
@End
"""


# ---------------------------------------------------------------------------
# FA response helpers
# ---------------------------------------------------------------------------

def _word_level_response(timings: list[dict[str, Any] | None]) -> dict[str, Any]:
    """Build a typed FaRawResponse::IndexedWordLevel payload."""
    return {"indexed_timings": timings}


def _token_level_response(tokens: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a typed FaRawResponse::TokenLevel payload."""
    return {"tokens": tokens}


def _word_timings(pairs: list[tuple[str, int, int] | None]) -> list[dict[str, Any] | None]:
    """Convert (word, start_ms, end_ms) tuples to FaIndexedTiming dicts."""
    result: list[dict[str, Any] | None] = []
    for p in pairs:
        if p is None:
            result.append(None)
        else:
            _, start, end = p
            result.append({"start_ms": start, "end_ms": end})
    return result


# ---------------------------------------------------------------------------
# ParsedChat handle helper
# ---------------------------------------------------------------------------

def _run_fa(chat_text: str, callback, **kwargs) -> str:
    """Run forced alignment via ParsedChat handle and return serialized CHAT."""
    handle = batchalign_core.ParsedChat.parse(chat_text)
    handle.add_forced_alignment(callback, **kwargs)
    return handle.serialize()


# ---------------------------------------------------------------------------
# Test: ParsedChat.add_forced_alignment with fake callbacks
# ---------------------------------------------------------------------------

class TestAddForcedAlignmentDirect:
    """Test the ParsedChat add_forced_alignment method directly with fake callbacks."""

    def test_simple_callback(self) -> None:
        """Basic: fake callback returns timings, Rust injects them."""
        def fake_callback(payload: dict[str, Any]) -> dict[str, Any]:
            words = payload["words"]
            assert words == ["hello", "world"]
            return _word_level_response(_word_timings([
                ("hello", 100, 2000), ("world", 2000, 4500),
            ]))

        result = _run_fa(TIMED_CHAT, fake_callback)
        assert "%wor:" in result

    def test_callback_receives_audio_range(self) -> None:
        """Callback payload includes audio_start_ms and audio_end_ms."""
        payloads: list[dict[str, Any]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, Any]:
            payloads.append(payload)
            return _word_level_response(_word_timings([
                ("hello", 100, 2000), ("world", 2000, 4500),
            ]))

        _run_fa(TIMED_CHAT, capture_callback)
        assert len(payloads) == 1
        assert payloads[0]["audio_start_ms"] == 0
        assert payloads[0]["audio_end_ms"] == 5000
        assert payloads[0]["word_ids"] == ["u0:w0", "u0:w1"]
        assert payloads[0]["word_utterance_indices"] == [0, 0]
        assert payloads[0]["word_utterance_word_indices"] == [0, 1]

    def test_callback_receives_pauses_flag(self) -> None:
        """pauses kwarg is forwarded to callback payload."""
        payloads: list[dict[str, Any]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, Any]:
            payloads.append(payload)
            return _word_level_response(_word_timings([
                ("hello", 100, 2000), ("world", 2000, 4500),
            ]))

        _run_fa(TIMED_CHAT, capture_callback, pauses=True)
        assert payloads[0]["pauses"] is True

    def test_multi_utterance(self) -> None:
        """Multiple utterances within same time window -> single group."""
        calls: list[list[str]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, Any]:
            calls.append(payload["words"])
            words = payload["words"]
            timings = _word_timings([
                (w, i * 1000, (i + 1) * 1000) for i, w in enumerate(words)
            ])
            return _word_level_response(timings)

        result = _run_fa(TWO_UTT_CHAT, capture_callback)
        # Both utterances fit in 20s window -> single group with all words
        total_words = sum(len(c) for c in calls)
        assert total_words == 4  # hello world good job
        assert "%wor:" in result

    def test_untimed_utterances_skipped(self) -> None:
        """Utterances without bullets are not included in groups (no total_audio_ms)."""
        calls: list[list[str]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, Any]:
            calls.append(payload["words"])
            return _word_level_response([])

        _run_fa(UNTIMED_CHAT, capture_callback)
        # No groups created -> no callback calls
        assert len(calls) == 0

    def test_untimed_estimated_with_total_audio(self) -> None:
        """Untimed utterances are included when total_audio_ms is provided."""
        calls: list[dict[str, Any]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, Any]:
            calls.append(payload)
            words = payload["words"]
            timings = _word_timings([
                (w, i * 1000, (i + 1) * 1000)
                for i, w in enumerate(words)  # type: ignore[union-attr]
            ])
            return _word_level_response(timings)

        result = _run_fa(
            UNTIMED_CHAT, capture_callback, total_audio_ms=10000,
        )
        # Untimed utterance should now be grouped
        assert len(calls) == 1
        assert calls[0]["words"] == ["hello", "world"]
        # FA should produce output with timing
        assert "%wor:" in result

    def test_callback_failure_preserves_original_chat(self) -> None:
        """A later-group callback failure should not leave partial timing injection behind."""

        handle = batchalign_core.ParsedChat.parse(TWO_UTT_CHAT)
        original = handle.serialize()
        calls = 0

        def flaky_callback(payload: dict[str, Any]) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("fa boom")
            words = payload["words"]
            timings = _word_timings([
                (w, i * 1000, (i + 1) * 1000) for i, w in enumerate(words)
            ])
            return _word_level_response(timings)

        with pytest.raises(Exception, match="fa boom"):
            handle.add_forced_alignment(
                flaky_callback,
                max_group_ms=1000,
            )

        assert calls == 2
        assert handle.serialize() == original

    def test_untimed_gets_bullet_after_fa(self) -> None:
        """Untimed utterances get utterance-level bullets after FA with estimation."""
        def fake_callback(payload: dict[str, Any]) -> dict[str, Any]:
            words = payload["words"]
            timings = _word_timings([
                (w, i * 1000, (i + 1) * 1000)
                for i, w in enumerate(words)  # type: ignore[union-attr]
            ])
            return _word_level_response(timings)

        result = _run_fa(
            UNTIMED_CHAT, fake_callback, total_audio_ms=10000,
        )
        # The output should contain a timing bullet on the utterance
        assert "\x15" in result

    def test_partial_timings_rejected(self) -> None:
        """Callback returning fewer timings than words raises contract error."""
        def callback(payload: dict[str, Any]) -> dict[str, Any]:
            # Return only one timing for a two-word utterance
            return _word_level_response([
                {"start_ms": 2000, "end_ms": 4500},
            ])

        with pytest.raises(ValueError, match="length mismatch"):
            _run_fa(TIMED_CHAT, callback)

    def test_empty_timings_rejected(self) -> None:
        """Callback returning empty timings list raises contract error."""
        def callback(payload: dict[str, Any]) -> dict[str, Any]:
            return _word_level_response([])

        with pytest.raises(ValueError, match="length mismatch"):
            _run_fa(TIMED_CHAT, callback)

    def test_progress_callback(self) -> None:
        """Progress callback receives (completed, total) per group."""
        progress: list[tuple[int, int]] = []

        def fake_callback(payload: dict[str, Any]) -> dict[str, Any]:
            words = payload["words"]
            timings = _word_timings([
                (w, i * 100, (i + 1) * 100)
                for i, w in enumerate(words)  # type: ignore[union-attr]
            ])
            return _word_level_response(timings)

        def progress_fn(completed: int, total: int) -> None:
            progress.append((completed, total))

        _run_fa(TIMED_CHAT, fake_callback, progress_fn=progress_fn)
        assert len(progress) >= 1
        # Last progress should be (total, total)
        assert progress[-1][0] == progress[-1][1]

    def test_groups_split_on_time(self) -> None:
        """Utterances exceeding max_group_ms are split into separate groups."""
        calls: list[dict[str, Any]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, Any]:
            calls.append(payload)
            words = payload["words"]
            timings = _word_timings([
                (w, i * 100, (i + 1) * 100)
                for i, w in enumerate(words)  # type: ignore[union-attr]
            ])
            return _word_level_response(timings)

        _run_fa(LONG_SPLIT_CHAT, capture_callback, max_group_ms=20000)
        # 0-10000 and 10000-25000 -> second group exceeds 20s from start -> split
        assert len(calls) == 2

    def test_retrace_words_included(self) -> None:
        """Retraced words (Wor domain) ARE included in FA groups."""
        words_seen: list[list[str]] = []

        def capture_callback(payload: dict[str, Any]) -> dict[str, Any]:
            words_seen.append(payload["words"])
            words = payload["words"]
            timings = _word_timings([
                (w, i * 100, (i + 1) * 100)
                for i, w in enumerate(words)  # type: ignore[union-attr]
            ])
            return _word_level_response(timings)

        _run_fa(RETRACE_CHAT, capture_callback)
        assert len(words_seen) == 1
        # Wor domain: retraced "I want" ARE included (they were spoken)
        assert len(words_seen[0]) == 5  # I want I need cookie

    def test_preserves_chat_structure(self) -> None:
        """Rust round-trip preserves CHAT headers and speaker lines."""
        def noop_callback(payload: dict[str, Any]) -> dict[str, Any]:
            words = payload["words"]
            return _word_level_response(
                _word_timings([(w, i * 100, (i + 1) * 100) for i, w in enumerate(words)])
            )

        result = _run_fa(TIMED_CHAT, noop_callback)
        assert "@UTF8" in result
        assert "@Begin" in result
        assert "@Languages:" in result
        assert "@End" in result
        assert "*CHI:" in result

    def test_callbacks_pass_python_objects(self) -> None:
        """Callbacks receive a dict payload and accept dict response."""
        seen_types: list[type[Any]] = []

        def typed_callback(payload: dict[str, Any]) -> dict[str, Any]:
            seen_types.append(type(payload))
            return {"indexed_timings": _word_timings([
                ("hello", 100, 2000), ("world", 2000, 4500),
            ])}

        result = _run_fa(TIMED_CHAT, typed_callback)
        assert seen_types == [dict]
        assert "%wor:" in result

    def test_wor_tier_has_timings(self) -> None:
        """Generated %wor tier contains timing bullets."""
        def fake_callback(payload_json: str) -> str:
            return _word_level_response(_word_timings([
                ("hello", 500, 2000), ("world", 2500, 4500),
            ]))

        result = _run_fa(TIMED_CHAT, fake_callback)
        # Find the %wor line
        wor_lines = [l for l in result.split("\n") if l.startswith("%wor:")]
        assert len(wor_lines) >= 1
        # %wor should contain bullet markers (U+0015)
        assert "\x15" in wor_lines[0]
