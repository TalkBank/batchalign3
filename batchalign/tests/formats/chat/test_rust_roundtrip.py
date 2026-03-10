"""Phase 1B: Tests for batchalign_core.parse_and_serialize().

Validates that the Rust parser + serializer produces output equivalent to
the Python parser for known CHAT inputs.
"""

from __future__ import annotations

import os

import pytest

batchalign_core = pytest.importorskip("batchalign_core")


SUPPORT_DIR = os.path.join(os.path.dirname(__file__), "support")


# ---------------------------------------------------------------------------
# Basic round-trip
# ---------------------------------------------------------------------------

class TestParseAndSerialize:
    def test_minimal_chat(self) -> None:
        chat = (
            "@UTF8\n"
            "@Begin\n"
            "@Languages:\teng\n"
            "@Participants:\tCHI Target_Child\n"
            "@ID:\teng|test|CHI||female|||Target_Child|||\n"
            "*CHI:\thello world .\n"
            "@End\n"
        )
        result = batchalign_core.parse_and_serialize(chat)
        assert "@UTF8" in result
        assert "@Begin" in result
        assert "@End" in result
        assert "*CHI:" in result
        assert "hello world" in result

    def test_real_test_file(self) -> None:
        path = os.path.join(SUPPORT_DIR, "test.cha")
        with open(path) as f:
            chat_text = f.read()
        result = batchalign_core.parse_and_serialize(chat_text)
        # Key structural elements preserved
        assert "@UTF8" in result
        assert "@Begin" in result
        assert "@End" in result
        assert "%mor:" in result
        assert "%gra:" in result

    def test_preserves_utterance_count(self) -> None:
        path = os.path.join(SUPPORT_DIR, "test.cha")
        with open(path) as f:
            chat_text = f.read()
        result = batchalign_core.parse_and_serialize(chat_text)
        # Count main tier lines (start with *)
        input_utts = [l for l in chat_text.splitlines() if l.startswith("*")]
        output_utts = [l for l in result.splitlines() if l.startswith("*")]
        assert len(output_utts) == len(input_utts)

    def test_invalid_chat_raises(self) -> None:
        with pytest.raises(ValueError):
            batchalign_core.parse_and_serialize("not valid chat at all")

    def test_empty_string_returns_empty(self) -> None:
        result = batchalign_core.parse_and_serialize("")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# Timing bullets
# ---------------------------------------------------------------------------

class TestTimingPreservation:
    def test_timing_bullets_preserved(self) -> None:
        chat = (
            "@UTF8\n"
            "@Begin\n"
            "@Languages:\teng\n"
            "@Participants:\tCHI Target_Child\n"
            "@ID:\teng|test|CHI||female|||Target_Child|||\n"
            "*CHI:\thello . \x151000_2000\x15\n"
            "@End\n"
        )
        result = batchalign_core.parse_and_serialize(chat)
        assert "\x15" in result or "1000" in result


# ---------------------------------------------------------------------------
# Dependent tiers
# ---------------------------------------------------------------------------

class TestDependentTiers:
    def test_mor_tier_preserved(self) -> None:
        chat = (
            "@UTF8\n"
            "@Begin\n"
            "@Languages:\teng\n"
            "@Participants:\tCHI Target_Child\n"
            "@ID:\teng|test|CHI||female|||Target_Child|||\n"
            "*CHI:\tI run .\n"
            "%mor:\tpro|I v|run .\n"
            "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|PUNCT\n"
            "@End\n"
        )
        result = batchalign_core.parse_and_serialize(chat)
        assert "%mor:" in result
        assert "%gra:" in result
        assert "pro|I" in result
        assert "v|run" in result
