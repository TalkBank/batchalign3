"""Tests for Rust build_chat function.

Tests the batchalign_core.build_chat function that generates valid CHAT
files from JSON transcript descriptions (for ASR output).
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if batchalign_core is not built
# ---------------------------------------------------------------------------

batchalign_core = pytest.importorskip("batchalign_core")


# ---------------------------------------------------------------------------
# Test: Basic CHAT generation
# ---------------------------------------------------------------------------

class TestBuildChatBasic:
    """Test basic CHAT file generation from JSON transcripts."""

    def test_simple_utterance(self) -> None:
        """Single utterance with timed words produces valid CHAT."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "PAR0", "name": "Participant", "role": "Participant"}
            ],
            "utterances": [
                {
                    "speaker": "PAR0",
                    "words": [
                        {"text": "hello", "start_ms": 100, "end_ms": 500},
                        {"text": "world", "start_ms": 500, "end_ms": 1000},
                        {"text": ".", "start_ms": None, "end_ms": None},
                    ],
                }
            ],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))

        assert "@UTF8" in result
        assert "@Begin" in result
        assert "@Languages:\teng" in result
        assert "@End" in result
        assert "*PAR0:" in result
        assert "hello world" in result

    def test_multi_utterance(self) -> None:
        """Multiple utterances from different speakers."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "PAR0", "name": "Speaker1", "role": "Participant"},
                {"id": "PAR1", "name": "Speaker2", "role": "Participant"},
            ],
            "utterances": [
                {
                    "speaker": "PAR0",
                    "words": [
                        {"text": "hello", "start_ms": 100, "end_ms": 500},
                        {"text": ".", "start_ms": None, "end_ms": None},
                    ],
                },
                {
                    "speaker": "PAR1",
                    "words": [
                        {"text": "hi", "start_ms": 600, "end_ms": 900},
                        {"text": ".", "start_ms": None, "end_ms": None},
                    ],
                },
            ],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))

        assert "*PAR0:" in result
        assert "*PAR1:" in result
        assert "hello" in result
        assert "hi" in result

    def test_question_terminator(self) -> None:
        """Question mark as last word produces question terminator."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "PAR0", "name": "Participant", "role": "Participant"}
            ],
            "utterances": [
                {
                    "speaker": "PAR0",
                    "words": [
                        {"text": "how", "start_ms": 100, "end_ms": 300},
                        {"text": "are", "start_ms": 300, "end_ms": 500},
                        {"text": "you", "start_ms": 500, "end_ms": 700},
                        {"text": "?", "start_ms": None, "end_ms": None},
                    ],
                }
            ],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))

        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*PAR0:")]
        assert len(utt_lines) == 1
        # Should have question mark terminator
        assert "?" in utt_lines[0]


# ---------------------------------------------------------------------------
# Test: Headers and metadata
# ---------------------------------------------------------------------------

class TestBuildChatHeaders:
    """Test header generation."""

    def test_languages_header(self) -> None:
        """Languages from JSON appear in @Languages header."""
        transcript = {
            "langs": ["eng", "spa"],
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "eng" in result
        assert "spa" in result

    def test_participant_id_header(self) -> None:
        """@ID header generated for each participant."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "CHI", "name": "Child", "role": "Target_Child"},
            ],
            "utterances": [],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "@ID:" in result
        assert "CHI" in result

    def test_media_header(self) -> None:
        """@Media header included when media_name provided."""
        transcript = {
            "langs": ["eng"],
            "media_name": "recording01",
            "media_type": "audio",
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "@Media:\trecording01, audio" in result

    def test_video_media_type(self) -> None:
        """media_type=video produces video in @Media."""
        transcript = {
            "langs": ["eng"],
            "media_name": "session01",
            "media_type": "video",
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "video" in result

    def test_no_media_header_when_absent(self) -> None:
        """No @Media header when media_name not provided."""
        transcript = {
            "langs": ["eng"],
            "write_wor": True,
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "@Media" not in result


# ---------------------------------------------------------------------------
# Test: Timing and %wor tier
# ---------------------------------------------------------------------------

class TestBuildChatTiming:
    """Test timing bullets and %wor tier generation."""

    def test_wor_tier_with_timing(self) -> None:
        """Timed words produce a %wor tier."""
        transcript = {
            "langs": ["eng"],
            "write_wor": True,
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [
                {
                    "speaker": "PAR0",
                    "words": [
                        {"text": "hello", "start_ms": 100, "end_ms": 500},
                        {"text": ".", "start_ms": None, "end_ms": None},
                    ],
                }
            ],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "%wor:" in result

    def test_no_wor_tier_without_timing(self) -> None:
        """Untimed words produce no %wor tier."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [
                {
                    "speaker": "PAR0",
                    "words": [
                        {"text": "hello", "start_ms": None, "end_ms": None},
                        {"text": ".", "start_ms": None, "end_ms": None},
                    ],
                }
            ],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "%wor:" not in result

    def test_utterance_bullet(self) -> None:
        """Utterance-level timing bullet present."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [
                {
                    "speaker": "PAR0",
                    "words": [
                        {"text": "hello", "start_ms": 100, "end_ms": 500},
                        {"text": "world", "start_ms": 500, "end_ms": 1000},
                        {"text": ".", "start_ms": None, "end_ms": None},
                    ],
                }
            ],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        # Utterance line should contain timing bullet with start_end
        lines = result.split("\n")
        utt_lines = [l for l in lines if l.startswith("*PAR0:")]
        assert len(utt_lines) == 1
        assert "100_1000" in utt_lines[0]


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------

class TestBuildChatEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_utterances(self) -> None:
        """No utterances produces valid CHAT with only headers."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "@UTF8" in result
        assert "@End" in result

    def test_default_language(self) -> None:
        """Missing langs defaults to eng."""
        transcript = {
            "participants": [
                {"id": "PAR0", "name": "P", "role": "Participant"}
            ],
            "utterances": [],
        }

        result = batchalign_core.build_chat(json.dumps(transcript))
        assert "eng" in result

    def test_invalid_json_raises(self) -> None:
        """Invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            batchalign_core.build_chat("not json")

    def test_missing_participants_raises(self) -> None:
        """Missing participants raises ValueError."""
        with pytest.raises(ValueError, match="participants"):
            batchalign_core.build_chat(json.dumps({"utterances": []}))

    def test_round_trip_parseable(self) -> None:
        """Output from build_chat can be parsed back by parse_and_serialize."""
        transcript = {
            "langs": ["eng"],
            "participants": [
                {"id": "PAR0", "name": "Participant", "role": "Participant"}
            ],
            "utterances": [
                {
                    "speaker": "PAR0",
                    "words": [
                        {"text": "hello", "start_ms": 100, "end_ms": 500},
                        {"text": ".", "start_ms": None, "end_ms": None},
                    ],
                }
            ],
        }

        chat_text = batchalign_core.build_chat(json.dumps(transcript))
        # Should be parseable by the Rust parser
        reparsed = batchalign_core.parse_and_serialize(chat_text)
        assert "hello" in reparsed
        assert "@UTF8" in reparsed
