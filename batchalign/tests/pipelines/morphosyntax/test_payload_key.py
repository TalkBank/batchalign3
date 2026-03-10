"""Tests for the 'key' field in extract_morphosyntax_payloads output.

The key is BLAKE3( "{words}|{lang}|mwt" ) — computed in Rust.
"""

from __future__ import annotations

import json

import pytest

batchalign_core = pytest.importorskip("batchalign_core")

ParsedChat = batchalign_core.ParsedChat

# ---------------------------------------------------------------------------
# Minimal CHAT with one English utterance
# ---------------------------------------------------------------------------

_CHAT = (
    "@UTF8\n"
    "@Begin\n"
    "@Languages:\teng\n"
    "@Participants:\tPAR Participant\n"
    "@ID:\teng|test|PAR|||||Participant|||\n"
    "*PAR:\thello world .\n"
    "@End"
)

_CHAT_TWO_UTTERANCES = (
    "@UTF8\n"
    "@Begin\n"
    "@Languages:\teng\n"
    "@Participants:\tPAR Participant\n"
    "@ID:\teng|test|PAR|||||Participant|||\n"
    "*PAR:\thello world .\n"
    "*PAR:\tgoodbye now .\n"
    "@End"
)


class TestMorphosyntaxPayloadKey:
    """Verify that extract_morphosyntax_payloads() includes a 'key' field."""

    def test_payload_has_key_field(self) -> None:
        handle = ParsedChat.parse(_CHAT)
        raw = handle.extract_morphosyntax_payloads("eng")  # type: ignore[attr-defined]
        payloads = json.loads(raw)
        assert len(payloads) >= 1
        assert "key" in payloads[0], (
            "MorphosyntaxPayloadJson must have a 'key' field"
        )

    def test_key_is_hex_string(self) -> None:
        handle = ParsedChat.parse(_CHAT)
        raw = handle.extract_morphosyntax_payloads("eng")  # type: ignore[attr-defined]
        payloads = json.loads(raw)
        key = payloads[0]["key"]
        # BLAKE3 hex digest is 64 lowercase hex characters
        assert isinstance(key, str)
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_key_matches_blake3_formula(self) -> None:
        """Rust key must equal BLAKE3('{words}|{lang}|mwt') for cache compat."""
        try:
            import blake3 as b3
        except ImportError:
            pytest.skip("blake3 not installed")

        handle = ParsedChat.parse(_CHAT)
        raw = handle.extract_morphosyntax_payloads("eng")  # type: ignore[attr-defined]
        payloads = json.loads(raw)
        p = payloads[0]
        combined = f"{' '.join(p['words'])}|{p['lang']}|mwt"
        expected = b3.blake3(combined.encode()).hexdigest()
        assert p["key"] == expected, (
            f"Key mismatch: Rust={p['key']!r}, Python={expected!r}"
        )

    def test_each_utterance_has_distinct_key(self) -> None:
        handle = ParsedChat.parse(_CHAT_TWO_UTTERANCES)
        raw = handle.extract_morphosyntax_payloads("eng")  # type: ignore[attr-defined]
        payloads = json.loads(raw)
        keys = [p["key"] for p in payloads]
        assert len(keys) == len(set(keys)), "Each utterance must have a unique key"

    def test_key_stable_across_calls(self) -> None:
        """Same input → same key (deterministic hash)."""
        handle1 = ParsedChat.parse(_CHAT)
        handle2 = ParsedChat.parse(_CHAT)
        key1 = json.loads(handle1.extract_morphosyntax_payloads("eng"))[0]["key"]  # type: ignore[attr-defined]
        key2 = json.loads(handle2.extract_morphosyntax_payloads("eng"))[0]["key"]  # type: ignore[attr-defined]
        assert key1 == key2
