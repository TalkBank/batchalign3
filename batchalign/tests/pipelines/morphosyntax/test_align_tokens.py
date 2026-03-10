"""Tests for batchalign_core.align_tokens() — Rust tokenizer realignment.

TDD: written before the Rust function exists.

align_tokens(original_words, stanza_tokens, alpha2) aligns Stanza tokenizer
output to original CHAT words.  When Stanza re-splits a word (e.g.
"ice-cream" → ["ice", "-", "cream"]), the function merges those tokens back
into the original word.

Return value: list where each element is either:
  - A plain str for 1:1 token↔word mappings (passed through unchanged)
  - A (str, True) tuple for English contractions (MWT expansion allowed)
  - A (str, False) tuple for other merges (MWT expansion suppressed)

The semantics match _realign_sentence() in _tokenizer_realign.py exactly,
verified by the cross-check tests below.
"""

from __future__ import annotations
from typing import Any

import pytest

batchalign_core = pytest.importorskip("batchalign_core")

align_tokens = batchalign_core.align_tokens  # type: ignore[attr-defined]

# Also import Python reference for cross-checking
from batchalign.inference._tokenizer_realign import (
    _realign_sentence,
    _conform,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conform_result(item: Any) -> Any:
    """Normalize output: str stays str, tuple stays (str, bool)."""
    return item


# ---------------------------------------------------------------------------
# Basic pass-through (1:1 mapping)
# ---------------------------------------------------------------------------

class TestPassThrough:
    def test_single_word_unchanged(self) -> None:
        result = align_tokens(["hello"], ["hello"], "en")
        assert result == ["hello"]

    def test_two_words_unchanged(self) -> None:
        result = align_tokens(["hello", "world"], ["hello", "world"], "en")
        assert result == ["hello", "world"]

    def test_empty_words_returns_empty(self) -> None:
        result = align_tokens([], [], "en")
        assert result == []

    def test_returns_list(self) -> None:
        result = align_tokens(["hi"], ["hi"], "en")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Spurious splits → (text, False)
# ---------------------------------------------------------------------------

class TestSpuriousSplitMerge:
    def test_hyphenated_word_merged_as_false(self) -> None:
        # Stanza splits "ice-cream" → ["ice", "-", "cream"]
        result = align_tokens(["ice-cream"], ["ice", "-", "cream"], "en")
        assert len(result) == 1
        assert result[0] == ("ice-cream", False)

    def test_spurious_split_non_english_also_false(self) -> None:
        result = align_tokens(["l'eau"], ["l", "'eau"], "fr")
        assert len(result) == 1
        assert result[0] == ("l'eau", False)

    def test_mixed_split_and_single(self) -> None:
        result = align_tokens(["ice-cream", "good", "bye"], ["ice", "-", "cream", "good", "bye"], "en")
        assert result[0] == ("ice-cream", False)
        assert result[1] == "good"
        assert result[2] == "bye"


# ---------------------------------------------------------------------------
# English contractions → (text, True)
# ---------------------------------------------------------------------------

class TestEnglishContractions:
    def test_dont_contraction_true(self) -> None:
        result = align_tokens(["don't"], ["don", "'t"], "en")
        assert result == [("don't", True)]

    def test_possessive_contraction_true(self) -> None:
        result = align_tokens(["Claus'"], ["Claus", "'"], "en")
        assert result == [("Claus'", True)]

    def test_its_contraction_true(self) -> None:
        result = align_tokens(["it's"], ["it", "'s"], "en")
        assert result == [("it's", True)]


# ---------------------------------------------------------------------------
# o' exclusion
# ---------------------------------------------------------------------------

class TestOClock:
    def test_oclock_is_false(self) -> None:
        # "o'clock" split → ["o", "'clock"] — o' prefix excluded
        result = align_tokens(["o'clock"], ["o", "'clock"], "en")
        assert result == [("o'clock", False)]


# ---------------------------------------------------------------------------
# Character mismatch → return tokens unchanged
# ---------------------------------------------------------------------------

class TestCharMismatch:
    def test_mismatch_returns_stanza_tokens(self) -> None:
        result = align_tokens(["something", "else"], ["totally", "different"], "en")
        assert result == ["totally", "different"]


# ---------------------------------------------------------------------------
# Cross-check: Rust result matches Python _realign_sentence()
# ---------------------------------------------------------------------------

class TestCrossCheckPython:
    """Verify Rust implementation matches Python reference for common cases."""

    def _check(self, original: list[str], stanza: list[str], alpha2: str) -> None:
        rust_result = align_tokens(original, stanza, alpha2)
        py_result = _realign_sentence(stanza, original, alpha2)
        # Normalize: convert plain tokens to strings for comparison
        rust_normalized = [
            (t[0], t[1]) if isinstance(t, tuple) else str(t)
            for t in rust_result
        ]
        py_normalized = [
            (t[0], t[1]) if isinstance(t, tuple) else _conform(t)
            for t in py_result
        ]
        assert rust_normalized == py_normalized, (
            f"Rust={rust_normalized!r} != Python={py_normalized!r}"
        )

    def test_pass_through(self) -> None:
        self._check(["hello", "world"], ["hello", "world"], "en")

    def test_hyphen_split(self) -> None:
        self._check(["ice-cream"], ["ice", "-", "cream"], "en")

    def test_english_contraction(self) -> None:
        self._check(["don't"], ["don", "'t"], "en")

    def test_possessive(self) -> None:
        self._check(["John's"], ["John", "'s"], "en")

    def test_french_no_contraction(self) -> None:
        self._check(["l'eau"], ["l", "'eau"], "fr")

    def test_mixed_batch(self) -> None:
        self._check(
            ["ice-cream", "don't", "know"],
            ["ice", "-", "cream", "don", "'t", "know"],
            "en",
        )
