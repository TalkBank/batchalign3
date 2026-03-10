"""Tests for WER benchmark computation (Rust-backed).

Exercises the Rust wer_compute function via Python wrapper.
"""

from __future__ import annotations

import pytest

batchalign_core = pytest.importorskip("batchalign_core")

from batchalign.inference.benchmark import compute_wer, compute_wer_from_words


class TestComputeWerFromWords:
    """Tests for the Rust-backed compute_wer_from_words."""

    def test_identical_words(self) -> None:
        wer, diff = compute_wer_from_words(["hello", "world"], ["hello", "world"])
        assert wer == 0.0

    def test_empty_reference(self) -> None:
        wer, diff = compute_wer_from_words(["hello"], [])
        assert wer == 0.0

    def test_empty_hypothesis(self) -> None:
        wer, diff = compute_wer_from_words([], ["hello", "world"])
        assert wer == 1.0

    def test_dash_removal(self) -> None:
        wer, _ = compute_wer_from_words(["ice-cream"], ["icecream"])
        assert wer == 0.0

    def test_case_insensitive(self) -> None:
        wer, _ = compute_wer_from_words(["Hello", "WORLD"], ["hello", "world"])
        assert wer == 0.0

    def test_paren_removal(self) -> None:
        wer, _ = compute_wer_from_words(["(hello)", "world"], ["hello", "world"])
        assert wer == 0.0

    def test_single_letter_combining(self) -> None:
        wer, _ = compute_wer_from_words(["a", "b", "c"], ["abc"])
        assert wer == 0.0

    def test_chinese_decomposition(self) -> None:
        wer, _ = compute_wer_from_words(["你好"], ["你好"], langs=["zho"])
        assert wer == 0.0

    def test_diff_string_nonempty_on_mismatch(self) -> None:
        wer, diff = compute_wer_from_words(["hello"], ["goodbye"])
        assert wer > 0.0
        assert len(diff) > 0


class TestComputeWer:
    """Tests for the high-level compute_wer wrapper."""

    def test_perfect_match(self) -> None:
        result = compute_wer(["the", "dog", "runs"], ["the", "dog", "runs"])
        assert result["wer"] == 0.0
        assert result["accuracy"] == 1.0
        assert result["total"] == 3
        assert result["matches"] == 3

    def test_total_mismatch(self) -> None:
        result = compute_wer(["cat"], ["the", "big", "dog"])
        assert result["wer"] > 0.0
        assert result["total"] == 3

    def test_empty_reference_result(self) -> None:
        result = compute_wer(["hello"], [])
        assert result["total"] == 0
