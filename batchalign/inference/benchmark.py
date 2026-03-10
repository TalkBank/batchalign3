"""Rust-backed WER helpers for the Python package.

This module is intentionally small. Benchmark scoring is not model inference,
so it no longer participates in the Python worker infer protocol. The only
reason it still exists at all is to provide a thin Python-facing convenience
wrapper around the Rust `batchalign_core.wer_metrics()` entry point.
"""

from __future__ import annotations

import json
from typing import cast
from typing import TypedDict


class WerResult(TypedDict):
    """Structured return payload for WER evaluation."""

    wer: float
    cer: float
    accuracy: float
    matches: int
    total: int
    error: str


def compute_wer_from_words(
    forms: list[str],
    gold_forms: list[str],
    langs: list[str] | None = None,
) -> tuple[float, str]:
    """Compute WER from pre-extracted word lists.

    Returns (wer_score, diff_string).
    Delegates entirely to the Rust implementation.
    """
    import batchalign_core

    raw = batchalign_core.wer_metrics(forms, gold_forms, langs)
    result = json.loads(raw)
    return float(result["wer"]), str(result["error"])


def compute_wer(
    hypothesis_words: list[str],
    reference_words: list[str],
) -> WerResult:
    """Compute Word Error Rate between hypothesis and reference word lists."""
    import batchalign_core

    raw = batchalign_core.wer_metrics(hypothesis_words, reference_words, None)
    return cast(WerResult, json.loads(raw))
