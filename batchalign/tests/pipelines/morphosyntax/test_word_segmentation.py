"""Tests for PyCantonese word segmentation and retokenize wire plumbing."""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Wire type tests
# ---------------------------------------------------------------------------


def test_morphosyntax_request_v2_accepts_retokenize():
    """MorphosyntaxRequestV2 must accept and round-trip the retokenize field."""
    from batchalign.worker._types_v2 import MorphosyntaxRequestV2

    req = MorphosyntaxRequestV2(
        kind="morphosyntax", lang="yue", payload_ref_id="p1", item_count=1, retokenize=True
    )
    assert req.retokenize is True


def test_morphosyntax_request_v2_retokenize_defaults_false():
    """Backward compat: retokenize defaults to False."""
    from batchalign.worker._types_v2 import MorphosyntaxRequestV2

    req = MorphosyntaxRequestV2(
        kind="morphosyntax", lang="eng", payload_ref_id="p1", item_count=1
    )
    assert req.retokenize is False


def test_batch_infer_request_accepts_retokenize():
    """BatchInferRequest must accept the retokenize field."""
    from batchalign.worker._types import BatchInferRequest, InferTask

    req = BatchInferRequest(task=InferTask.MORPHOSYNTAX, lang="eng", items=[], retokenize=True)
    assert req.retokenize is True


# ---------------------------------------------------------------------------
# PyCantonese segmentation tests
# ---------------------------------------------------------------------------


def test_segment_cantonese_basic():
    """Characters should be grouped into words by PyCantonese."""
    from batchalign.inference.morphosyntax import _segment_cantonese

    result = _segment_cantonese(["故", "事", "係", "好"])
    # PyCantonese should group some characters into words
    assert len(result) <= 4  # At minimum shouldn't get more than input
    assert "".join(result) == "故事係好"  # All characters preserved


def test_segment_cantonese_empty():
    """Empty input returns empty output."""
    from batchalign.inference.morphosyntax import _segment_cantonese

    assert _segment_cantonese([]) == []


def test_segment_cantonese_single_char():
    """Single character input passes through."""
    from batchalign.inference.morphosyntax import _segment_cantonese

    result = _segment_cantonese(["好"])
    assert result == ["好"]
