"""Golden tests for DP-dependent behavior in batchalign_core."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from batchalign.worker._types import WorkerJSONValue

batchalign_core = pytest.importorskip("batchalign_core")

# Alias for FA callback dict types (payload and response).
FaDict = dict[str, WorkerJSONValue]


def _compare_or_update(
    output: str,
    expected_path: Path,
    update_golden: bool,
    label: str,
) -> None:
    if update_golden:
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text(output)
        pytest.skip(f"Updated golden expectation: {expected_path.name}")

    assert expected_path.exists(), (
        f"Golden expectation missing: {expected_path}\n"
        f"Run with --update-golden to generate."
    )
    expected = expected_path.read_text()
    assert output == expected, (
        f"Golden mismatch for {label}.\n"
        f"Run with --update-golden to regenerate.\n"
        f"Expected file: {expected_path}"
    )


@pytest.mark.golden
def test_dp_align_json_golden(golden_dir: Path, update_golden: bool) -> None:
    raw = batchalign_core.dp_align(
        ["i", "need", "cookie"],
        ["i", "want", "cookie"],
        True,
    )
    canonical = json.dumps(json.loads(raw), indent=2, sort_keys=True) + "\n"
    _compare_or_update(
        canonical,
        golden_dir / "expected" / "dp_align_json.expected",
        update_golden,
        "dp_align_json",
    )


@pytest.mark.golden
def test_add_utterance_timing_golden(golden_dir: Path, update_golden: bool) -> None:
    input_text = (golden_dir / "fixtures" / "dp_utr_input.cha").read_text()
    handle = batchalign_core.ParsedChat.parse(input_text)
    asr_words = [
        {"word": "the", "start_ms": 100, "end_ms": 200},
        {"word": "dog", "start_ms": 250, "end_ms": 400},
        {"word": "is", "start_ms": 450, "end_ms": 500},
        {"word": "big", "start_ms": 550, "end_ms": 700},
    ]
    handle.add_utterance_timing(json.dumps(asr_words))
    output = handle.serialize()
    _compare_or_update(
        output,
        golden_dir / "expected" / "dp_add_utterance_timing.expected",
        update_golden,
        "dp_add_utterance_timing",
    )


@pytest.mark.golden
def test_add_utterance_timing_with_word_ids_golden(
    golden_dir: Path, update_golden: bool
) -> None:
    input_text = (golden_dir / "fixtures" / "dp_utr_input.cha").read_text()
    handle = batchalign_core.ParsedChat.parse(input_text)
    asr_words = [
        {"word": "big", "start_ms": 550, "end_ms": 700, "word_id": "u0:w3"},
        {"word": "is", "start_ms": 450, "end_ms": 500, "word_id": "u0:w2"},
        {"word": "dog", "start_ms": 250, "end_ms": 400, "word_id": "u0:w1"},
        {"word": "the", "start_ms": 100, "end_ms": 200, "word_id": "u0:w0"},
    ]
    handle.add_utterance_timing(json.dumps(asr_words))
    output = handle.serialize()
    _compare_or_update(
        output,
        golden_dir / "expected" / "dp_add_utterance_timing_with_word_ids.expected",
        update_golden,
        "dp_add_utterance_timing_with_word_ids",
    )


@pytest.mark.golden
def test_add_utterance_timing_with_repeated_word_ids_golden(
    golden_dir: Path, update_golden: bool
) -> None:
    input_text = (golden_dir / "fixtures" / "dp_utr_repeated_input.cha").read_text()
    handle = batchalign_core.ParsedChat.parse(input_text)
    asr_words = [
        {"word": "the", "start_ms": 300, "end_ms": 400, "word_id": "u0:w1"},
        {"word": "dog", "start_ms": 500, "end_ms": 700, "word_id": "u0:w2"},
        {"word": "the", "start_ms": 100, "end_ms": 200, "word_id": "u0:w0"},
    ]
    handle.add_utterance_timing(json.dumps(asr_words))
    output = handle.serialize()
    _compare_or_update(
        output,
        golden_dir / "expected" / "dp_add_utterance_timing_repeated_word_ids.expected",
        update_golden,
        "dp_add_utterance_timing_repeated_word_ids",
    )


@pytest.mark.golden
def test_add_utterance_timing_with_mixed_word_ids_golden(
    golden_dir: Path, update_golden: bool
) -> None:
    input_text = (golden_dir / "fixtures" / "dp_utr_input.cha").read_text()
    handle = batchalign_core.ParsedChat.parse(input_text)
    asr_words = [
        {"word": "big", "start_ms": 550, "end_ms": 700, "word_id": "u0:w3"},
        {"word": "the", "start_ms": 100, "end_ms": 200, "word_id": "u0:w0"},
        {"word": "dog", "start_ms": 250, "end_ms": 400},
        {"word": "is", "start_ms": 450, "end_ms": 500},
    ]
    handle.add_utterance_timing(json.dumps(asr_words))
    output = handle.serialize()
    _compare_or_update(
        output,
        golden_dir / "expected" / "dp_add_utterance_timing_mixed_word_ids.expected",
        update_golden,
        "dp_add_utterance_timing_mixed_word_ids",
    )


@pytest.mark.golden
def test_add_utterance_timing_window_constrained_fallback_golden(
    golden_dir: Path, update_golden: bool
) -> None:
    input_text = (golden_dir / "fixtures" / "dp_utr_window_input.cha").read_text()
    handle = batchalign_core.ParsedChat.parse(input_text)
    asr_words = [
        {"word": "gamma", "start_ms": 1200, "end_ms": 1300},
        {"word": "delta", "start_ms": 1400, "end_ms": 1500},
        {"word": "alpha", "start_ms": 100, "end_ms": 200},
        {"word": "beta", "start_ms": 300, "end_ms": 400},
    ]
    handle.add_utterance_timing(json.dumps(asr_words))
    output = handle.serialize()
    _compare_or_update(
        output,
        golden_dir / "expected" / "dp_add_utterance_timing_window_constrained.expected",
        update_golden,
        "dp_add_utterance_timing_window_constrained",
    )


@pytest.mark.golden
def test_fa_token_mapping_golden(golden_dir: Path, update_golden: bool) -> None:
    input_text = (golden_dir / "fixtures" / "dp_fa_token_input.cha").read_text()
    handle = batchalign_core.ParsedChat.parse(input_text)

    def callback(payload: FaDict) -> FaDict:
        assert payload["words"] == ["don't", "know"]
        return {
            "tokens": [
                {"text": "do", "time_s": 0.4},
                {"text": "n't", "time_s": 0.8},
                {"text": "know", "time_s": 1.2},
            ]
        }

    handle.add_forced_alignment(callback, total_audio_ms=4000)
    output = handle.serialize()
    _compare_or_update(
        output,
        golden_dir / "expected" / "dp_fa_token_mapping.expected",
        update_golden,
        "dp_fa_token_mapping",
    )


@pytest.mark.golden
def test_fa_indexed_mapping_golden(golden_dir: Path, update_golden: bool) -> None:
    input_text = (golden_dir / "fixtures" / "dp_fa_token_input.cha").read_text()
    handle = batchalign_core.ParsedChat.parse(input_text)

    def callback(payload: FaDict) -> FaDict:
        assert payload["words"] == ["don't", "know"]
        return {
            "indexed_timings": [
                {"start_ms": 400, "end_ms": 1200},
                {"start_ms": 1200, "end_ms": 1700},
            ]
        }

    handle.add_forced_alignment(callback, total_audio_ms=4000)
    output = handle.serialize()
    _compare_or_update(
        output,
        golden_dir / "expected" / "dp_fa_indexed_mapping.expected",
        update_golden,
        "dp_fa_indexed_mapping",
    )
