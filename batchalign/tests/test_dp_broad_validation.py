"""Broad DP-migration validation matrix for UTR timing transfer.

This test extends fixture-level goldens into a larger perturbation matrix:
multiple fixtures x multiple ASR-order/ID variants. It compares:

- Baseline: transcript-level global DP (`batchalign_core.dp_align`)
- Redesigned: `ParsedChat.add_utterance_timing` deterministic mapping path

The goal is to keep a high-signal quality gate as we remove avoidable DP.
"""

from __future__ import annotations

import json
import os
import random
from glob import glob
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

batchalign_core = pytest.importorskip("batchalign_core")

ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = ROOT / "batchalign" / "tests" / "golden" / "fixtures"


def _normalize(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _has_utterance_windows(chat_text: str) -> bool:
    tiers = json.loads(batchalign_core.extract_timed_tiers(chat_text, False))
    return any(items for items in tiers.values())


def _extract_reference_words(chat_text: str) -> list[dict[str, Any]]:
    extracted = json.loads(batchalign_core.extract_nlp_words(chat_text, "wor"))
    words: list[dict[str, Any]] = []
    for utt in extracted:
        for word in utt["words"]:
            words.append(
                {
                    "speaker": utt["speaker"],
                    "utterance_index": utt["utterance_index"],
                    "word_index": word["utterance_word_index"],
                    "word_id": word["word_id"],
                    "text": word["text"],
                }
            )
    return words


def _oracle_timing_by_id(ref_words: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    oracle: dict[str, tuple[int, int]] = {}
    for idx, word in enumerate(ref_words):
        start = 100 + idx * 120
        oracle[word["word_id"]] = (start, start + 90)
    return oracle


def _build_asr_words(
    ref_words: list[dict[str, Any]],
    oracle_by_id: dict[str, tuple[int, int]],
    *,
    variant: str,
    seed: int | None,
) -> list[dict[str, Any]]:
    words = [
        {
            "word": w["text"],
            "word_id": w["word_id"],
            "start_ms": oracle_by_id[w["word_id"]][0],
            "end_ms": oracle_by_id[w["word_id"]][1],
        }
        for w in ref_words
    ]

    if variant == "ordered_with_ids":
        ordered = words
    elif variant == "reverse_with_ids":
        ordered = list(reversed(words))
    elif variant == "shuffle_with_ids":
        rng = random.Random(seed)
        ordered = words[:]
        rng.shuffle(ordered)
    elif variant == "reverse_no_ids":
        ordered = list(reversed(words))
        for w in ordered:
            w.pop("word_id", None)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return ordered


def _baseline_predict(
    ref_words: list[dict[str, Any]], asr_words: list[dict[str, Any]]
) -> list[tuple[int, int] | None]:
    payload = [w["word"].casefold() for w in asr_words]
    reference = [w["text"].casefold() for w in ref_words]
    alignment = json.loads(batchalign_core.dp_align(payload, reference, True))

    pred: list[tuple[int, int] | None] = [None] * len(ref_words)
    for item in alignment:
        if item.get("type") == "match":
            payload_idx = item["payload_idx"]
            reference_idx = item["reference_idx"]
            pred[reference_idx] = (
                int(asr_words[payload_idx]["start_ms"]),
                int(asr_words[payload_idx]["end_ms"]),
            )
    return pred


def _redesigned_predict(
    chat_text: str, ref_words: list[dict[str, Any]], asr_words: list[dict[str, Any]]
) -> list[tuple[int, int] | None]:
    handle = batchalign_core.ParsedChat.parse(chat_text)
    handle.add_utterance_timing(json.dumps(asr_words))
    output = handle.serialize()
    timed = json.loads(batchalign_core.extract_timed_tiers(output, True))

    speaker_ref: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, word in enumerate(ref_words):
        speaker_ref[word["speaker"]].append((idx, _normalize(word["text"])))

    pred: list[tuple[int, int] | None] = [None] * len(ref_words)
    for speaker, timed_words in timed.items():
        ref_seq = speaker_ref.get(speaker, [])
        cursor = 0
        for tw in timed_words:
            tnorm = _normalize(tw.get("text", ""))
            if not tnorm:
                continue
            while cursor < len(ref_seq) and ref_seq[cursor][1] != tnorm:
                cursor += 1
            if cursor >= len(ref_seq):
                break
            ref_idx = ref_seq[cursor][0]
            pred[ref_idx] = (int(tw["start_ms"]), int(tw["end_ms"]))
            cursor += 1
    return pred


def _metrics(
    ref_words: list[dict[str, Any]],
    oracle_by_id: dict[str, tuple[int, int]],
    pred: list[tuple[int, int] | None],
) -> dict[str, float | int | None]:
    total_words = len(ref_words)
    timed_words = sum(1 for p in pred if p is not None)
    timed_cov = timed_words / total_words if total_words else 1.0

    word_diffs = []
    for idx, p in enumerate(pred):
        if p is None:
            continue
        os, oe = oracle_by_id[ref_words[idx]["word_id"]]
        ps, pe = p
        word_diffs.append((abs(ps - os) + abs(pe - oe)) / 2.0)
    word_mae = (sum(word_diffs) / len(word_diffs)) if word_diffs else None

    utt_ids = sorted({w["utterance_index"] for w in ref_words})
    pred_by_utt: dict[int, list[tuple[int, int]]] = defaultdict(list)
    oracle_by_utt: dict[int, list[tuple[int, int]]] = defaultdict(list)

    for idx, w in enumerate(ref_words):
        utt = w["utterance_index"]
        oracle_by_utt[utt].append(oracle_by_id[w["word_id"]])
        if pred[idx] is not None:
            pred_by_utt[utt].append(pred[idx])  # type: ignore[arg-type]

    covered = 0
    boundary_diffs = []
    for utt in utt_ids:
        if not pred_by_utt[utt]:
            continue
        covered += 1
        p_start = min(s for s, _ in pred_by_utt[utt])
        p_end = max(e for _, e in pred_by_utt[utt])
        o_start = min(s for s, _ in oracle_by_utt[utt])
        o_end = max(e for _, e in oracle_by_utt[utt])
        boundary_diffs.append((abs(p_start - o_start) + abs(p_end - o_end)) / 2.0)

    boundary_cov = covered / len(utt_ids) if utt_ids else 1.0
    boundary_mae = (sum(boundary_diffs) / len(boundary_diffs)) if boundary_diffs else None

    return {
        "timed_word_coverage": timed_cov,
        "boundary_coverage": boundary_cov,
        "word_l1_mae_ms": word_mae,
        "boundary_l1_mae_ms": boundary_mae,
        "total_words": total_words,
        "total_utterances": len(utt_ids),
    }


def _aggregate(rows: list[dict[str, float | int | None]]) -> dict[str, float | int | None]:
    total_words = sum(int(r["total_words"]) for r in rows)
    total_utterances = sum(int(r["total_utterances"]) for r in rows)
    timed_words = sum(float(r["timed_word_coverage"]) * int(r["total_words"]) for r in rows)
    covered_utterances = sum(
        float(r["boundary_coverage"]) * int(r["total_utterances"]) for r in rows
    )

    def weighted_mae(key: str, weight_key: str) -> float | None:
        num = 0.0
        den = 0
        for r in rows:
            value = r[key]
            if value is None:
                continue
            weight = int(r[weight_key])
            num += float(value) * weight
            den += weight
        return (num / den) if den else None

    return {
        "timed_word_coverage": (timed_words / total_words) if total_words else 1.0,
        "boundary_coverage": (covered_utterances / total_utterances) if total_utterances else 1.0,
        "word_l1_mae_ms": weighted_mae("word_l1_mae_ms", "total_words"),
        "boundary_l1_mae_ms": weighted_mae("boundary_l1_mae_ms", "total_utterances"),
        "total_words": total_words,
        "total_utterances": total_utterances,
    }


def _run_matrix(corpus_paths: list[Path]) -> dict[str, Any]:
    cases = []
    variants = [
        ("ordered_with_ids", None),
        ("reverse_with_ids", None),
        ("reverse_no_ids", None),
        *[("shuffle_with_ids", seed) for seed in range(10)],
    ]

    for path in corpus_paths:
        chat_text = path.read_text()
        ref_words = _extract_reference_words(chat_text)
        if len(ref_words) < 2:
            continue
        oracle_by_id = _oracle_timing_by_id(ref_words)
        has_windows = _has_utterance_windows(chat_text)

        for variant, seed in variants:
            asr_words = _build_asr_words(
                ref_words, oracle_by_id, variant=variant, seed=seed
            )
            baseline = _baseline_predict(ref_words, asr_words)
            redesigned = _redesigned_predict(chat_text, ref_words, asr_words)

            case_name = f"{path.stem}:{variant}" + (f":seed{seed}" if seed is not None else "")
            cases.append(
                {
                    "name": case_name,
                    "variant": variant,
                    "has_windows": has_windows,
                    "baseline": _metrics(ref_words, oracle_by_id, baseline),
                    "redesigned": _metrics(ref_words, oracle_by_id, redesigned),
                }
            )

    return {
        "per_case": cases,
        "overall": {
            "baseline": _aggregate([c["baseline"] for c in cases]),
            "redesigned": _aggregate([c["redesigned"] for c in cases]),
        },
    }


def test_utr_redesign_broad_matrix_improves_quality() -> None:
    corpus_paths = sorted(FIXTURES_DIR.glob("dp_utr*_input.cha"))
    assert corpus_paths, "expected dp_utr fixtures for broad validation matrix"

    result = _run_matrix(corpus_paths)
    overall_base = result["overall"]["baseline"]
    overall_new = result["overall"]["redesigned"]

    assert overall_new["timed_word_coverage"] > overall_base["timed_word_coverage"]
    assert overall_new["boundary_coverage"] > overall_base["boundary_coverage"]
    assert overall_new["word_l1_mae_ms"] < overall_base["word_l1_mae_ms"]
    assert overall_new["boundary_l1_mae_ms"] < overall_base["boundary_l1_mae_ms"]

    # A stronger floor to keep this test a true migration gate.
    assert overall_new["timed_word_coverage"] >= 0.95
    assert overall_new["boundary_coverage"] >= 0.97


def test_utr_redesign_external_corpus_gate_when_configured() -> None:
    corpus_glob = os.environ.get("BATCHALIGN_DP_VALIDATION_GLOB")
    if not corpus_glob:
        pytest.skip("set BATCHALIGN_DP_VALIDATION_GLOB to enable external corpus validation")

    corpus_paths = sorted(Path(p) for p in glob(corpus_glob, recursive=True))
    if not corpus_paths:
        pytest.skip(f"no files matched BATCHALIGN_DP_VALIDATION_GLOB={corpus_glob!r}")

    max_files = int(os.environ.get("BATCHALIGN_DP_VALIDATION_MAX_FILES", "100"))
    corpus_paths = corpus_paths[:max_files]
    result = _run_matrix(corpus_paths)
    overall_base = result["overall"]["baseline"]
    overall_new = result["overall"]["redesigned"]

    assert overall_new["timed_word_coverage"] >= overall_base["timed_word_coverage"]
    assert overall_new["boundary_coverage"] >= overall_base["boundary_coverage"]
    assert overall_new["word_l1_mae_ms"] <= overall_base["word_l1_mae_ms"]
    assert overall_new["boundary_l1_mae_ms"] <= overall_base["boundary_l1_mae_ms"]
