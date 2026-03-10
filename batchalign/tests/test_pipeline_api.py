"""Tests for the thin Python facade over the Rust-owned CHAT pipeline."""

from __future__ import annotations

from typing import Any

import pytest

from batchalign.pipeline_api import (
    BatchInferProviderInvoker,
    LocalProviderInvoker,
    PipelineOperation,
    run_pipeline,
    unwrap_batch_results,
)
from batchalign.providers import BatchInferResponse, InferResponse

pytest.importorskip("batchalign_core")

_TRANSLATE_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies .
@End
"""

_MOR_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
*CHI:\tgoodbye moon .
@End
"""

_FA_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
@Media:\ttest, audio
*CHI:\thello world . \x150_5000\x15
@End
"""

_UTR_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog is big .
@End
"""

_UTSEG_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tI eat cookies and he likes cake .
@End
"""

_COREF_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog ran .
*CHI:\tit was fast .
@End
"""


def _raw_sentences(words: list[str]) -> dict[str, Any]:
    """Build one raw-Stanza-like response for morphosyntax tests."""
    rows = []
    for i, word in enumerate(words, start=1):
        rows.append(
            {
                "id": i,
                "text": word,
                "lemma": word,
                "upos": "INTJ" if i == 1 else "NOUN",
                "head": 0 if i == 1 else 1,
                "deprel": "root" if i == 1 else "obj",
            }
        )
    return {"raw_sentences": [rows]}


class TestPipelineOperation:
    """Verify the thin operation record stays simple and explicit."""

    def test_rejects_name_override_in_options(self) -> None:
        with pytest.raises(ValueError, match="must not contain 'name'"):
            PipelineOperation("translate", {"name": "fa"})

    def test_serializes_to_rust_wire_shape(self) -> None:
        operation = PipelineOperation("morphosyntax", {"retokenize": True})
        assert operation.to_wire() == {"name": "morphosyntax", "retokenize": True}


class TestRunPipelineWithLocalProviders:
    """Verify the Python facade delegates CHAT semantics to Rust."""

    def test_runs_translation(self) -> None:
        seen_batches: list[list[dict[str, Any]]] = []

        def _translate(lang: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            assert lang == "eng"
            seen_batches.append(items)
            return [{"translation": items[0]["text"].upper()}]

        output = run_pipeline(
            _TRANSLATE_CHAT,
            lang="eng",
            provider=LocalProviderInvoker({"translate": _translate}),
            operations=[PipelineOperation("translate")],
        )

        assert seen_batches == [[{"text": "I eat cookies", "speaker": "CHI"}]]
        assert "%xtra:" in output
        assert "I EAT COOKIES" in output

    def test_runs_morphosyntax(self) -> None:
        seen_batches: list[list[dict[str, Any]]] = []

        def _morphosyntax(
            lang: str,
            items: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            assert lang == "eng"
            seen_batches.append(items)
            return [_raw_sentences(item["words"]) for item in items]

        output = run_pipeline(
            _MOR_CHAT,
            lang="eng",
            provider=LocalProviderInvoker({"morphosyntax": _morphosyntax}),
            operations=[PipelineOperation("morphosyntax")],
        )

        assert len(seen_batches) == 1
        assert seen_batches[0][0]["words"] == ["hello", "world"]
        assert seen_batches[0][1]["words"] == ["goodbye", "moon"]
        assert output.count("%mor:") == 2
        assert output.count("%gra:") == 2

    def test_runs_forced_alignment(self) -> None:
        seen_batches: list[list[dict[str, Any]]] = []

        def _fa(lang: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            assert lang == "eng"
            seen_batches.append(items)
            return [
                {
                    "indexed_timings": [
                        {"start_ms": 100, "end_ms": 2000},
                        {"start_ms": 2000, "end_ms": 4500},
                    ]
                }
            ]

        output = run_pipeline(
            _FA_CHAT,
            lang="eng",
            provider=LocalProviderInvoker({"fa": _fa}),
            operations=[PipelineOperation("fa")],
        )

        assert seen_batches[0][0]["words"] == ["hello", "world"]
        assert "%wor:" in output
        assert "\x15" in output

    def test_runs_utterance_segmentation(self) -> None:
        seen_batches: list[list[dict[str, Any]]] = []

        def _utseg(lang: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            assert lang == "eng"
            seen_batches.append(items)
            return [{"assignments": [0, 0, 0, 1, 1, 1, 1]}]

        output = run_pipeline(
            _UTSEG_CHAT,
            lang="eng",
            provider=LocalProviderInvoker({"utseg": _utseg}),
            operations=[PipelineOperation("utseg")],
        )

        utt_lines = [line for line in output.splitlines() if line.startswith("*CHI:")]
        assert len(seen_batches) == 1
        assert seen_batches[0][0]["words"] == [
            "I",
            "eat",
            "cookies",
            "and",
            "he",
            "likes",
            "cake",
        ]
        assert len(utt_lines) == 2

    def test_coref_operation_is_not_yet_supported(self) -> None:
        seen_batches: list[list[dict[str, Any]]] = []

        def _coref(lang: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            assert lang == "eng"
            seen_batches.append(items)
            return [
                {
                    "annotations": [
                        {
                            "sentence_idx": 0,
                            "words": [
                                [{"chain_id": 0, "is_start": True, "is_end": True}],
                                [],
                                [],
                            ],
                        },
                        {
                            "sentence_idx": 1,
                            "words": [
                                [{"chain_id": 0, "is_start": True, "is_end": True}],
                                [],
                                [],
                            ],
                        },
                    ]
                }
            ]

        with pytest.raises(ValueError, match="unsupported pipeline operation: coref"):
            run_pipeline(
                _COREF_CHAT,
                lang="eng",
                provider=LocalProviderInvoker({"coref": _coref}),
                operations=[PipelineOperation("coref")],
            )

        assert seen_batches == []

    def test_runs_utterance_timing_without_provider(self) -> None:
        output = run_pipeline(
            _UTR_CHAT,
            lang="eng",
            operations=[
                PipelineOperation(
                    "utr",
                    {
                        "timed_words": [
                            {
                                "word": "the",
                                "start_ms": 100,
                                "end_ms": 200,
                                "word_id": "u0:w0",
                            },
                            {
                                "word": "dog",
                                "start_ms": 250,
                                "end_ms": 400,
                                "word_id": "u0:w1",
                            },
                            {
                                "word": "is",
                                "start_ms": 450,
                                "end_ms": 500,
                                "word_id": "u0:w2",
                            },
                            {
                                "word": "big",
                                "start_ms": 550,
                                "end_ms": 700,
                                "word_id": "u0:w3",
                            },
                        ]
                    },
                )
            ],
        )

        assert "\x15" in output

    def test_runs_multiple_operations_over_one_document(self) -> None:
        def _translate(lang: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            assert lang == "eng"
            return [{"translation": items[0]["text"].upper()}]

        def _morphosyntax(
            lang: str,
            items: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            assert lang == "eng"
            return [_raw_sentences(item["words"]) for item in items]

        output = run_pipeline(
            _MOR_CHAT,
            lang="eng",
            provider=LocalProviderInvoker(
                {
                    "translate": _translate,
                    "morphosyntax": _morphosyntax,
                }
            ),
            operations=[
                PipelineOperation("translate"),
                PipelineOperation("morphosyntax"),
            ],
        )

        assert "%xtra:" in output
        assert output.count("%mor:") == 2
        assert output.count("%gra:") == 2


class TestBatchInferProviderInvoker:
    """Verify the worker-style provider adapter integrates with the Rust loop."""

    def test_runs_pipeline_via_batch_infer_host_contract(self) -> None:
        seen_requests: list[tuple[str, str, list[dict[str, Any]]]] = []

        def _host(req: Any) -> BatchInferResponse:
            seen_requests.append((req.task.value, req.lang, list(req.items)))
            if req.task.value == "translate":
                return BatchInferResponse(
                    results=[
                        InferResponse(
                            result={"translation": req.items[0]["text"].upper()},
                            elapsed_s=0.0,
                        )
                    ]
                )
            if req.task.value == "morphosyntax":
                return BatchInferResponse(
                    results=[
                        InferResponse(
                            result=_raw_sentences(item["words"]),
                            elapsed_s=0.0,
                        )
                        for item in req.items
                    ]
                )
            raise AssertionError(f"unexpected task {req.task}")

        output = run_pipeline(
            _MOR_CHAT,
            lang="eng",
            provider=BatchInferProviderInvoker(infer=_host),
            operations=[
                PipelineOperation("translate"),
                PipelineOperation("morphosyntax"),
            ],
        )

        assert [task for task, _lang, _items in seen_requests] == [
            "translate",
            "translate",
            "morphosyntax",
        ]
        assert "%xtra:" in output
        assert output.count("%mor:") == 2
        assert output.count("%gra:") == 2

    def test_runs_utseg_via_batch_infer_host_contract(self) -> None:
        seen_requests: list[tuple[str, str, list[dict[str, Any]]]] = []

        def _host(req: Any) -> BatchInferResponse:
            seen_requests.append((req.task.value, req.lang, list(req.items)))
            assert req.task.value == "utseg"
            return BatchInferResponse(
                results=[
                    InferResponse(
                        result={"assignments": [0, 0, 0, 1, 1, 1, 1]},
                        elapsed_s=0.0,
                    )
                ]
            )

        output = run_pipeline(
            _UTSEG_CHAT,
            lang="eng",
            provider=BatchInferProviderInvoker(infer=_host),
            operations=[PipelineOperation("utseg")],
        )

        utt_lines = [line for line in output.splitlines() if line.startswith("*CHI:")]
        assert seen_requests == [
            (
                "utseg",
                "eng",
                [
                    {
                        "words": [
                            "I",
                            "eat",
                            "cookies",
                            "and",
                            "he",
                            "likes",
                            "cake",
                        ],
                        "text": "I eat cookies and he likes cake",
                    }
                ],
            )
        ]
        assert len(utt_lines) == 2

    def test_coref_batch_infer_operation_is_not_yet_supported(self) -> None:
        seen_requests: list[tuple[str, str, list[dict[str, Any]]]] = []

        def _host(req: Any) -> BatchInferResponse:
            seen_requests.append((req.task.value, req.lang, list(req.items)))
            assert req.task.value == "coref"
            return BatchInferResponse(
                results=[
                    InferResponse(
                        result={
                            "annotations": [
                                {
                                    "sentence_idx": 0,
                                    "words": [
                                        [{"chain_id": 0, "is_start": True, "is_end": True}],
                                        [],
                                        [],
                                    ],
                                },
                                {
                                    "sentence_idx": 1,
                                    "words": [
                                        [{"chain_id": 0, "is_start": True, "is_end": True}],
                                        [],
                                        [],
                                    ],
                                },
                            ]
                        },
                        elapsed_s=0.0,
                    )
                ]
            )

        with pytest.raises(ValueError, match="unsupported pipeline operation: coref"):
            run_pipeline(
                _COREF_CHAT,
                lang="eng",
                provider=BatchInferProviderInvoker(infer=_host),
                operations=[PipelineOperation("coref")],
            )

        assert seen_requests == []

    def test_unwrap_batch_results_raises_on_provider_errors(self) -> None:
        response = BatchInferResponse(
            results=[InferResponse(error="boom", elapsed_s=0.0)]
        )

        with pytest.raises(RuntimeError, match="translate provider batch failed"):
            unwrap_batch_results("translate", response)

    def test_unwrap_batch_results_rejects_non_object_results(self) -> None:
        response = BatchInferResponse(
            results=[InferResponse(result=["not", "an", "object"], elapsed_s=0.0)]
        )

        with pytest.raises(TypeError, match="translate provider returned a non-object result at index 0"):
            unwrap_batch_results("translate", response)
