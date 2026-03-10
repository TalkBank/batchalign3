"""Thin worker-protocol V2 forced-alignment wrapper.

Rust now owns the prepared-artifact reads, request validation, backend dispatch,
and typed V2 response shaping for the worker FA boundary. Python stays only at
the model-host callback edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from batchalign.worker._types_v2 import ExecuteRequestV2, ExecuteResponseV2, ForcedAlignmentRequestV2

if TYPE_CHECKING:
    from collections.abc import Callable

    from batchalign.inference.types import Wave2VecFAHandle, WhisperFAHandle


class PreparedFaPayloadV2(BaseModel):
    """Prepared FA payload written by the Rust-side V2 request builder."""

    words: list[str]
    word_ids: list[str]
    word_utterance_indices: list[int]
    word_utterance_word_indices: list[int]


@dataclass(frozen=True, slots=True)
class ForcedAlignmentExecutionHostV2:
    """Injected FA execution hooks for the live V2 path."""

    whisper_runner: Callable[[np.ndarray, str, bool], list[tuple[str, float]]] | None = None
    wave2vec_runner: Callable[[np.ndarray, list[str]], list[tuple[str, tuple[int, int]]]] | None = None
    canto_runner: (
        Callable[
            [np.ndarray, PreparedFaPayloadV2, ForcedAlignmentRequestV2],
            list[tuple[str, tuple[int, int]]],
        ]
        | None
    ) = None


def build_default_fa_execution_host_v2(
    *,
    whisper_model: WhisperFAHandle | None,
    wave2vec_model: Wave2VecFAHandle | None,
) -> ForcedAlignmentExecutionHostV2:
    """Build the live V2 FA host from already loaded model handles."""

    from batchalign.inference.fa import infer_wave2vec_fa, infer_whisper_fa
    import torch

    def _as_tensor(audio: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(audio, dtype=np.float32))

    whisper_runner = None
    if whisper_model is not None:

        def _run_whisper(audio: np.ndarray, text: str, pauses: bool) -> list[tuple[str, float]]:
            return infer_whisper_fa(whisper_model, _as_tensor(audio), text, pauses=pauses)

        whisper_runner = _run_whisper

    wave2vec_runner = None
    if wave2vec_model is not None:

        def _run_wave2vec(
            audio: np.ndarray,
            words: list[str],
        ) -> list[tuple[str, tuple[int, int]]]:
            return infer_wave2vec_fa(wave2vec_model, _as_tensor(audio), words)

        wave2vec_runner = _run_wave2vec

    return ForcedAlignmentExecutionHostV2(
        whisper_runner=whisper_runner,
        wave2vec_runner=wave2vec_runner,
    )


def _wrap_canto_runner(
    runner: Callable[[np.ndarray, PreparedFaPayloadV2, ForcedAlignmentRequestV2], object] | None,
) -> Callable[[np.ndarray, str, str], object] | None:
    """Adapt the legacy typed Cantonese host hook to the Rust bridge shape."""

    if runner is None:
        return None

    def _run(audio: np.ndarray, payload_json: str, request_json: str) -> object:
        return runner(
            audio,
            PreparedFaPayloadV2.model_validate_json(payload_json),
            ForcedAlignmentRequestV2.model_validate_json(request_json),
        )

    return _run


def execute_forced_alignment_request_v2(
    request: ExecuteRequestV2,
    host: ForcedAlignmentExecutionHostV2,
) -> ExecuteResponseV2:
    """Execute one live V2 forced-alignment request through the Rust bridge."""

    import batchalign_core

    return ExecuteResponseV2.model_validate_json(
        batchalign_core.execute_forced_alignment_request_v2(
            request,
            host.whisper_runner,
            host.wave2vec_runner,
            _wrap_canto_runner(host.canto_runner),
        )
    )
