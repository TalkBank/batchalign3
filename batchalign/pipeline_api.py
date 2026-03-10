"""Thin Python facade for Rust-owned CHAT pipeline execution.

This module intentionally keeps Python out of CHAT parsing and document
mutation. Python still matters for one reason: most of the model ecosystems we
must call are Python-native. Everything else in the CHAT-aware pipeline should
be owned by Rust.

The public shape here is therefore minimal:

- provider adapters that know how to call model backends
- a small typed operation record that describes *what* Rust should do
- one `run_pipeline()` entry point that forwards the operation sequence into
  `batchalign_core.run_provider_pipeline()`

Pre-processing, payload extraction, result injection, and post-processing all
live in Rust now.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
from typing import Protocol, cast

from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    WorkerJSONValue,
)

# Callback payloads and responses are intentionally plain JSON-shaped dicts.
# The Rust core owns the structural semantics of those payloads.
PayloadDict = dict[str, WorkerJSONValue]
ResponseDict = dict[str, WorkerJSONValue]
RustProviderPayload = dict[str, object]
RustProviderBatchRunner = Callable[
    [str, str, list[RustProviderPayload]],
    list[RustProviderPayload | None],
]


class ProviderBatchRunner(Protocol):
    """Callable contract for one raw provider batch invocation.

    Rust calls back into Python with:

    - `task`: logical task name such as `"translate"` or `"fa"`
    - `lang`: ISO-639-3 language code
    - `items`: JSON-shaped payloads extracted in Rust from the CHAT AST

    The return value must be one response object per payload. Any model-specific
    parsing should already have happened inside the Python engine adapter.
    """

    def __call__(
        self,
        task: str,
        lang: str,
        items: list[PayloadDict],
    ) -> list[ResponseDict | None]: ...


@dataclass(frozen=True, slots=True)
class PipelineOperation:
    """One Rust-owned document operation to apply to a parsed CHAT file.

    `name` selects the Rust operation. `options` are operation-local settings
    forwarded unchanged to the PyO3 bridge, which validates them again at the
    Rust boundary.
    """

    name: str
    options: dict[str, WorkerJSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject option maps that would obscure the operation identity."""
        if "name" in self.options:
            raise ValueError("PipelineOperation.options must not contain 'name'")

    def to_wire(self) -> PayloadDict:
        """Convert the operation into the wire shape consumed by Rust."""
        return {"name": self.name, **self.options}


@dataclass(slots=True)
class LocalProviderInvoker:
    """Run pipeline provider requests against in-process Python handlers.

    This is primarily useful for tests and small local integrations where the
    caller already has model-call functions in memory.
    """

    handlers: dict[str, Callable[[str, list[PayloadDict]], list[ResponseDict]]]

    def __call__(
        self,
        task: str,
        lang: str,
        items: list[PayloadDict],
    ) -> list[ResponseDict | None]:
        """Dispatch one logical task through the configured local handler."""
        handler = self.handlers.get(task)
        if handler is None:
            raise KeyError(f"No provider configured for task {task!r}")
        return [dict(result) for result in handler(lang, items)]


@dataclass(slots=True)
class BatchInferProviderInvoker:
    """Route pipeline provider requests through the worker batch-infer IPC.

    This adapter is the normal bridge for the worker/process path: Rust owns
    CHAT semantics, and Python delegates raw inference requests to the existing
    worker task router.
    """

    infer: Callable[[BatchInferRequest], BatchInferResponse]

    def __call__(
        self,
        task: str,
        lang: str,
        items: list[PayloadDict],
    ) -> list[ResponseDict | None]:
        """Marshal one logical pipeline task through the Rust-owned adapter."""

        import batchalign_core

        return cast(
            list[ResponseDict | None],
            json.loads(
                batchalign_core.call_batch_infer_provider(
                    task,
                    lang,
                    items,
                    self.infer,
                )
            ),
        )


def unwrap_batch_results(
    task: str,
    response: BatchInferResponse,
) -> list[ResponseDict | None]:
    """Convert worker batch results into plain JSON dicts or raise.

    The Python API surface keeps this helper for callers/tests, but Rust now
    owns the batch-infer result unwrapping and validation rules.
    """

    import batchalign_core

    return cast(
        list[ResponseDict | None],
        json.loads(batchalign_core.unwrap_batch_infer_results(task, response)),
    )


def _unused_provider(
    task: str,
    lang: str,
    items: list[PayloadDict],
) -> list[ResponseDict | None]:
    """Raise when a provider-backed operation is attempted without a provider."""
    raise RuntimeError(
        f"Operation {task!r} requires a provider runner, but no provider was supplied"
    )


def run_pipeline(
    chat_text: str,
    *,
    lang: str,
    operations: Sequence[PipelineOperation],
    provider: ProviderBatchRunner | None = None,
    lenient: bool = False,
) -> str:
    """Run a Rust-owned sequence of CHAT operations over one document.

    The Python side only contributes the provider callback for raw model
    batches. Rust owns the operation loop, payload extraction, result
    validation, document mutation, and final serialization.
    """

    import batchalign_core
    rust_provider = cast(RustProviderBatchRunner, provider or _unused_provider)
    rust_operations = cast(
        list[RustProviderPayload],
        [operation.to_wire() for operation in operations],
    )

    return batchalign_core.run_provider_pipeline(
        chat_text,
        lang=lang,
        provider_batch_fn=rust_provider,
        operations=rust_operations,
        lenient=lenient,
    )


__all__ = [
    "BatchInferProviderInvoker",
    "LocalProviderInvoker",
    "PayloadDict",
    "PipelineOperation",
    "ProviderBatchRunner",
    "ResponseDict",
    "run_pipeline",
    "unwrap_batch_results",
]
