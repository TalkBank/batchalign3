"""Stdio JSON-lines IPC loop for the Python worker."""

from __future__ import annotations

import json
import os
import sys

from batchalign.worker._protocol_ops import dispatch_protocol_message
from batchalign.worker._types import WorkerJSONValue


def _write_json(payload: dict[str, WorkerJSONValue]) -> None:
    """Emit a single JSON message line to stdout."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _write_error(message: str) -> None:
    """Emit protocol-level error response for malformed requests/ops."""
    _write_json({"op": "error", "error": message})


def _print_ready() -> None:
    """Print a JSON ready line to stdout so the Rust parent can discover us."""
    _write_json({"ready": True, "pid": os.getpid(), "transport": "stdio"})


def _serve_stdio() -> None:
    """Run the stdio request loop until shutdown or EOF."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_error(f"invalid JSON request: {exc}")
            continue

        dispatch = dispatch_protocol_message(message)
        _write_json(dispatch.payload)
        if dispatch.should_shutdown:
            break
