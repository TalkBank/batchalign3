"""Python-owned console entry point for ``batchalign3``.

This keeps packaging and script wiring on the Python side while the actual
CLI implementation stays in Rust. In a source checkout, the wrapper can fall
back to the repo CLI when the embedded Rust bridge is intentionally omitted
for a faster extension-only rebuild.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _repo_root() -> Path | None:
    """Return the repo root when running from a checkout, else ``None``."""

    root = Path(__file__).resolve().parent.parent
    if (root / "Cargo.toml").exists() and (root / "crates" / "batchalign-cli").exists():
        return root
    return None


def _exec_repo_cli_fallback() -> None:
    """Exec the standalone Rust CLI from a repo checkout.

    This is the fast local-dev path: it keeps ``uv run batchalign3`` usable
    after ``make build-python`` without forcing the heavier packaged bridge.
    """

    root = _repo_root()
    if root is None:
        raise SystemExit(
            "batchalign3 could not find the embedded Rust CLI bridge, and this "
            "environment does not look like a batchalign3 source checkout."
        )

    built_binary = root / "target" / "debug" / "batchalign3"
    if built_binary.exists():
        os.execv(str(built_binary), [str(built_binary), *sys.argv[1:]])

    cargo = shutil.which("cargo")
    if cargo:
        os.execvp(
            cargo,
            [
                cargo,
                "run",
                "-q",
                "-p",
                "batchalign-cli",
                "--bin",
                "batchalign3",
                "--",
                *sys.argv[1:],
            ],
        )

    raise SystemExit(
        "batchalign3 was built without the embedded CLI bridge for faster local "
        "maturin iteration, but neither a compiled target/debug/batchalign3 nor "
        "a cargo executable was available. For the fastest loop, run "
        "`cargo build -p batchalign-cli` or `make build-rust`; otherwise "
        "rebuild with `make build-python-full`."
    )


def main() -> None:
    """Run the installed ``batchalign3`` command.

    The current implementation delegates to the native Rust CLI bridge exposed
    by ``batchalign_core``. Keeping the console entry point in Python gives us
    a stable seam for future packaging and boundary refactors.
    """

    try:
        from batchalign_core import cli_main
    except ImportError as exc:  # pragma: no cover - environment-specific failure
        if _repo_root() is not None:
            _exec_repo_cli_fallback()
        raise SystemExit(
            "batchalign3 could not import the native batchalign_core module. "
            "Reinstall batchalign3 or rebuild the local environment. In a "
            "source checkout, use `make build-python` for the fast extension "
            "build or `make build-python-full` to restore the embedded CLI "
            "bridge."
        ) from exc

    cli_main()
