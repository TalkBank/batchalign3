#!/usr/bin/env python3
"""Verify version consistency between cli-pyproject.toml and Cargo.toml."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    # cli-pyproject.toml [project].version
    pyproject = REPO / "cli-pyproject.toml"
    py_version = tomllib.loads(pyproject.read_text())["project"]["version"]

    # crates/batchalign-cli/Cargo.toml [package].version
    cargo = REPO / "crates" / "batchalign-cli" / "Cargo.toml"
    cargo_version = tomllib.loads(cargo.read_text())["package"]["version"]

    ok = True
    if py_version != cargo_version:
        print(
            f"MISMATCH: cli-pyproject.toml version={py_version!r} "
            f"!= batchalign-cli/Cargo.toml version={cargo_version!r}",
            file=sys.stderr,
        )
        ok = False

    if ok:
        print(f"CLI version sync OK: {py_version}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
