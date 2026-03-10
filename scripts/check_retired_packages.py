#!/usr/bin/env python3
"""Ensure retired Python packages have zero tracked files.

These packages were fully deleted in the radical Python simplification
(commit 6766d0e8). No files should re-appear under them.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RETIRED_PATHS = ["batchalign/cli", "batchalign/serve"]


def _git_ls_files(pathspec: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", pathspec],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    failures: list[str] = []

    for pathspec in RETIRED_PATHS:
        tracked = [
            rel for rel in _git_ls_files(pathspec) if (ROOT / rel).exists()
        ]
        for rel in tracked:
            failures.append(
                f"{rel}: unexpected tracked file under retired package path `{pathspec}`"
            )

    if failures:
        print("Retired package boundary check failed:")
        for failure in failures:
            print(f"- {failure}")
        print(
            "\nRetired packages must stay deleted. Move active runtime files into "
            "Rust or non-retired paths."
        )
        return 1

    print("Retired package boundary check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
