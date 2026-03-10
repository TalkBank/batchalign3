#!/usr/bin/env python3
"""Fail CI when retired legacy names appear in active runtime/docs surfaces."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ACTIVE_PATHS: tuple[Path, ...] = (
    ROOT / "README.md",
    ROOT / ".github/workflows",
    ROOT / "batchalign",
    ROOT / "scripts",
    ROOT / "frontend/src",
    ROOT / "book/src/introduction.md",
    ROOT / "book/src/user-guide",
    ROOT / "book/src/developer/building.md",
)

SCAN_SUFFIXES = {
    ".css",
    ".js",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-314t",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
}

BANNED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbatchalign-next\b"), "retired command name"),
    (re.compile(r"\bbatchalign_next\b"), "retired package/module name"),
    (re.compile(r"\brust" r"-next\b"), "retired workspace path"),
    (re.compile(r"/opt/python/bin/" r"python"), "hardcoded interpreter path"),
    (re.compile(r"\bbatchalign\.cli\b"), "retired Python CLI package path"),
)

DOC_BANNED_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbatchalign2\b"), "legacy repository/name in active docs"),
)

DOC_ACTIVE_PREFIXES: tuple[str, ...] = (
    "book/src/introduction.md",
    "book/src/user-guide/",
    "book/src/developer/building.md",
)

# Narrow allowlist for one-time migration logic.
ALLOWLIST_LINE_SUBSTRINGS: dict[Path, tuple[str, ...]] = {
    Path("batchalign/runtime.py"): (
        "One-time migration: ~/.batchalign" "-next",
        'old = Path.home() / ".batchalign' '-next"',
    ),
}

# Files excluded entirely from scanning (contain banned strings as test data).
EXCLUDED_FILES: frozenset[Path] = frozenset({
    Path("crates/batchalign-cli/tests/ci_checks.rs"),
})


def _should_scan(path: Path, root: Path) -> bool:
    if not path.is_file() or path.suffix not in SCAN_SUFFIXES:
        return False
    if any(part in SKIP_DIRS for part in path.parts):
        return False
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return rel not in EXCLUDED_FILES


def _iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for base in ACTIVE_PATHS:
        if not base.exists():
            continue
        if _should_scan(base, ROOT):
            files.append(base)
            continue
        if base.is_dir():
            for path in base.rglob("*"):
                if _should_scan(path, ROOT):
                    files.append(path)
    files.sort()
    return files


def main() -> int:
    failures: list[str] = []

    for path in _iter_scan_files():
        rel = path.relative_to(ROOT)
        allow_substrings = ALLOWLIST_LINE_SUBSTRINGS.get(rel, ())
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for pattern, reason in BANNED_PATTERNS:
                for match in pattern.finditer(line):
                    if any(token in line for token in allow_substrings):
                        continue
                    failures.append(
                        f"{rel}:{line_no}: `{match.group(0)}` ({reason})\n"
                        f"  {line.strip()}"
                    )
            rel_s = rel.as_posix()
            if rel_s.startswith(DOC_ACTIVE_PREFIXES):
                for pattern, reason in DOC_BANNED_PATTERNS:
                    for match in pattern.finditer(line):
                        failures.append(
                            f"{rel}:{line_no}: `{match.group(0)}` ({reason})\n"
                            f"  {line.strip()}"
                        )

    if failures:
        print("Legacy term check failed. Replace retired names in active runtime/docs:")
        for failure in failures:
            print(f"- {failure}")
        print(
            "\nIf a legacy mention is intentionally historical, move it into\n"
            "book/src/decisions or add a narrowly scoped allowlist entry."
        )
        return 1

    print("Legacy term check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
