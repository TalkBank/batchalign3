"""Verify that retired package paths raise ImportError (modules deleted)."""
from __future__ import annotations

import importlib
import sys

import pytest

CLI_MODULE = ".".join(("batchalign", "cli"))
SERVE_MODULE = ".".join(("batchalign", "serve"))
PIPELINES_MODULE = ".".join(("batchalign", "pipelines"))


def _expect_removed_import(module_name: str) -> None:
    sys.modules.pop(module_name, None)
    with pytest.raises((ImportError, ModuleNotFoundError)):
        importlib.import_module(module_name)


def test_cli_package_is_removed() -> None:
    _expect_removed_import(CLI_MODULE)


def test_server_package_is_removed() -> None:
    _expect_removed_import(SERVE_MODULE)


def test_pipelines_package_is_removed() -> None:
    _expect_removed_import(PIPELINES_MODULE)
