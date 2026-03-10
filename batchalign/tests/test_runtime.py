"""Tests for lightweight shared runtime constants."""

from __future__ import annotations

from batchalign.runtime import Cmd2Task


def test_cmd2task_keeps_server_owned_commands_in_shared_constants() -> None:
    """Rust and Python should share the same command/task map."""

    assert Cmd2Task["transcribe"] == "asr"
    assert Cmd2Task["transcribe_s"] == "asr"
    assert Cmd2Task["benchmark"] == "asr,eval"
