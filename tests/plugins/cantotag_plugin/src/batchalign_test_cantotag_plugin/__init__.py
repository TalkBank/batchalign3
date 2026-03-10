"""Synthetic plugin used by integration tests."""

from __future__ import annotations

from batchalign.providers import PluginDescriptor

plugin = PluginDescriptor(
    cmd2task={"cantotag": "morphosyntax"},
    command_probes={"cantotag": ()},
    command_base_mb={"cantotag": 2000},
)
