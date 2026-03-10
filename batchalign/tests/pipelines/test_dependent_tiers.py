"""Tests for batchalign_core.add_dependent_tiers.

Verifies:
- User-defined tiers with 'x' prefix round-trip correctly
- Standard tier labels (%mor, %gra) are accepted
- Empty content is silently skipped
- Output re-parses without errors
"""

from __future__ import annotations

import json

import pytest

batchalign_core = pytest.importorskip("batchalign_core")


CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR0 Participant
@ID:\teng|test|PAR0|||||Participant|||
*PAR0:\thello world .
*PAR0:\tgood morning .
@End
"""


class TestAddDependentTiers:
    """Tests for the add_dependent_tiers function."""

    def test_xcoref_tier_added(self) -> None:
        """User-defined tier with 'x' prefix is added."""
        tiers = [{"utterance_index": 0, "label": "xcoref", "content": "(0) -"}]
        result = batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers))
        assert "%xcoref:" in result

    def test_xcoref_round_trips(self) -> None:
        """Output with %xcoref re-parses cleanly."""
        tiers = [{"utterance_index": 0, "label": "xcoref", "content": "(0) -"}]
        result = batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers))
        reparsed = batchalign_core.parse_and_serialize(result)
        assert "%xcoref:" in reparsed

    def test_multiple_tiers_on_different_utterances(self) -> None:
        """Tiers on separate utterances are all added."""
        tiers = [
            {"utterance_index": 0, "label": "xcoref", "content": "(0) -"},
            {"utterance_index": 1, "label": "xcoref", "content": "- (1)"},
        ]
        result = batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers))
        assert result.count("%xcoref:") == 2

    def test_empty_content_skipped(self) -> None:
        """Tiers with empty content are silently skipped."""
        tiers = [{"utterance_index": 0, "label": "xcoref", "content": ""}]
        result = batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers))
        assert "%xcoref:" not in result

    def test_empty_label_skipped(self) -> None:
        """Tiers with empty label are silently skipped."""
        tiers = [{"utterance_index": 0, "label": "", "content": "(0) -"}]
        result = batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers))
        # No new tier should appear
        result_parsed = batchalign_core.parse_and_serialize(result)
        assert result_parsed.strip() == batchalign_core.parse_and_serialize(CHAT).strip()

    def test_empty_tiers_list_returns_unchanged(self) -> None:
        """Empty tiers list returns CHAT unchanged."""
        result = batchalign_core.add_dependent_tiers(CHAT, json.dumps([]))
        assert "@End" in result

    def test_tier_replaces_existing(self) -> None:
        """Adding a tier with the same label replaces the existing one."""
        tiers1 = [{"utterance_index": 0, "label": "xcoref", "content": "old content"}]
        intermediate = batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers1))
        assert "old content" in intermediate

        tiers2 = [{"utterance_index": 0, "label": "xcoref", "content": "new content"}]
        result = batchalign_core.add_dependent_tiers(intermediate, json.dumps(tiers2))
        assert "new content" in result
        assert result.count("%xcoref:") == 1

    def test_standard_label_rejected(self) -> None:
        """Standard tier labels (e.g., 'mor', 'gra') are rejected."""
        tiers = [{"utterance_index": 0, "label": "mor", "content": "n|hello"}]
        with pytest.raises(ValueError, match="standard CHAT tier"):
            batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers))

    def test_non_x_prefix_rejected(self) -> None:
        """User-defined labels without 'x' prefix are rejected."""
        tiers = [{"utterance_index": 0, "label": "coref", "content": "(0) -"}]
        with pytest.raises(ValueError, match="must start with 'x'"):
            batchalign_core.add_dependent_tiers(CHAT, json.dumps(tiers))

    def test_handle_method_produces_same_result(self) -> None:
        """ParsedChat.add_dependent_tiers matches module-level function."""
        tiers_json = json.dumps([
            {"utterance_index": 0, "label": "xcoref", "content": "(0) -"},
        ])
        # Module-level function
        result_fn = batchalign_core.add_dependent_tiers(CHAT, tiers_json)
        # Handle method
        handle = batchalign_core.ParsedChat.parse(CHAT)
        handle.add_dependent_tiers(tiers_json)
        result_handle = handle.serialize()

        # Both should contain the tier
        assert "%xcoref:" in result_fn
        assert "%xcoref:" in result_handle
