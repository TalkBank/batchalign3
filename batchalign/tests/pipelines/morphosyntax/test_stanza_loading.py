"""Focused tests for worker-side Stanza language mapping."""

from __future__ import annotations

import logging

from batchalign.worker._stanza_loading import iso3_to_alpha2


def test_iso3_to_alpha2_maps_known_languages() -> None:
    assert iso3_to_alpha2("eng") == "en"
    assert iso3_to_alpha2("yue") == "zh"


def test_iso3_to_alpha2_preserves_existing_alpha2_codes() -> None:
    assert iso3_to_alpha2("en") == "en"
    assert iso3_to_alpha2("ja") == "ja"


def test_iso3_to_alpha2_leaves_unknown_iso3_unchanged(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="batchalign.worker"):
        assert iso3_to_alpha2("zzz") == "zzz"

    assert "Unknown ISO-639-3 code 'zzz' - passing through unchanged for Stanza" in caplog.text
