"""Tests for the Stanza capability table builder.

The capability table is the single source of truth for what Stanza
supports per language. It replaces 7 scattered hardcoded tables.
"""


def test_table_is_non_empty():
    """resources.json should produce a non-empty capability table."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()
    assert len(table.languages) > 40, (
        f"Expected 40+ languages, got {len(table.languages)}"
    )


def test_english_has_constituency():
    """English is one of ~11 languages with constituency parsing."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()
    assert "eng" in table.languages
    assert table.languages["eng"].has_constituency


def test_dutch_has_no_constituency():
    """Dutch does NOT have constituency — this caused Brian's crash."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()
    assert "nld" in table.languages
    assert not table.languages["nld"].has_constituency


def test_dutch_has_core_processors():
    """Dutch has tokenize, pos, lemma, depparse — morphotag should work."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()
    nl = table.languages["nld"]
    assert nl.has_tokenize
    assert nl.has_pos
    assert nl.has_lemma
    assert nl.has_depparse


def test_iso3_mapping_covers_core_languages():
    """The derived iso3 mapping must cover languages that Stanza actually
    supports with at least tokenize. Languages in the old hardcoded table
    that Stanza doesn't actually support are intentionally excluded."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()

    # Core languages that Stanza definitely supports with full processors.
    core = {
        "eng", "spa", "fra", "deu", "ita", "por", "nld", "jpn",
        "kor", "ara", "heb", "tur", "fin", "dan", "swe", "pol",
        "ces", "ron", "hun", "bul", "hrv", "slk", "slv", "ukr",
        "ell", "fas", "hin", "urd", "tha", "vie", "ind", "cat",
        "eus", "cym", "est", "lav", "lit", "isl", "rus", "afr",
        "lat",
    }
    missing = core - set(table.iso3_to_alpha2.keys())
    assert not missing, (
        f"These core languages should be in the derived mapping: {missing}"
    )


def test_mwt_matches_resources():
    """MWT availability should come from resources.json, not hardcoded list."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()

    # French definitely has MWT (du = de + le)
    assert table.languages["fra"].has_mwt
    # English has MWT (gum package)
    assert table.languages["eng"].has_mwt


def test_japanese_has_constituency():
    """Japanese has constituency parsing."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()
    ja = table.languages["jpn"]
    assert ja.has_constituency


def test_unsupported_language_not_in_table():
    """Languages Stanza can't process should not appear."""
    from batchalign.worker._stanza_capabilities import build_stanza_capability_table

    table = build_stanza_capability_table()
    # Quechua, Jamaican Creole — not in Stanza
    assert "que" not in table.languages
    assert "jam" not in table.languages
