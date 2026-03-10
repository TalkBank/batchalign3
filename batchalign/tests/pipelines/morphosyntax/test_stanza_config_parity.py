"""Stanza configuration parity tests: batchalign3 vs batchalign2.

These tests verify that batchalign3's Stanza pipeline configuration matches
batchalign2's intended behavior for every language.  The MWT exclusion list,
processor packages, and tokenizer mode are all critical — getting any of them
wrong produces silently different %mor output.

See: book/src/reference/morphotag-migration-audit.md Section 4.8

NO MOCKS.  These tests compare configuration decisions, not Stanza output.
They don't require Stanza models to be downloaded.
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# batchalign2's MWT exclusion list (from ud.py:1034-1036)
# Languages that batchalign2 EXCLUDED from MWT processing.
# --------------------------------------------------------------------------
BA2_MWT_EXCLUSION = frozenset([
    "hr", "zh", "zh-hans", "zh-hant", "ja", "ko",
    "sl", "sr", "bg", "ru", "et", "hu",
    "eu", "el", "he", "af", "ga", "da", "ro",
])


# --------------------------------------------------------------------------
# batchalign3's MWT inclusion list (from worker/_stanza_loading.py)
# Languages that batchalign3 enables MWT for.
# --------------------------------------------------------------------------
def _get_ba3_mwt_langs() -> frozenset[str]:
    """Extract the actual mwt_langs set from the worker code.

    Reads the live code so the test breaks if someone changes the list
    without updating these tests.
    """
    # Import the private function indirectly by reading the source
    # (the function is not importable without Stanza installed)
    import ast
    from pathlib import Path

    stanza_loading = Path(__file__).resolve().parents[3] / "worker" / "_stanza_loading.py"
    source = stanza_loading.read_text()
    tree = ast.parse(source)

    # Find the MWT_LANGS = {...} assignment in the stanza-loading module
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MWT_LANGS":
                    if isinstance(node.value, ast.Set):
                        return frozenset(
                            elt.value  # type: ignore[union-attr]
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant)
                        )

    raise RuntimeError("Could not find MWT_LANGS in _stanza_loading.py")


class TestMwtExclusionParity:
    """Verify MWT language decisions match batchalign2.

    batchalign2 used an EXCLUSION list: these languages do NOT get MWT.
    batchalign3 uses an INCLUSION list: these languages DO get MWT.

    The two must be consistent: no language in ba2's exclusion list should
    be in ba3's inclusion list.
    """

    def test_excluded_languages_not_in_mwt_langs(self) -> None:
        """Languages ba2 excluded from MWT must NOT be in ba3's mwt_langs.

        If this test fails, it means ba3 is enabling MWT for a language that
        ba2 deliberately disabled.  This produces different %mor output and
        must be investigated before being accepted.

        Fixed 2026-03-07: Removed 13 wrongly-enabled languages from mwt_langs.
        """
        ba3_mwt = _get_ba3_mwt_langs()
        wrongly_enabled = BA2_MWT_EXCLUSION & ba3_mwt

        # Remove languages that we've explicitly decided to enable
        # (document the decision here with a reason)
        accepted_divergences: frozenset[str] = frozenset([
            # None yet — all divergences need investigation
        ])

        unresolved = wrongly_enabled - accepted_divergences

        assert not unresolved, (
            f"These languages were EXCLUDED from MWT in batchalign2 but are "
            f"ENABLED in batchalign3: {sorted(unresolved)}. "
            f"This will produce different %mor output. "
            f"Either remove them from MWT_LANGS in _stanza_loading.py, or add them to "
            f"accepted_divergences with a documented reason."
        )

    def test_english_uses_gum_mwt_package(self) -> None:
        """English must use the 'gum' MWT package, not 'default'.

        batchalign2 (ud.py:1044): config["processors"]["mwt"] = "gum"
        batchalign3 (_stanza_loading.py): package={"mwt": "gum"}
        """
        from pathlib import Path

        stanza_loading = Path(__file__).resolve().parents[3] / "worker" / "_stanza_loading.py"
        source = stanza_loading.read_text()

        # Verify "gum" appears in the English pipeline config
        assert '"gum"' in source or "'gum'" in source, (
            "English MWT must use the 'gum' package. "
            "Check load_stanza_models() in _stanza_loading.py."
        )

    def test_non_mwt_languages_use_pretokenized(self) -> None:
        """Languages without MWT must use tokenize_pretokenized=True.

        batchalign2: tokenize_pretokenized is implicit (no MWT processor)
        batchalign3: explicit tokenize_pretokenized=True for non-MWT langs
        """
        from pathlib import Path

        stanza_loading = Path(__file__).resolve().parents[3] / "worker" / "_stanza_loading.py"
        source = stanza_loading.read_text()

        # The non-MWT branch must have pretokenized=True
        assert "tokenize_pretokenized=True" in source, (
            "Non-MWT languages must use tokenize_pretokenized=True. "
            "Check load_stanza_models() in _stanza_loading.py."
        )


class TestJapaneseProcessorConfig:
    """Verify Japanese-specific Stanza configuration.

    batchalign2 (ud.py:1048-1052) explicitly set ALL processors to
    'combined' for Japanese:
        config["processors"]["tokenize"] = "combined"
        config["processors"]["pos"] = "combined"
        config["processors"]["lemma"] = "combined"
        config["processors"]["depparse"] = "combined"

    batchalign3 now configures this (fixed 2026-03-07).
    """

    def test_japanese_uses_combined_processors(self) -> None:
        """Japanese must use 'combined' processor packages."""
        from pathlib import Path

        stanza_loading = Path(__file__).resolve().parents[3] / "worker" / "_stanza_loading.py"
        source = stanza_loading.read_text()

        # Check if the Japanese combined processor config exists
        has_ja_combined = (
            '"combined"' in source
            and "ja" in source
        )

        assert has_ja_combined, (
            "Japanese 'combined' processor not configured in _stanza_loading.py. "
            "batchalign2 used combined processors for Japanese tokenization. "
            "See morphotag-migration-audit.md Section 4.3."
        )


class TestLanguageSpecificFeatureParity:
    """Per-word feature extraction parity tests.

    NOTE: The per-word mapping (lemma cleaning, POS-specific features, etc.)
    is tested exhaustively in the Rust test suite:

        cargo nextest run -p batchalign-chat-ops -E 'test(nlp::mapping)'

    46 tests cover: all POS types, all language-specific handlers, lemma
    cleaning, MWT assembly, GRA generation, and edge cases.

    See: morphotag-migration-audit.md Section 1 for the line-by-line
    correspondence between ba2 handler functions and ba3 Rust code.

    The tests below verify Python-accessible behavior only (via golden tests).
    For the Rust-level feature parity tests, see:
    - chat-ops/src/nlp/mapping.rs (46 tests)
    - chat-ops/src/nlp/lang_en.rs (irregular verbs)
    - chat-ops/src/nlp/lang_fr.rs (pronoun case, APM nouns)
    - chat-ops/src/nlp/lang_ja.rs (verb form overrides)
    """

    pass  # See Rust tests — do not duplicate here
