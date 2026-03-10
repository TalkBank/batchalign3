"""Stanza and language-code loading helpers for the worker process.

This module exists to keep Stanza-specific bootstrap policy out of the generic
worker entrypoint and the request-time inference routers. It owns:

- ISO language-code normalization for Stanza
- the MWT/non-MWT processor policy
- installation of preloaded Stanza pipelines into worker runtime state
- the utseg-specific stanza-config builder used by inference dispatch
"""

from __future__ import annotations

import logging
import threading

from batchalign.inference._domain_types import LanguageCode, LanguageCode2
from batchalign.worker._types import _state

L = logging.getLogger("batchalign.worker")

# IMPORTANT: This set must match batchalign2's MWT exclusion-list inversion.
# Languages in the exclusion list must not appear here or we will silently
# diverge from historical tokenizer behavior.
MWT_LANGS = {
    "fr", "de", "it", "es", "pt", "ca", "cs", "pl", "nl", "ar",
    "tr", "fi", "lv", "lt", "sk", "uk", "sv", "nb", "nn", "is",
    "gl", "cy", "gd", "mt", "ka", "hy", "fa", "hi", "ur", "bn",
    "ta", "te", "kn", "ml", "th", "vi", "id", "ms", "tl",
}


def iso3_to_alpha2(iso3: LanguageCode) -> LanguageCode2:
    """Convert ISO-639-3 language code to ISO-639-1 for Stanza.

    Batchalign uses ISO-639-3 broadly, but Stanza is configured with mostly
    ISO-639-1-style identifiers plus a few special cases. This function is the
    canonical bridge so the rest of the worker code does not embed ad hoc
    language-code fallbacks or guess at unsupported codes.
    """
    mapping: dict[str, str] = {
        "eng": "en", "spa": "es", "fra": "fr", "deu": "de",
        "ita": "it", "por": "pt", "nld": "nl", "zho": "zh",
        "jpn": "ja", "kor": "ko", "ara": "ar", "heb": "he",
        "tur": "tr", "fin": "fi", "dan": "da", "swe": "sv",
        "nor": "nb", "pol": "pl", "ces": "cs", "ron": "ro",
        "hun": "hu", "bul": "bg", "hrv": "hr", "slk": "sk",
        "slv": "sl", "ukr": "uk", "ell": "el", "fas": "fa",
        "hin": "hi", "urd": "ur", "ben": "bn", "tam": "ta",
        "tel": "te", "kan": "kn", "mal": "ml", "tha": "th",
        "vie": "vi", "ind": "id", "msa": "ms", "tgl": "tl",
        "kat": "ka", "hye": "hy", "cat": "ca", "glg": "gl",
        "eus": "eu", "cym": "cy", "gle": "ga", "gla": "gd",
        "mlt": "mt", "est": "et", "lav": "lv", "lit": "lt",
        "isl": "is", "yue": "zh",
        "cmn": "zh",
    }
    if iso3 in mapping:
        return mapping[iso3]
    if len(iso3) == 2:
        return iso3
    L.warning(
        "Unknown ISO-639-3 code %r - passing through unchanged for Stanza",
        iso3,
    )
    return iso3


def load_stanza_models(lang: LanguageCode) -> None:
    """Load Stanza morphosyntax models for one language.

    The resulting pipeline, tokenizer context, and lock are installed into the
    shared worker state so request handlers can do pure inference routing
    without rebuilding Stanza pipelines on every call.
    """
    import stanza
    from stanza import DownloadMethod

    from batchalign.inference._tokenizer_realign import (
        TokenizerContext,
        make_tokenizer_postprocessor,
    )

    alpha2 = iso3_to_alpha2(lang)
    has_mwt = alpha2 in MWT_LANGS
    processors = "tokenize,pos,lemma,depparse"
    if has_mwt:
        processors += ",mwt"

    ctx = TokenizerContext()
    lock = threading.Lock()

    # The Stanza pipeline shape varies by language because tokenization and MWT
    # support are not uniform across the supported languages.
    if alpha2 == "ja":
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_pretokenized=True,
            package={
                "tokenize": "combined",
                "pos": "combined",
                "lemma": "combined",
                "depparse": "combined",
            },
        )
    elif not has_mwt:
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_pretokenized=True,
        )
    elif alpha2 == "en":
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_postprocessor=make_tokenizer_postprocessor(ctx, alpha2),
            package={"mwt": "gum"},
        )
    else:
        nlp = stanza.Pipeline(
            lang=alpha2,
            processors=processors,
            download_method=DownloadMethod.REUSE_RESOURCES,
            tokenize_no_ssplit=True,
            tokenize_postprocessor=make_tokenizer_postprocessor(ctx, alpha2),
        )

    # Preserve any pipelines already loaded for other languages in this worker.
    existing_pipelines = _state.stanza_pipelines or {}
    existing_contexts = _state.stanza_contexts or {}
    existing_pipelines[lang] = nlp
    existing_contexts[lang] = ctx
    _state.stanza_pipelines = existing_pipelines
    _state.stanza_contexts = existing_contexts
    _state.stanza_nlp_lock = lock

    try:
        _state.stanza_version = stanza.__version__
    except AttributeError:
        _state.stanza_version = "unknown"


def load_utseg_builder(lang: LanguageCode) -> None:
    """Load the utseg config builder for one primary language.

    Utterance segmentation uses a lighter-weight configuration boundary than
    morphosyntax. Instead of preloading full pipelines here, the worker stores a
    callable that can derive the necessary Stanza config bundle from a set of
    languages at inference time.
    """
    alpha2 = iso3_to_alpha2(lang)
    mwt_exclude = {"zh", "ja", "ko", "th", "vi", "my"}
    has_mwt = alpha2 not in mwt_exclude

    def build_stanza_config_from_langs(
        langs: list[str],
    ) -> tuple[list[str], dict[str, dict[str, str | bool]]]:
        """Build the Stanza config payload expected by utseg inference."""
        lang_alpha2: list[str] = []
        configs: dict[str, dict[str, str | bool]] = {}
        for language in langs:
            alpha2_code = iso3_to_alpha2(language)
            if alpha2_code == "zh":
                alpha2_code = "zh-hans"
            lang_alpha2.append(alpha2_code)
            processors: set[str] = {"tokenize", "pos", "lemma", "constituency"}
            if has_mwt:
                processors.add("mwt")
            configs[alpha2_code] = {
                "processors": ",".join(sorted(processors)),
                "tokenize_pretokenized": True,
            }
        return lang_alpha2, configs

    _state.utseg_config_builder = build_stanza_config_from_langs

    try:
        import stanza

        _state.utseg_version = stanza.__version__
    except (ImportError, AttributeError):
        _state.utseg_version = "unknown"
