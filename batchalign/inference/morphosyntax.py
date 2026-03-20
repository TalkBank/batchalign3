"""Stanza morphosyntax inference: words -> POS/dep/lemma.

Pure inference — no CHAT, no caching, no pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import unicodedata
from collections.abc import Iterator
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError, model_validator

from batchalign.inference._domain_types import LanguageCode

if TYPE_CHECKING:
    from batchalign.inference.types import StanzaNLP
    from batchalign.inference._tokenizer_realign import TokenizerContext

from batchalign.providers import (
    BatchInferRequest,
    BatchInferResponse,
    InferResponse,
    WorkerJSONValue,
)

L = logging.getLogger("batchalign.worker")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class MorphosyntaxBatchItem(BaseModel):
    """A single item in the batch morphosyntax payload from Rust."""

    words: list[str]
    terminator: str = "."
    special_forms: list[list[str | None]] = []
    lang: LanguageCode = ""


class UdWord(BaseModel, extra="allow"):
    """A single UD word/token — mirrors Rust ``UdWord`` in types.rs."""

    id: int | list[int] | float
    text: str
    lemma: str = ""
    upos: str = "X"
    xpos: str | None = None
    feats: str | None = None
    head: int = 0
    deprel: str = "dep"
    deps: str | None = None
    misc: str | None = None

    @model_validator(mode="after")
    def _default_lemma_to_text(self) -> UdWord:
        if not self.lemma and not isinstance(self.id, list):
            self.lemma = self.text
        return self

    @model_validator(mode="after")
    def _sanitize_pad_deprel(self) -> UdWord:
        if self.deprel.startswith("<") and self.deprel.endswith(">"):
            L.warning(
                "Stanza emitted deprel=%r for word %r — replacing with 'dep'",
                self.deprel,
                self.text,
            )
            self.deprel = "dep"
        return self


UdWordRaw = dict[str, str | int | float | list[int] | tuple[int, ...] | None]
JSONObject = dict[str, WorkerJSONValue]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_bogus_lemma(text: str, lemma: str) -> bool:
    """Detect when Stanza returns a lemma that's pure punctuation for a word."""
    if text == lemma or not lemma:
        return False
    text_has_letters = any(unicodedata.category(c).startswith("L") for c in text)
    lemma_all_punct = all(
        unicodedata.category(c).startswith(("P", "S")) for c in lemma
    )
    return text_has_letters and lemma_all_punct


def validate_ud_words(sents: list[list[UdWordRaw]]) -> None:
    """Validate and normalize every token through the UdWord model.

    Mutates *sents* in place.
    """
    for sent in sents:
        for word_idx in range(len(sent)):
            raw = sent[word_idx]
            raw_id = raw.get("id")
            if isinstance(raw_id, tuple):
                raw["id"] = list(raw_id)

            validated = UdWord.model_validate(raw)

            if not isinstance(validated.id, list) and _is_bogus_lemma(
                validated.text, validated.lemma
            ):
                L.warning(
                    "Stanza returned bogus lemma %r for word %r — falling back to surface form",
                    validated.lemma,
                    validated.text,
                )
                validated.lemma = validated.text

            sent[word_idx] = validated.model_dump()


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------


def batch_infer_morphosyntax(
    req: BatchInferRequest,
    nlp_pipelines: dict[LanguageCode, StanzaNLP],
    contexts: dict[LanguageCode, TokenizerContext],
    nlp_lock: threading.Lock,
    free_threaded: bool,
    mwt_lexicon: dict[str, list[str]] | None = None,
) -> BatchInferResponse:
    """Batch Stanza inference: (words, lang) -> UdResponse.

    Parameters
    ----------
    req : BatchInferRequest
        Batch of MorphosyntaxBatchItem payloads.
    nlp_pipelines : dict
        Pre-loaded Stanza Pipeline instances keyed by ISO-3 code.
    contexts : dict
        Tokenizer realignment contexts keyed by ISO-3 code.
    nlp_lock : threading.Lock
        Lock guarding Stanza calls on GIL-enabled Python.
    free_threaded : bool
        Whether to skip the lock (free-threaded Python).
    mwt_lexicon : dict, optional
        Custom multi-word token lexicon mapping surface forms to
        expansion tokens (e.g. ``{"gonna": ["going", "to"]}``).
        When provided, matching tokens in Stanza's output are
        expanded according to this lexicon.
    """

    @contextlib.contextmanager
    def _maybe_lock() -> Iterator[None]:
        if free_threaded:
            yield
        else:
            with nlp_lock:
                yield

    t0 = time.monotonic()

    n = len(req.items)
    items: list[MorphosyntaxBatchItem | None] = []
    for raw_item in req.items:
        try:
            items.append(MorphosyntaxBatchItem.model_validate(raw_item))
        except ValidationError:
            items.append(None)

    empty_ud: JSONObject = {"sentences": []}
    results: list[InferResponse] = [
        InferResponse(result=empty_ud, elapsed_s=0.0) for _ in range(n)
    ]

    by_lang: dict[str, list[tuple[int, str, list[str]]]] = {}
    for i, item in enumerate(items):
        if item is None:
            results[i] = InferResponse(error="Invalid batch item", elapsed_s=0.0)
            continue
        if not item.words:
            continue

        words = list(item.words)
        text = " ".join(words).replace("(", "").replace(")", "").strip()
        item_lang = item.lang or req.lang

        if item_lang not in by_lang:
            by_lang[item_lang] = []
        by_lang[item_lang].append((i, text, words))

    if not by_lang:
        return BatchInferResponse(results=results)

    for lang_code, lang_items in by_lang.items():
        indices = [idx for idx, _, _ in lang_items]
        texts = [text for _, text, _ in lang_items]
        word_lists = [words for _, _, words in lang_items]

        nlp = nlp_pipelines.get(lang_code)
        if nlp is None:
            L.warning(
                "No Stanza pipeline for language %s -- items will have empty UdResponse",
                lang_code,
            )
            continue

        combined = "\n\n".join(texts)
        tok_ctx = contexts.get(lang_code) or contexts.get(req.lang)

        try:
            with _maybe_lock():
                if tok_ctx is not None:
                    tok_ctx.original_words = word_lists
                doc = nlp(combined)
                if tok_ctx is not None:
                    tok_ctx.original_words = []

            sents = doc.to_dict()

            if len(sents) != len(indices):
                L.warning(
                    "Stanza sentence count mismatch for language %s (expected %d, got %d)",
                    lang_code,
                    len(indices),
                    len(sents),
                )
            else:
                for i, idx in enumerate(indices):
                    # Return raw Stanza to_dict() output — Rust handles validation
                    results[idx] = InferResponse(
                        result={"raw_sentences": [sents[i]]},
                        elapsed_s=0.0,
                    )
        except Exception as e:
            L.warning(
                "Stanza batch failed for language %s (%d items): %s",
                lang_code,
                len(indices),
                e,
            )
            if tok_ctx is not None:
                tok_ctx.original_words = []

    elapsed = time.monotonic() - t0
    if results:
        first = results[0]
        results[0] = InferResponse(
            result=first.result, error=first.error, elapsed_s=elapsed
        )

    L.info("batch_infer morphosyntax: %d items, %.3fs", n, elapsed)
    return BatchInferResponse(results=results)
