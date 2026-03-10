"""Tests for Rust-based morphosyntax callback handling."""

from __future__ import annotations

import json
from typing import Any

import pytest

batchalign_core = pytest.importorskip("batchalign_core")


CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
@End
"""

CHAT_TWO = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\thello world .
*CHI:\tgoodbye moon .
@End
"""


def _raw_sentences(words: list[str]) -> dict[str, Any]:
    rows = []
    for i, word in enumerate(words, start=1):
        rows.append(
            {
                "id": i,
                "text": word,
                "lemma": word,
                "upos": "INTJ" if i == 1 else "NOUN",
                "head": 0 if i == 1 else 1,
                "deprel": "root" if i == 1 else "obj",
            }
        )
    return {"raw_sentences": [rows]}


class TestRustMorphosyntaxCallbacks:
    def test_callbacks_receive_python_objects(self) -> None:
        handle = batchalign_core.ParsedChat.parse(CHAT)
        seen_types: list[type[Any]] = []

        def callback(payload: dict[str, Any], lang: str) -> dict[str, Any]:
            seen_types.append(type(payload))
            assert lang == "eng"
            return _raw_sentences(payload["words"])

        handle.add_morphosyntax("eng", callback)
        result = handle.serialize()
        assert seen_types == [dict]
        assert "%mor:" in result
        assert "%gra:" in result

    def test_batched_callbacks_accept_object_lists(self) -> None:
        handle = batchalign_core.ParsedChat.parse(CHAT_TWO)
        seen_types: list[type[Any]] = []

        def callback(items: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
            seen_types.append(type(items))
            assert lang == "eng"
            return [_raw_sentences(item["words"]) for item in items]

        handle.add_morphosyntax_batched("eng", callback)
        result = handle.serialize()
        assert seen_types == [list]
        assert result.count("%mor:") == 2
        assert result.count("%gra:") == 2

    def test_callback_failure_preserves_original_chat(self) -> None:
        handle = batchalign_core.ParsedChat.parse(CHAT_TWO)
        original = handle.serialize()
        calls = 0

        def flaky_callback(payload: dict[str, Any], lang: str) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            assert lang == "eng"
            if calls == 2:
                raise RuntimeError("morphosyntax boom")
            return _raw_sentences(payload["words"])

        with pytest.raises(Exception, match="morphosyntax boom"):
            handle.add_morphosyntax("eng", flaky_callback)

        assert calls == 2
        assert handle.serialize() == original

    def test_batched_progress_failure_preserves_original_chat(self) -> None:
        handle = batchalign_core.ParsedChat.parse(CHAT_TWO)
        original = handle.serialize()

        def callback(items: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
            assert lang == "eng"
            return [_raw_sentences(item["words"]) for item in items]

        def progress_fn(_completed: int, _total: int) -> None:
            raise RuntimeError("batched morphosyntax progress boom")

        with pytest.raises(Exception, match="batched morphosyntax progress boom"):
            handle.add_morphosyntax_batched("eng", callback, progress_fn=progress_fn)

        assert handle.serialize() == original

    def test_cache_injection_failure_preserves_original_chat(self) -> None:
        source = batchalign_core.ParsedChat.parse(CHAT_TWO)
        payloads = json.loads(source.extract_morphosyntax_payloads("eng"))
        line_indices = [payload["line_idx"] for payload in payloads]
        source.add_morphosyntax_batched(
            "eng",
            lambda items, lang: [_raw_sentences(item["words"]) for item in items],
        )
        cached_entries = json.loads(source.extract_morphosyntax_strings(json.dumps(line_indices)))
        cached_entries[1]["line_idx"] = 999

        target = batchalign_core.ParsedChat.parse(CHAT_TWO)
        original = target.serialize()

        with pytest.raises(Exception, match="Line index 999 out of range"):
            target.inject_morphosyntax_from_cache(json.dumps(cached_entries))

        assert target.serialize() == original
