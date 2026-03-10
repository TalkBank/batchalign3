"""Tests for the BA2 compatibility shim (batchalign.compat).

Covers CHATFile read/write, Document.new(), subscript access (doc[0][0]),
morphology/gra extraction, and deprecation warnings. BatchalignPipeline
execution is not tested here because it requires a working batchalign3 CLI
installation with ML models.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import batchalign_core
import pytest


# Minimal valid CHAT content (needs @UTF8, @Begin, @Languages, @Participants, @ID).
_MINIMAL_CHAT = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\thello world .
@End
"""

# CHAT content with %mor and %gra tiers for subscript access tests.
_CHAT_WITH_MOR = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Child
@ID:\teng|test|CHI|||||Child|||
*CHI:\tthe dog runs .
%mor:\tdet|the n|dog v|run-3S .
%gra:\t1|2|DET 2|3|SUBJ 3|0|ROOT 4|3|PUNCT
@End
"""

# CHAT with two utterances for multi-utterance tests.
_CHAT_TWO_UTTS = """\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Child, MOT Mother
@ID:\teng|test|CHI|||||Child|||
@ID:\teng|test|MOT|||||Mother|||
*CHI:\thello .
%mor:\tco|hello .
%gra:\t1|0|ROOT 2|1|PUNCT
*MOT:\thi there .
%mor:\tco|hi adv|there .
%gra:\t1|0|ROOT 2|1|JCT 3|1|PUNCT
@End
"""


class TestDeprecationWarning:
    """Importing batchalign.compat should emit a DeprecationWarning."""

    def test_import_warns(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import importlib

            import batchalign.compat as compat_mod

            importlib.reload(compat_mod)

            deprecation_warnings = [
                w for w in caught if issubclass(w.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()


class TestCHATFile:
    """CHATFile read/write roundtrip."""

    def test_read_from_path(self, tmp_path: Path) -> None:
        from batchalign.compat import CHATFile

        cha_path = tmp_path / "test.cha"
        cha_path.write_text(_MINIMAL_CHAT, "utf-8")

        chat = CHATFile(path=str(cha_path))
        assert chat.doc is not None
        serialized = chat.doc.serialize()
        assert "hello world" in serialized

    def test_write_roundtrip(self, tmp_path: Path) -> None:
        from batchalign.compat import CHATFile

        cha_path = tmp_path / "test.cha"
        cha_path.write_text(_MINIMAL_CHAT, "utf-8")

        chat = CHATFile(path=str(cha_path))
        out_path = tmp_path / "output.cha"
        chat.write(out_path)

        roundtripped = out_path.read_text("utf-8")
        assert "hello world" in roundtripped

    def test_from_document(self) -> None:
        from batchalign.compat import CHATFile, Document

        doc = Document._from_text(_MINIMAL_CHAT)
        chat = CHATFile(doc=doc)
        assert "hello world" in chat.doc.serialize()

    def test_both_args_raises(self) -> None:
        from batchalign.compat import CHATFile, Document

        doc = Document._from_text(_MINIMAL_CHAT)
        with pytest.raises(ValueError, match="not both"):
            CHATFile(path="foo.cha", doc=doc)

    def test_no_args_raises(self) -> None:
        from batchalign.compat import CHATFile

        with pytest.raises(ValueError, match="Provide either"):
            CHATFile()


class TestDocument:
    """Document creation and serialization."""

    def test_from_text(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_MINIMAL_CHAT)
        assert "hello world" in doc.serialize()
        assert doc.transcript == _MINIMAL_CHAT

    def test_new_creates_valid_chat(self) -> None:
        from batchalign.compat import Document

        doc = Document.new("this is a test .")
        serialized = doc.serialize()

        parsed = batchalign_core.ParsedChat.parse(serialized)
        re_serialized = parsed.serialize()
        assert "this is a test" in re_serialized

    def test_new_with_lang(self) -> None:
        from batchalign.compat import Document

        doc = Document.new("hola mundo .", lang="spa")
        serialized = doc.serialize()
        assert "spa" in serialized

    def test_new_empty(self) -> None:
        from batchalign.compat import Document

        doc = Document.new()
        serialized = doc.serialize()
        assert "@Languages" in serialized

    def test_validate(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_MINIMAL_CHAT)
        errors = doc.validate()
        assert isinstance(errors, list)


class TestSubscriptAccess:
    """Document[i] and Document[i][j] subscript access with morphology."""

    def test_utterance_count(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_TWO_UTTS)
        assert len(doc) == 2

    def test_utterance_speaker(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_TWO_UTTS)
        assert doc[0].speaker == "CHI"
        assert doc[1].speaker == "MOT"

    def test_word_access(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_WITH_MOR)
        utt = doc[0]
        assert len(utt) == 3  # "the", "dog", "runs" (terminator not a word)
        assert utt[0].text == "the"
        assert utt[1].text == "dog"
        assert utt[2].text == "runs"

    def test_morphology_present(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_WITH_MOR)
        word = doc[0][1]  # "dog"
        assert word.morphology is not None
        assert word.morphology.pos == "n"
        assert word.morphology.lemma == "dog"
        assert "n|dog" in word.morphology.mor

    def test_pos_shortcut(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_WITH_MOR)
        assert doc[0][0].pos == "det"
        assert doc[0][1].pos == "n"
        assert doc[0][2].pos == "v"

    def test_lemma_shortcut(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_WITH_MOR)
        assert doc[0][0].lemma == "the"
        assert doc[0][1].lemma == "dog"
        assert doc[0][2].lemma == "run"

    def test_gra_present(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_WITH_MOR)
        word = doc[0][1]  # "dog" — should be SUBJ
        assert len(word.gra) == 1
        assert word.gra[0].relation == "SUBJ"
        assert word.gra[0].head == 3  # head is "runs"

    def test_no_morphology(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_MINIMAL_CHAT)
        word = doc[0][0]  # "hello" — no %mor tier
        assert word.morphology is None
        assert word.pos is None
        assert word.lemma is None
        assert word.gra == []

    def test_iteration(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_WITH_MOR)
        words = [w.text for w in doc[0]]
        assert words == ["the", "dog", "runs"]

    def test_document_iteration(self) -> None:
        from batchalign.compat import Document

        doc = Document._from_text(_CHAT_TWO_UTTS)
        speakers = [utt.speaker for utt in doc]
        assert speakers == ["CHI", "MOT"]


class TestBatchalignPipeline:
    """BatchalignPipeline construction (execution not tested — requires CLI)."""

    def test_new_single_task(self) -> None:
        from batchalign.compat import BatchalignPipeline

        nlp = BatchalignPipeline.new("morphosyntax", lang="eng")
        assert nlp._tasks == ["morphosyntax"]
        assert nlp._lang == "eng"

    def test_new_multi_task(self) -> None:
        from batchalign.compat import BatchalignPipeline

        nlp = BatchalignPipeline.new("asr,morphosyntax,fa", lang="eng")
        assert nlp._tasks == ["asr", "morphosyntax", "fa"]

    def test_unknown_task_raises(self) -> None:
        from batchalign.compat import BatchalignPipeline, Document

        nlp = BatchalignPipeline.new("nonexistent_task")
        doc = Document.new("test .")
        with pytest.raises(ValueError, match="Unknown task"):
            nlp(doc)
