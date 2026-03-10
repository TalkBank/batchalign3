"""Batchalign2 compatibility shim.

Wraps BA3's Rust-backed API in BA2-style classes so that code written against
the old ``CHATFile``, ``Document``, ``BatchalignPipeline``, and subscript-access
surfaces (``doc[0][0].morphology``) continues to work during the transition
period.

**This module is deprecated.** New code should use ``batchalign_core.ParsedChat``
directly. See the Python API docs and the migration guide for the recommended
replacements.

Behavioral differences from BA2
-------------------------------
- ``BatchalignPipeline`` delegates to the ``batchalign3`` CLI subprocess.
  The first invocation may start a background daemon process that keeps ML
  models warm in memory. This daemon persists after the Python process exits.
  Stop it explicitly with ``batchalign3 serve stop``.
- Analysis results are cached in a local SQLite database
  (``~/.cache/batchalign3/``). Re-processing identical input returns cached
  results instantly. Clear with ``batchalign3 cache clear``.
- ML models are downloaded on first use (~2 GB). Subsequent runs use cached
  models.
- Individual engine classes (``WhisperEngine``, ``StanzaEngine``, etc.) are
  not provided. Use CLI commands or ``run_pipeline()`` instead.
- ``TextGridFile`` is not shimmed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import batchalign_core

warnings.warn(
    "batchalign.compat is deprecated. "
    "Use batchalign_core.ParsedChat directly. "
    "See: book/src/user-guide/python-api.md",
    DeprecationWarning,
    stacklevel=2,
)

# Task string → CLI command mapping for BatchalignPipeline.
_TASK_TO_COMMAND: dict[str, str] = {
    "asr": "transcribe",
    "morphosyntax": "morphotag",
    "fa": "align",
    "align": "align",
    "translate": "translate",
    "utseg": "utseg",
}


# ---------------------------------------------------------------------------
# Subscript wrapper classes (read-only views over Rust AST data)
# ---------------------------------------------------------------------------


class GrammaticalRelation:
    """A single dependency edge from the %gra tier.

    Attributes
    ----------
    index:
        1-based position of this word in the %mor chunk list.
    head:
        1-based position of the head word (0 = ROOT).
    relation:
        Dependency relation label (e.g., ``"SUBJ"``, ``"ROOT"``, ``"OBJ"``).
    """

    __slots__ = ("index", "head", "relation")

    def __init__(self, index: int, head: int, relation: str) -> None:
        self.index = index
        self.head = head
        self.relation = relation

    def __repr__(self) -> str:
        return f"{self.index}|{self.head}|{self.relation}"


class Morphology:
    """Per-word morphological annotation from the %mor tier.

    Attributes
    ----------
    mor:
        Full %mor notation for this word (e.g., ``"pro:sub|it~aux|be&3S"``).
    pos:
        Part-of-speech tag from the main MorWord (e.g., ``"pro:sub"``).
    lemma:
        Lemma from the main MorWord (e.g., ``"it"``).
    gra:
        Grammatical relations from %gra for this word's chunks.
    """

    __slots__ = ("mor", "pos", "lemma", "gra")

    def __init__(
        self,
        mor: str,
        pos: str,
        lemma: str,
        gra: list[GrammaticalRelation],
    ) -> None:
        self.mor = mor
        self.pos = pos
        self.lemma = lemma
        self.gra = gra

    def __repr__(self) -> str:
        return self.mor


class Word:
    """A single word in an utterance.

    Supports BA2-style attribute access: ``word.text``, ``word.morphology``,
    ``word.pos``, ``word.lemma``.

    Attributes
    ----------
    text:
        Cleaned word text (CHAT markers removed).
    morphology:
        ``Morphology`` object if %mor tier is present, else ``None``.
    """

    __slots__ = ("text", "morphology")

    def __init__(self, text: str, morphology: Morphology | None) -> None:
        self.text = text
        self.morphology = morphology

    @property
    def pos(self) -> str | None:
        """Part-of-speech tag (shortcut for ``word.morphology.pos``)."""
        return self.morphology.pos if self.morphology else None

    @property
    def lemma(self) -> str | None:
        """Lemma (shortcut for ``word.morphology.lemma``)."""
        return self.morphology.lemma if self.morphology else None

    @property
    def gra(self) -> list[GrammaticalRelation]:
        """Grammatical relations (shortcut for ``word.morphology.gra``)."""
        return self.morphology.gra if self.morphology else []

    def __repr__(self) -> str:
        if self.morphology:
            return f"Word({self.text!r}, {self.morphology!r})"
        return f"Word({self.text!r})"


class Utterance(Sequence[Word]):
    """A single utterance (main tier line) in a CHAT document.

    Supports BA2-style subscript access: ``utt[0]`` returns a ``Word``,
    ``len(utt)`` returns word count, iteration yields words.

    Attributes
    ----------
    speaker:
        Speaker code (e.g., ``"CHI"``, ``"MOT"``).
    words:
        List of ``Word`` objects.
    """

    __slots__ = ("speaker", "words")

    def __init__(self, speaker: str, words: list[Word]) -> None:
        self.speaker = speaker
        self.words = words

    def __getitem__(self, index: int) -> Word:  # type: ignore[override]
        return self.words[index]

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self) -> Iterator[Word]:
        return iter(self.words)

    def __repr__(self) -> str:
        return f"Utterance({self.speaker!r}, {len(self.words)} words)"


def _build_utterances(parsed: batchalign_core.ParsedChat) -> list[Utterance]:
    """Extract structured utterances from a ParsedChat via the Rust AST."""
    raw = json.loads(parsed.extract_document_structure())
    utterances: list[Utterance] = []
    for utt_json in raw:
        words: list[Word] = []
        for w_json in utt_json["words"]:
            morphology: Morphology | None = None
            if w_json.get("mor") is not None:
                gra_list = [
                    GrammaticalRelation(g["index"], g["head"], g["relation"])
                    for g in w_json.get("gra", [])
                ]
                morphology = Morphology(
                    mor=w_json["mor"],
                    pos=w_json["pos"],
                    lemma=w_json["lemma"],
                    gra=gra_list,
                )
            words.append(Word(text=w_json["text"], morphology=morphology))
        utterances.append(Utterance(speaker=utt_json["speaker"], words=words))
    return utterances


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class Document(Sequence[Utterance]):
    """BA2-compatible Document wrapper around ``ParsedChat``.

    Supports ``serialize()``, ``validate()``, subscript access
    (``doc[0]`` for utterances, ``doc[0][0]`` for words), and iteration.
    """

    def __init__(self, text: str, parsed: batchalign_core.ParsedChat) -> None:
        self._text = text
        self._parsed = parsed
        self._utterances: list[Utterance] | None = None

    def _ensure_utterances(self) -> list[Utterance]:
        """Lazily build utterance wrappers on first subscript access."""
        if self._utterances is None:
            self._utterances = _build_utterances(self._parsed)
        return self._utterances

    @classmethod
    def _from_text(cls, chat_text: str) -> Document:
        """Internal: create from raw CHAT text.

        Uses lenient parsing to tolerate BA2-era files that may lack
        ``@UTF8`` or ``@Begin`` headers.
        """
        parsed = batchalign_core.ParsedChat.parse_lenient(chat_text)
        return cls(chat_text, parsed)

    @classmethod
    def new(
        cls,
        text: str | None = None,
        *,
        media_path: str | None = None,
        lang: str = "eng",
    ) -> Document:
        """Create a minimal CHAT document from plain text.

        Parameters
        ----------
        text:
            Plain transcript text (one utterance). If ``None``, creates an
            empty document.
        media_path:
            Optional media file path to include in the header.
        lang:
            Three-letter language code (default ``"eng"``).
        """
        utterances: list[dict[str, object]] = []
        if text is not None:
            utterances.append({
                "speaker": "PAR0",
                "words": [
                    {"text": w, "start_ms": None, "end_ms": None}
                    for w in text.split()
                ],
            })

        transcript: dict[str, object] = {
            "langs": [lang],
            "participants": [
                {"id": "PAR0", "name": "Participant", "role": "Participant"}
            ],
            "utterances": utterances,
        }

        if media_path is not None:
            transcript["media_name"] = Path(media_path).stem
            transcript["media_type"] = "audio"

        chat_text = batchalign_core.build_chat(json.dumps(transcript))

        parsed = batchalign_core.ParsedChat.parse(chat_text)
        return cls(chat_text, parsed)

    def serialize(self) -> str:
        """Serialize to CHAT text."""
        return self._parsed.serialize()

    def validate(self) -> list[str]:
        """Validate and return a list of error strings."""
        return self._parsed.validate()

    @property
    def transcript(self) -> str:
        """Return the original text used to create this document."""
        return self._text

    # --- Sequence protocol (subscript access) ---

    def __getitem__(self, index: int) -> Utterance:  # type: ignore[override]
        return self._ensure_utterances()[index]

    def __len__(self) -> int:
        return len(self._ensure_utterances())

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self._ensure_utterances())


# ---------------------------------------------------------------------------
# CHATFile
# ---------------------------------------------------------------------------


class CHATFile:
    """BA2-compatible CHATFile for reading and writing ``.cha`` files.

    Parameters
    ----------
    path:
        Path to a ``.cha`` file to read.
    doc:
        An existing ``Document`` to wrap.

    Provide exactly one of ``path`` or ``doc``.
    """

    def __init__(
        self,
        *,
        path: str | Path | None = None,
        doc: Document | None = None,
    ) -> None:
        if path is not None and doc is not None:
            msg = "Provide either path or doc, not both"
            raise ValueError(msg)
        if path is not None:
            text = Path(path).read_text("utf-8")
            self._doc = Document._from_text(text)
        elif doc is not None:
            self._doc = doc
        else:
            msg = "Provide either path or doc"
            raise ValueError(msg)

    @property
    def doc(self) -> Document:
        """Access the wrapped Document."""
        return self._doc

    def write(self, path: str | Path) -> None:
        """Write the document to a ``.cha`` file."""
        Path(path).write_text(self._doc.serialize(), "utf-8")


# ---------------------------------------------------------------------------
# BatchalignPipeline
# ---------------------------------------------------------------------------


class BatchalignPipeline:
    """BA2-compatible pipeline that delegates to the ``batchalign3`` CLI.

    This shim shells out to the ``batchalign3`` binary for each task.
    It does NOT run inference in-process.

    **Behavioral note:** The CLI may start a background daemon process that
    keeps ML models warm. This daemon persists after the Python process exits.
    Stop it with ``batchalign3 serve stop``.

    Parameters
    ----------
    tasks:
        List of task strings (``"asr"``, ``"morphosyntax"``, ``"fa"``, etc.).
    lang:
        Three-letter language code.
    num_speakers:
        Number of speakers (used for ASR/diarization).
    """

    def __init__(
        self,
        tasks: list[str],
        lang: str = "eng",
        num_speakers: int = 2,
    ) -> None:
        self._tasks = tasks
        self._lang = lang
        self._num_speakers = num_speakers

    @classmethod
    def new(
        cls,
        tasks: str,
        lang: str = "eng",
        num_speakers: int = 2,
    ) -> BatchalignPipeline:
        """Create a pipeline from a comma-separated task string.

        Example::

            nlp = BatchalignPipeline.new("morphosyntax", lang="eng")
            nlp = BatchalignPipeline.new("asr,morphosyntax", lang="eng")
        """
        return cls(
            [t.strip() for t in tasks.split(",") if t.strip()],
            lang,
            num_speakers,
        )

    def __call__(self, doc_or_path: Document | str | Path) -> Document:
        """Run the pipeline on a document or file path.

        Returns a new ``Document`` with the pipeline results.
        """
        batchalign3 = shutil.which("batchalign3")
        if batchalign3 is None:
            msg = (
                "batchalign3 CLI not found on PATH. "
                "Install with: uv tool install batchalign3"
            )
            raise RuntimeError(msg)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            input_file = input_dir / "input.cha"

            # Write input file.
            if isinstance(doc_or_path, Document):
                input_file.write_text(doc_or_path.serialize(), "utf-8")
            else:
                src = Path(doc_or_path)
                input_file.write_text(src.read_text("utf-8"), "utf-8")

            # Run each task sequentially.
            current_input = input_dir
            for task in self._tasks:
                command = _TASK_TO_COMMAND.get(task)
                if command is None:
                    msg = (
                        f"Unknown task: {task!r}. "
                        f"Known tasks: {', '.join(sorted(_TASK_TO_COMMAND))}"
                    )
                    raise ValueError(msg)

                step_output = Path(tmpdir) / f"step_{task}"
                step_output.mkdir(exist_ok=True)

                cmd = [
                    batchalign3,
                    command,
                    str(current_input),
                    "-o",
                    str(step_output),
                    "--lang",
                    self._lang,
                ]

                if command == "transcribe":
                    cmd.extend(["--num-speakers", str(self._num_speakers)])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    msg = (
                        f"batchalign3 {command} failed "
                        f"(exit {result.returncode}):\n{result.stderr}"
                    )
                    raise RuntimeError(msg)

                current_input = step_output

            # Read the output file.
            output_files = list(current_input.glob("*.cha"))
            if not output_files:
                msg = "Pipeline produced no .cha output files"
                raise RuntimeError(msg)

            result_text = output_files[0].read_text("utf-8")
            return Document._from_text(result_text)
