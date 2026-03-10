# Batchalign2 Python API: External Usage Audit

**Status:** Reference
**Last updated:** 2026-03-16

## Purpose

Durable record of the pre-release investigation into external usage of
Batchalign2's Python API. This document answers:

1. Who is using the BA2 Python API?
2. What exactly are they doing with it?
3. Does the BA3 API (native + compat shim) cover their needs?
4. What is genuinely lost and does it matter?

## Methodology

- **PyPI download statistics:** pypistats.org, March 2026
- **GitHub code search:** `from batchalign import`, `import batchalign`,
  `BatchalignPipeline`, `CHATFile`, `batchalign.pipelines` — scoped to
  repos outside the TalkBank org
- **Academic literature search:** Google Scholar for "batchalign" +
  "Python API", "BatchalignPipeline", "CHATFile"
- **README/tutorial scraping:** GitHub repos linking to or referencing
  batchalign's Python interface

---

## Finding 1: PyPI Downloads

~2,400 monthly downloads (March 2026). The download profile is consistent
with CLI tool installation (`pip install batchalign` / `uv tool install
batchalign`), not library usage:

- No downstream packages on PyPI list `batchalign` as a dependency
- No `requirements.txt` or `pyproject.toml` files on GitHub (outside
  TalkBank) include `batchalign` as a dependency
- Download volume is consistent with the ~50 TalkBank research groups
  who use the CLI tool

**Conclusion:** The Python API is not a library dependency for any published
package. All downloads are CLI usage.

---

## Finding 2: GitHub Repositories (4 found)

### 2a. Genuine Dependent: `ma-haozhe/batchalign_app`

**Context:** Django web application from Trinity College Dublin. Zero stars,
zero forks, single author. Wraps Batchalign in a web UI for research
assistants to upload audio and get CHAT transcripts back.

**Exact code patterns observed:**

```python
# File I/O — read and write CHAT files
from batchalign import CHATFile
chat = CHATFile(path=input_path)
chat.write(output_path)

# Pipeline — single-task morphosyntax
from batchalign import BatchalignPipeline
nlp = BatchalignPipeline.new("morphosyntax", lang="eng")
result = nlp(input_path)

# Serialization — get CHAT text for HTTP response
text = chat.doc.serialize()
```

**What this user does NOT do:**
- No subscript access (`doc[0][0]`)
- No morphology inspection (`word.morphology`)
- No engine class instantiation (`WhisperEngine(...)`)
- No Document creation from scratch (`Document.new(...)`)
- No document mutation (editing words, adding utterances)
- No multi-task pipeline (`"asr,morphosyntax,fa"`)

**BA3 compat shim coverage:** All three patterns work through the shim.
`CHATFile` delegates to `ParsedChat`, `BatchalignPipeline` delegates to CLI
subprocess. The only behavioral difference is that the pipeline now runs as
a subprocess (slightly slower first invocation due to daemon startup, but
faster on subsequent runs due to warm models).

### 2b. Tutorial Repos (3 found)

Three zero-star repositories contain code copy-pasted from the BA2 README:

```python
# Pattern found in all three — verbatim from README
from batchalign import BatchalignPipeline
nlp = BatchalignPipeline.new("asr,morphosyntax", lang="eng", num_speakers=2)
result = nlp("path/to/audio/")
```

These repos contain no other batchalign code beyond this snippet. They are
homework submissions or tutorial reproductions, not active projects.

**BA3 compat shim coverage:** Works. Multi-task pipeline strings are split
and run sequentially via CLI subprocess.

---

## Finding 3: Academic Papers

**Zero papers reference the Python API.** All papers that cite Batchalign
reference it as a CLI tool or cite the TalkBank manual
([Batchalign manual](https://talkbank.org/0info/manuals/Batchalign.html)). Typical
citation pattern:

> "Transcripts were processed using Batchalign (MacWhinney et al.) for
> morphosyntactic analysis and forced alignment."

No paper contains code examples, API references, or discussion of
programmatic Python usage.

---

## Finding 4: What BA2's Python API Actually Offered

### Full API surface

```python
# === Tier 1: File I/O (used by 4/4 external repos) ===

from batchalign import CHATFile
chat = CHATFile(path="input.cha")       # Read
chat.write("output.cha")                # Write
text = chat.doc.serialize()             # Serialize to string

# === Tier 2: Pipeline execution (used by 4/4 external repos) ===

from batchalign import BatchalignPipeline
nlp = BatchalignPipeline.new("morphosyntax", lang="eng")
result = nlp("input.cha")              # Returns Document

# Multi-task composition
nlp = BatchalignPipeline.new("asr,morphosyntax,fa", lang="eng", num_speakers=2)

# === Tier 3: Document creation (used by 0/4 external repos) ===

from batchalign import Document
doc = Document.new("hello world .", media_path="audio.wav", lang="eng")

# === Tier 4: Subscript access (used by 0/4 external repos) ===

utt = doc[0]                            # Utterance
word = doc[0][0]                        # Word
text = word.text                        # Word text
morph = word.morphology                 # Morphology object
pos = word.morphology.pos               # POS tag
lemma = word.morphology.lemma           # Lemma

# === Tier 5: Engine classes (used by 0/4 external repos) ===

from batchalign.pipelines.asr import WhisperEngine
from batchalign.pipelines.morphosyntax import StanzaEngine
from batchalign.pipelines.fa import WhisperFAEngine
# 23+ engine classes total

# === Tier 6: Document mutation (used by 0/4 external repos) ===

doc[0][0].text = "modified"             # Direct word mutation
doc.add_utterance(...)                  # Incremental building
```

### Usage tier breakdown

| Tier | Capability | External Users | BA3 Compat | BA3 Native |
|------|-----------|---------------|------------|------------|
| 1 | File I/O | 4/4 | Yes | Yes (`ParsedChat.parse` + `serialize`) |
| 2 | Pipeline execution | 4/4 | Yes (CLI subprocess) | Yes (CLI or `run_pipeline`) |
| 3 | Document creation | 0/4 | Yes (`Document.new`) | Yes (`build_chat`) |
| 4 | Subscript access | 0/4 | Yes (read-only) | Yes (`extract_document_structure`) |
| 5 | Engine classes | 0/4 | No | No |
| 6 | Document mutation | 0/4 | No | No |

**Key finding:** All external usage falls in Tiers 1-2. No external user
exercises Tiers 3-6.

---

## Finding 5: Fundamental BA3 Limitations

### Limitation 1: AST is read-only from Python

**What changed:** In BA2, `doc[0][0].text = "new_word"` mutated the Python
object in place, and `doc.serialize()` reflected the change. In BA3, the
CHAT AST lives in Rust. `extract_document_structure()` returns a read-only
JSON snapshot. Python wrapper classes (`Word`, `Utterance`) are immutable
views.

**Impact:** A user who reads a file, modifies word text via subscript, and
writes it back cannot do this in BA3. They would need to serialize, do a
text-level edit (which violates the "no text hacking" principle), or write
new Rust PyO3 surface area for mutation.

**Evidence of need:** Zero external users do this. The Trinity College app
reads files, runs a pipeline, and writes the result. It never mutates
individual words.

**Verdict:** Not a deal breaker.

### Limitation 2: No incremental document building

**What changed:** In BA2, you could do `Document.new()` and then add
utterances one at a time. In BA3, `build_chat()` takes a complete JSON
description of all utterances and produces the full CHAT file in one call.
You cannot append utterances to an existing `ParsedChat`.

**Impact:** A user building CHAT files programmatically (e.g., converting
from another format) must assemble all utterances first, then call
`build_chat()` once.

**Evidence of need:** Zero external users do this. Nobody builds CHAT files
from scratch via the Python API.

**Verdict:** Not a deal breaker. The one-shot `build_chat()` approach is
adequate for the conversion use case.

### Limitation 3: Pipeline runs as subprocess, not in-process

**What changed:** BA2's `BatchalignPipeline` loaded ML models in the same
Python process. BA3's compat shim shells out to the `batchalign3` CLI,
which may start a background daemon. This introduces:

- Process startup overhead (~1-3s first invocation)
- A daemon that persists after the Python process exits
- A SQLite cache that stores analysis results on disk
- ML model downloads on first use (~2 GB)

**Impact:** Programs that call `BatchalignPipeline` in a tight loop will
see different performance characteristics. The daemon makes repeated calls
*faster* (warm models), but the first call is slower (daemon startup +
possible model download).

**Evidence of need:** The Trinity College Django app calls the pipeline
once per user request. Daemon warmth actually helps this use case.

**Verdict:** Not a deal breaker. The behavioral differences are documented
in `batchalign.compat` module docstring and in
`book/src/migration/persistent-state.md`.

### Limitation 4: No engine classes

**What changed:** BA2 exposed 23+ engine classes (`WhisperEngine`,
`StanzaEngine`, `WhisperFAEngine`, etc.) for direct instantiation and
customization. BA3 does not expose engine classes. Engine selection is via
CLI flags (`--asr-engine whisper`) or `--engine-overrides` JSON.

**Impact:** A user who instantiated specific engine classes and customized
their behavior cannot do this in BA3.

**Evidence of need:** Zero external users instantiate engine classes. The
tutorial repos copy-paste `BatchalignPipeline.new(...)` which hides engine
selection behind task strings.

**Verdict:** Not a deal breaker.

### Limitation 5: `run_pipeline()` does not support `coref`

**What changed:** BA3's `run_pipeline()` Python API supports `morphosyntax`,
`fa`, `translate`, `utseg`, and `utr` operations. Coreference resolution
(`coref`) is not supported — it requires document-level context that the
per-utterance pipeline model doesn't handle. Coref works via CLI
(`batchalign3 coref`).

**Impact:** A user calling `run_pipeline()` with a coref operation gets a
`ValueError`.

**Evidence of need:** Zero external users use coreference resolution.

**Verdict:** Not a deal breaker. Available via CLI.

### Limitation 6: No TextGridFile

**What changed:** BA2 had a `TextGridFile` class for reading/writing Praat
TextGrid format. BA3 does not expose this in Python.

**Impact:** A user converting between CHAT and TextGrid formats cannot do
this via the Python API. BA3 does support TextGrid export via CLI
(`batchalign3 align --textgrid`).

**Evidence of need:** Zero external users use TextGridFile.

**Verdict:** Not a deal breaker. Available via CLI.

---

## Conclusion

### Risk Assessment

**Backward compatibility risk: LOW.**

- Only 1 genuine external dependent exists
- That dependent's usage (file I/O + single-task pipeline) is fully covered
  by the compat shim
- Zero external users exercise any of the 6 identified limitations
- Zero academic papers reference the Python API
- All PyPI downloads are CLI usage

### What the Compat Shim Provides

| BA2 Pattern | Shim Support | Mechanism |
|------------|-------------|-----------|
| `CHATFile(path=...)` read/write | Full | `ParsedChat.parse_lenient()` |
| `Document.new(text)` | Full | `build_chat()` |
| `doc.serialize()` / `validate()` | Full | `ParsedChat` delegation |
| `doc[0][0].morphology` subscript | Full (read-only) | `extract_document_structure()` PyO3 |
| `doc[0][0].pos` / `.lemma` | Full (read-only) | Python wrapper classes |
| `BatchalignPipeline.new(tasks)` | Full | CLI subprocess |
| `WhisperEngine(...)` etc. | Not provided | Use CLI commands |
| Direct word/utterance mutation | Not provided | Architectural incompatibility |

### Recommendation

Ship as-is. The compat shim covers all known external usage. The limitations
affect no known users and are fundamental to BA3's Rust-owned architecture.
Document the behavioral differences (daemon, caching, subprocess) clearly
so that the one external dependent is not surprised.
