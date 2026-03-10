# Python API

The current public Python surface is centered on two modules:

- `batchalign.pipeline_api` — thin CHAT-aware facade over Rust-owned document execution
- `batchalign.providers` — typed request/response models shared with provider
  and worker code

This is not the older `BatchalignPipeline` API. If you are migrating from that
surface, treat this page as the supported replacement entry point.

## Install the Python package

```bash
uv venv .venv
uv pip install batchalign3
```

## Parse, validate, and serialize CHAT directly

```python
import batchalign_core

doc = batchalign_core.ParsedChat.parse(chat_text)
errors = doc.validate()
output_chat = doc.serialize()
```

## Run Rust-owned pipeline steps over one document

`run_pipeline()` parses once in Rust, runs a sequence of Rust-owned document
operations, and returns the final serialized CHAT. Python only provides the raw
model callback.

```python
from batchalign.pipeline_api import (
    LocalProviderInvoker,
    PipelineOperation,
    run_pipeline,
)


def morphosyntax(lang: str, items: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for item in items:
        sentence = []
        for idx, word in enumerate(item["words"], start=1):
            sentence.append(
                {
                    "id": idx,
                    "text": word,
                    "lemma": word,
                    "upos": "NOUN",
                    "head": 0 if idx == 1 else 1,
                    "deprel": "root" if idx == 1 else "obj",
                }
            )
        rows.append({"raw_sentences": [sentence]})
    return rows


output_chat = run_pipeline(
    chat_text,
    lang="eng",
    provider=LocalProviderInvoker({"morphosyntax": morphosyntax}),
    operations=[PipelineOperation("morphosyntax")],
)
```

## Public pipeline surface

`batchalign.pipeline_api` currently exports:

- `PipelineOperation`
- `LocalProviderInvoker`
- `BatchInferProviderInvoker`
- `run_pipeline`

The operation names currently supported by `PipelineOperation` are:

- `translate`
- `morphosyntax`
- `fa`
- `utseg`
- `utr`

## Provider wire types

`batchalign.providers` re-exports the typed worker payload surface:

- `BatchInferRequest`
- `BatchInferResponse`
- `InferResponse`
- `InferTask`
- `WorkerJSONValue`

Use those types when you build provider adapters or tests around the worker
contract.

## What not to build against

- Do not build new public integrations against `batchalign.worker._*`.
- Do not assume undocumented inference modules are stable public API.
- Do not rely on the old `BatchalignPipeline`, `WhisperEngine`, or similar
  pipeline-constructor surfaces; they are not the current public contract.

## BA2 compatibility shim

A compatibility layer (`batchalign.compat`) wraps BA3's API in BA2-style
classes for code written against the old `CHATFile`, `Document`, and
`BatchalignPipeline` surfaces:

```python
from batchalign.compat import CHATFile, Document, BatchalignPipeline

# File I/O (same as BA2)
chat = CHATFile(path="input.cha")
chat.write("output.cha")

# Subscript access (same as BA2)
doc = chat.doc
utt = doc[0]            # Utterance
word = doc[0][0]        # Word
pos = word.pos           # POS tag from %mor
lemma = word.lemma       # Lemma from %mor
mor = word.morphology    # Full Morphology object

# Pipeline (delegates to batchalign3 CLI subprocess)
nlp = BatchalignPipeline.new("morphosyntax", lang="eng")
result = nlp("input.cha")
```

The module emits a `DeprecationWarning` on import. See the
[migration guide](../migration/developer-migration.md) for details and
[persistent state differences](../migration/persistent-state.md) for
behavioral changes (daemon, caching) that affect compat shim users.

## See also

- [CLI Reference](cli-reference.md)
- [Extension Layers](../architecture/extension-layers.md)
- [API Stability](../developer/api-stability.md)
- [Persistent State](../migration/persistent-state.md)
