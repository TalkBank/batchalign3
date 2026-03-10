# Language-Specific Processing Overview

This page is the single entry point for understanding how batchalign3 handles
non-English languages. It maps every stage of the processing pipeline to the
language-specific behavior at that stage.

## Pipeline Stages and Language Divergence

Every audio file flows through the same pipeline. At each stage, the pipeline
checks the language code and may take a different path:

```mermaid
flowchart TD
    input["Audio + lang code"]
    resolve["Model Resolution\n(lang → fine-tuned model)"]
    asr["ASR Transcription"]
    compound["Compound Merging"]
    numexp["Number Expansion\n(12 table langs + Chinese/Japanese)"]
    cantonorm["Cantonese Normalization\n(lang=yue only)"]
    rtlpunct["RTL Punctuation\nNormalization"]
    retok["Retokenization\n(punctuation-based)"]
    utseg["Utterance Segmentation\n(3 dedicated models)"]
    morpho["Morphosyntax\n(Stanza + per-lang workarounds)"]
    fa["Forced Alignment\n(Whisper/Wave2Vec/Cantonese FA)"]

    input --> resolve --> asr --> compound --> numexp
    numexp --> cantonorm --> rtlpunct --> retok --> utseg --> morpho --> fa

    style cantonorm fill:#f9e2b0
    style numexp fill:#d4edda
    style rtlpunct fill:#d4edda
    style morpho fill:#cce5ff
    style fa fill:#e2d5f1
```

## Where Each Language Diverges

### Stage 1: Model Resolution

Language determines which fine-tuned model is loaded for ASR. See
[Language Code Resolution](language-code-resolution.md) for the full mapping.

| Language | ASR Model | UTR Model |
|----------|-----------|-----------|
| English (eng) | `talkbank/CHATWhisper-en` | `talkbank/CHATWhisper-en-large-v1` |
| Cantonese (yue) | `alvanlii/whisper-small-cantonese` | `openai/whisper-large-v2` |
| Hebrew (heb) | `ivrit-ai/whisper-large-v3` | `openai/whisper-large-v2` |
| All others | `openai/whisper-large-v3` | `openai/whisper-large-v2` |

Only the HuggingFace `--asr-engine whisper` engine uses language-specific
resolution. `--asr-engine whisper-oai` always loads `whisper-turbo`;
`--asr-engine whisperx` always loads `whisper-large-v2`. See
[Whisper ASR](whisper-asr.md).

### Stage 2: Number Expansion

Digit strings in ASR output are converted to language-appropriate word forms.

| Language group | Method | Example |
|----------------|--------|---------|
| Mandarin (zho, cmn) | `num2chinese` (simplified) | 42 → 四十二 |
| Cantonese (yue), Japanese (jpn) | `num2chinese` (traditional) | 42 → 四十二, 10000 → 一萬 |
| 12 table languages | NUM2LANG JSON lookup | 5 → "five" (eng), "cinco" (spa), "cinq" (fra) |
| All others | Pass-through (no expansion) | 42 → "42" |

The 12 table languages: `deu`, `ell`, `eng`, `eus`, `fra`, `hrv`, `ind`,
`jpn`, `nld`, `por`, `spa`, `tha`.

See [Number Expansion](number-expansion.md) for details on the Chinese
character conversion algorithm and the table-based approach.

### Stage 3: Cantonese Text Normalization (yue only)

This stage **only activates when lang=yue**. It applies two transformations:

1. **Simplified → HK Traditional** via `zhconv` (Aho-Corasick automata from
   OpenCC + MediaWiki rulesets)
2. **31-entry domain replacement table** for Cantonese-specific character
   corrections (e.g., 系→係, 呀→啊, 中意→鍾意)

This runs in the core Rust pipeline (`batchalign-chat-ops`), **not** in a
separate plugin package. Every ASR engine's output benefits from it
automatically.

See [Cantonese Processing](cantonese-processing.md) for the full replacement
table and architecture.

### Stage 4: RTL Punctuation Normalization

Arabic/Persian/Urdu punctuation is normalized to ASCII equivalents:

| RTL | ASCII |
|-----|-------|
| ؟ | ? |
| ۔ | . |
| ، | , |
| ؛ | ; |

Additionally, Japanese full-width period (。) is normalized to `.`, and
Spanish inverted punctuation (¿, ¡) is removed.

### Stage 5: Utterance Segmentation

Three languages have dedicated BERT-based utterance segmentation models:

| Language | Model | Source |
|----------|-------|--------|
| English | `talkbank/CHATUtterance-en` | TalkBank fine-tuned |
| Mandarin | `talkbank/CHATUtterance-zh_CN` | TalkBank fine-tuned |
| Cantonese | `PolyU-AngelChanLab/Cantonese-Utterance-Segmentation` | PolyU |

All other languages fall back to **punctuation-based splitting** (`.`, `?`,
`!`, and CHAT-specific terminators like `+...`, `+/.`).

See [Utterance Segmentation](utterance-segmentation.md).

### Stage 6: Morphosyntax (Stanza + Workarounds)

Stanza is the backbone for POS tagging, lemmatization, and dependency parsing.
Language-specific workarounds correct systematic errors:

| Language | Workarounds | Reference |
|----------|-------------|-----------|
| English | 201-entry irregular-form table, contraction MWT hints, GUM package | [Non-English Workarounds](../developer/non-english-workarounds.md) §E1-E3 |
| French | 20-entry pronoun-case lookup, 158 APM noun forms, MWT overrides | §F1-F3 |
| Japanese | Order-dependent verb-form override chain, combined package, comma normalization | [Japanese Morphosyntax](japanese-morphosyntax.md), §J1-J3 |
| Hebrew | HebBinyan/HebExistential feature extraction | [Hebrew Morphosyntax](hebrew-morphosyntax.md) |
| Italian | "l'" MWT suppression, "lei" merge | §I1-I2 |
| Portuguese | "d'água" MWT forcing | §P1 |
| Dutch | Possessive "'s" MWT suppression | §D1 |

Cross-language infrastructure:

| Feature | What | Reference |
|---------|------|-----------|
| MWT dispatch table | Explicit allowlist; 39 languages currently enable MWT | §X1 |
| ISO 639-3 → 639-1 mapping | 55 explicit mappings, including yue→zh and cmn→zh | [Language Code Resolution](language-code-resolution.md) |
| Number expansion | 12 table languages + Chinese/Japanese | [Number Expansion](number-expansion.md) |

### Stage 7: Forced Alignment

| Engine | Languages | Method |
|--------|-----------|--------|
| `whisper_fa` | All (default) | Whisper large-v2 cross-attention DTW |
| `wav2vec_fa` | All | MMS FA CTC alignment |
| `wav2vec_canto` | Cantonese only | Hanzi→jyutping romanization + Wave2Vec MMS |

The Cantonese FA engine converts Chinese characters to tone-stripped jyutping
romanization before alignment, because Wave2Vec MMS was trained on romanized
text. See [Cantonese Processing](cantonese-processing.md).

## Language Code Flow

```mermaid
flowchart LR
    cli["CLI: --lang=yue"]
    iso3["ISO 639-3\n(3-letter, internal)"]
    resolve["Model resolver\n(yue → fine-tuned model)"]
    stanza["Stanza mapping\n(yue → zh)"]
    pipeline["Pipeline stages\n(yue triggers Cantonese norm)"]

    cli --> iso3
    iso3 --> resolve
    iso3 --> stanza
    iso3 --> pipeline
```

batchalign3 uses **ISO 639-3** (3-letter codes) internally everywhere.
Conversion to 2-letter codes only happens at the Stanza boundary. See
[Language Code Resolution](language-code-resolution.md).

## Cantonese Normalization Is Now Core

Older Python-only code paths did not apply Cantonese normalization uniformly
across every ASR path. Current `batchalign3` implements simplified-to-HK
conversion plus the Cantonese replacement table once in Rust core and applies
it as a shared ASR post-processing stage, so every ASR engine benefits from
the same normalization contract.

## Related Pages

- [Language Code Resolution](language-code-resolution.md) — ISO mapping, model resolution
- [Cantonese Processing](cantonese-processing.md) — normalization, char tokenization, FA
- [Hebrew Morphosyntax](hebrew-morphosyntax.md) — HebBinyan, HebExistential
- [Japanese Morphosyntax](japanese-morphosyntax.md) — verb forms, combined package
- [Number Expansion](number-expansion.md) — num2chinese, NUM2LANG tables
- [Utterance Segmentation](utterance-segmentation.md) — per-language models
- [Non-English Workarounds](../developer/non-english-workarounds.md) — workaround and convention catalog
- [Whisper ASR](whisper-asr.md) — engine selection, model IDs
- [HK/Cantonese Engines](../architecture/hk-cantonese-engines.md) — Tencent, Aliyun, FunASR
