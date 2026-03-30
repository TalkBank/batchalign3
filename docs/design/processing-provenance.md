# Processing Provenance via @Comment Headers

**Status:** Draft
**Last updated:** 2026-03-29

## Goal

Every batchalign3 command that creates or modifies a CHAT file records
what it did in a structured `@Comment` header. This provides:

- **Reproducibility:** anyone can see exactly what tool, engine, and
  options produced the output
- **Auditability:** a processing history accumulates as multiple
  commands are applied
- **Debugging:** when output is wrong, the comment says which engine
  version and options were used

## Format

Each batchalign3 processing step adds one `@Comment` line:

```
@Comment:	[ba3 <command> | <key>=<value> ; <key>=<value> ; ... | <ISO-8601 timestamp>]
```

The `[ba3 ... ]` bracketing makes batchalign3 comments machine-parseable
and visually distinct from user comments. The prefix `ba3` is short,
unique, and greppable.

### Examples

```
@Comment:	[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
@Comment:	[ba3 align | fa=whisper-fa-large-v2 ; utr=rev ; wor=true | 2026-03-29T19:15:00-04:00]
@Comment:	[ba3 transcribe | asr=rev ; diarize=false ; lang=eng | 2026-03-29T20:00:00-04:00]
@Comment:	[ba3 utseg | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T20:30:00-04:00]
@Comment:	[ba3 translate | engine=googletrans-v1 ; lang=spa | 2026-03-29T21:00:00-04:00]
@Comment:	[ba3 coref | engine=stanza ; lang=eng | 2026-03-29T21:30:00-04:00]
@Comment:	[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng ; incremental=true | 2026-03-29T22:00:00-04:00]
```

### Per-Command Key-Value Pairs

Only options that affect output semantics are recorded. Runtime options
(workers, timeout, batch-window, verbose, server URL) are omitted.

#### morphotag

| Key | Value | Example |
|-----|-------|---------|
| `engine` | Stanza version from worker | `stanza-1.11.1` |
| `lang` | Language code | `eng` |
| `retokenize` | Whether CJK retokenization was applied | `true` |
| `incremental` | Whether `--before` diff was used | `true` |

#### align

| Key | Value | Example |
|-----|-------|---------|
| `fa` | Forced alignment engine+version | `whisper-fa-large-v2`, `wav2vec` |
| `utr` | UTR engine (if used) | `rev`, `whisper`, `none` |
| `wor` | Whether %wor tier was written | `true`, `false` |
| `lang` | Language code | `eng` |
| `incremental` | Whether `--before` diff was used | `true` |

#### transcribe

| Key | Value | Example |
|-----|-------|---------|
| `asr` | ASR engine | `rev`, `whisper`, `tencent`, `aliyun`, `funaudio` |
| `diarize` | Speaker diarization | `true`, `false` |
| `lang` | Language code | `eng`, `yue` |
| `wor` | Whether %wor tier was written | `true`, `false` |

#### utseg

| Key | Value | Example |
|-----|-------|---------|
| `engine` | Stanza version | `stanza-1.11.1` |
| `lang` | Language code | `eng` |

#### translate

| Key | Value | Example |
|-----|-------|---------|
| `engine` | Translation engine | `googletrans-v1`, `seamless` |
| `lang` | Source language | `spa` |

#### coref

| Key | Value | Example |
|-----|-------|---------|
| `engine` | Coref engine | `stanza` |
| `lang` | Language code | `eng` |

## Behavior Rules

### Accumulation

Multiple commands produce multiple `@Comment` lines, creating a
processing history:

```
@Comment:	[ba3 transcribe | asr=whisper ; lang=eng | 2026-03-29T18:00:00-04:00]
@Comment:	[ba3 morphotag | engine=stanza-1.11.1 ; lang=eng | 2026-03-29T18:30:00-04:00]
@Comment:	[ba3 align | fa=whisper-fa-large-v2 ; utr=rev | 2026-03-29T19:00:00-04:00]
```

### Replacement on re-run

When the same command is run again on the same file, the previous
`@Comment` for that command is **replaced**, not duplicated. Detection
is by prefix: any existing `@Comment` matching `[ba3 morphotag |` is
replaced when morphotag runs again.

### Placement

Provenance comments are placed after the last `@ID` header and before
the first utterance, grouped with other `@Comment` lines.

### Parsing

To extract provenance programmatically:

```python
import re
PROVENANCE_RE = re.compile(
    r'^\[ba3 (\w+) \| (.*?) \| (\S+)\]$'
)
# Groups: (1) command, (2) key=value pairs, (3) timestamp
```

### Legacy

The existing transcribe comment format:

```
@Comment:	Batchalign 0.1.0, ASR Engine rev. Unchecked output of ASR model.
```

will be replaced by the new structured format. During the transition,
both may appear. The old format has no `[ba3` prefix and is not
machine-parseable.

## Implementation

### Data Model

```rust
/// Processing provenance to inject as @Comment.
pub struct ProvenanceComment {
    /// Command name (e.g., "morphotag", "align").
    pub command: ReleasedCommand,
    /// Key-value pairs for semantic options.
    pub fields: BTreeMap<String, String>,
    /// Timestamp of processing (UTC, formatted as ISO 8601).
    pub timestamp: chrono::DateTime<chrono::Local>,
}
```

### Where to inject

Each dispatch pipeline (morphosyntax, FA, transcribe, utseg, translate,
coref) constructs a `ProvenanceComment` from the engine versions
reported by the worker and the command options. The comment is injected
into the CHAT AST during the serialization step, right before
`to_chat_string()`.

The injection function:
1. Parses the CHAT AST
2. Finds existing `[ba3 <command> |` comments and removes them
3. Inserts the new provenance comment after the last `@ID`
4. Serializes

### Engine version source

Engine versions come from `WorkerCapabilities.engine_versions` — the
live detection map reported by workers at spawn time. This is already
available in `DispatchHostContext` and stored in the health endpoint.

Example values from a real worker:
```json
{
  "morphosyntax": "1.11.1",
  "fa": "whisper-fa-large-v2",
  "asr": "rev",
  "utseg": "1.11.1",
  "coref": "stanza",
  "translate": "googletrans-v1"
}
```

## Future Extensions

- `[ba3 validate | chatter=0.5.0 ; errors=0 ; warnings=3 | ...]` —
  validation provenance
- `[ba3 retokenize | engine=pycantonese ; lang=yue | ...]` — if
  retokenize becomes a standalone command
- `host=net` — which machine processed the file (useful for fleet
  debugging, but might be noise for external users)
