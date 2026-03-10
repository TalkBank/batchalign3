# Tracing and Debugging

This document describes the tracing and debugging strategy across the
batchalign3 stack: Rust (batchalign-core PyO3 bridge), Rust (CLI and server
control plane), and Python (pipeline engines and worker process).

## Verbosity Levels

A single `-v` / `-vv` / `-vvv` flag on the CLI controls both Rust tracing and
Python logging across the entire stack.

| Level | Rust (`tracing`) | Python (`logging`) | When to use |
|-------|-------------------|--------------------|-------------|
| 0 (default) | `WARN` | `WARNING` | Normal operation |
| 1 (`-v`) | `INFO` | `INFO` | Server start/stop, job lifecycle |
| 2 (`-vv`) | `DEBUG` | `DEBUG` | Per-file progress, engine boundary data |
| 3 (`-vvv`) | `TRACE` | `DEBUG` | Full payload dumps (truncated) |

### How verbosity propagates

```
CLI (main.rs)
  │
  ├─ init_tracing(verbose)          ← sets Rust filter level
  │
  └─ serve_cmd::start(args, verbose)
       │
       └─ PoolConfig { verbose, .. }
            │
            └─ WorkerConfig { verbose, .. }
                 │
                 └─ python3 ... --verbose N   ← forwarded to batchalign.worker
                      │
                      └─ logging.basicConfig(level=...)
```

In background mode (`batchalign3 serve start` without `--foreground`), the
`-v` flags are forwarded to the re-exec'd background process.

## Engine Boundary Tracing

The highest-risk surface in the stack is the Rust-Python boundary where data
crosses serialization layers. This boundary is instrumented at three points:

### 1. Morphosyntax batch callback (`pyo3/src/morphosyntax_ops.rs`)

The `add_morphosyntax_batched_inner` function has three phases:

- **Phase 1** (pure Rust — extract words): `debug!` logs utterance and word
  counts extracted from the AST.
- **Phase 2** (GIL — Python callback): `debug!` logs response item count.
- **Phase 3** (pure Rust — inject results): `debug!` logs completion.

### 2. Python inference module (`batchalign/inference/morphosyntax.py`)

The `batch_infer_morphosyntax` function logs:

- Item count and elapsed time at `INFO` level on completion
- Sentence count mismatch warnings at `WARNING` level
- Stanza batch failure warnings at `WARNING` level

### 3. Worker IPC (`handle.rs`)

Worker spawn, shutdown, health checks, and IPC dispatch are logged at `info!`
and `debug!` levels. Worker stderr is captured for crash diagnostics.

## Performance

The `tracing` crate's `debug!` and `trace!` macros cost **~1–5 ns** when the
corresponding level is filtered out (the default level is `WARN`). All
instrumented functions are per-file or per-utterance, never per-word. There is
no measurable performance impact during normal operation.

Python `logging.debug()` calls are similarly inexpensive when the logger level
is `WARNING`.

## Safe AST Construction

### The problem

Raw text from NLP engines (Stanza, Whisper) must be converted to CHAT AST
nodes. Directly constructing AST nodes with `Word::new_unchecked` bypasses the
lexical validation that the parser would normally enforce, allowing malformed
words into the AST. These silently propagate until pre-serialization validation,
at which point the error is far from the root cause.

### Policy

1. **Always try `DirectParser::parse_word()` first** — if the text is valid
   CHAT syntax, the parser returns a properly validated `Word`.
2. **Only fall back to `new_unchecked` when the input is genuinely unparseable**
   (e.g., ASR returned non-CHAT characters). Log a `warn!` when this happens.
3. **Never fall back to `new_unchecked` in retokenization** — if a Stanza-split
   token can't be parsed, keep the original CHAT word unchanged.

### Implementation

Three categories of `new_unchecked` usage have been addressed:

**A. ASR transcript construction** (`lib.rs` / `build_chat_inner`):
ASR engines return raw text that must become CHAT words. The code now tries
`DirectParser::parse_word()` first and only falls back to `new_unchecked` with
a `warn!` if parsing fails.

**B. Retokenization fallback** (`retokenize.rs`):
When Stanza splits a CHAT word into MWT sub-tokens, each sub-token must be
parsed back into a CHAT `Word`. The `try_parse_token_as_word()` family of
functions return `Option<Word>` instead of always succeeding. On parse failure,
the original word is preserved (no invalid content enters the AST).

**C. Temporary scaffolding** (`lib.rs:1323`):
A temporary word used only as input to `resolve_word_language()` — never
injected into the AST. This is a documented acceptable use of `new_unchecked`.

### Injection-time alignment check (`inject.rs`)

Before injecting MOR/GRA tiers into an utterance, the code now validates that
the number of MOR items matches the number of alignable words extracted from
the AST. A mismatch is a bug — it means the extraction or NLP mapping is wrong.

```rust
// inject.rs — count alignment check
let word_count = extracted.len();
let mor_count = mors.len();
if word_count != mor_count {
    tracing::warn!(word_count, mor_count, ...);
    return Err(format!("MOR item count ({mor_count}) does not match ..."));
}
```

This catches problems at the point of injection (close to root cause) rather
than deferring to the pre-serialization validation pass.

## Debugging Workflows

### Diagnosing a morphosyntax failure

1. Run with `-vv` to see per-utterance word counts and Stanza I/O:
   ```bash
   batchalign3 -vv morphotag input/ output/
   ```

2. If a specific utterance fails, the `warn!` from `inject.rs` will report the
   exact word count mismatch and utterance text.

3. Run with `-vvv` (trace) to see the full JSON payload sent to Stanza and the
   JSON response (truncated to 500 chars).

### Diagnosing a retokenization issue

When Stanza splits a word into MWT sub-tokens and one sub-token is
unparseable:

1. A `warn!` is logged: `"Token is not valid CHAT syntax; keeping original word"`.
2. The original word is preserved in the AST.
3. The MOR cursor advances past the sub-token indices to stay in sync.

### Diagnosing an ASR construction issue

When ASR returns text that isn't valid CHAT:

1. A `warn!` is logged: `"ASR word is not valid CHAT syntax; using unchecked fallback"`.
2. The unchecked word enters the AST — this is expected for non-CHAT characters.
3. Pre-serialization validation will catch any downstream issues.

### Checking worker verbosity

To verify that verbosity reaches Python workers:

```bash
batchalign3 -vv serve start --foreground
```

Worker stderr will show `DEBUG`-level messages from `batchalign.worker` and
`batchalign.inference.morphosyntax`.

## Debug Dumps (`--debug-dir`)

The `--debug-dir PATH` flag (or `BATCHALIGN_DEBUG_DIR` env var) writes structured
CHAT/JSON artifacts at each pipeline stage:

```bash
batchalign3 align input/ output/ --lang eng --debug-dir /tmp/ba3-debug
```

This produces files like `{stem}_utr_input.cha`, `{stem}_utr_tokens.json`,
`{stem}_fa_grouping.json`, `{stem}_fa_output.cha`, etc. These artifacts enable
offline TDD — load the fixture data, call pipeline functions, and assert on
output without running ML models.

When `--debug-dir` is set, `debug_traces` is also enabled on job submissions,
populating the dashboard trace store for `GET /jobs/{id}/traces` visualization.

See [Debugging and Tracing Migration](../migration/debugging-and-tracing.md) for
the full directory layout and workflow.

## Fine-Grained Cache Overrides (`--override-cache-tasks`)

For experiment-grade control, `--override-cache-tasks` bypasses cache only for
specific NLP tasks:

```bash
# Skip UTR ASR cache but keep morphosyntax and FA caches
batchalign3 align input/ output/ --override-cache-tasks utr_asr

# Skip multiple tasks (comma-separated)
batchalign3 morphotag input/ output/ --override-cache-tasks morphosyntax,translation
```

Valid task names: `morphosyntax`, `utr_asr`, `forced_alignment`,
`utterance_segmentation`, `translation`.

The existing `--override-cache` continues to skip all cache domains.
Internally, `CacheOverrides::Tasks(BTreeSet<CacheTaskName>)` resolves per-task
at each cache call site via `policy_for(CacheTaskName)`.

## Stanza Anomaly Detection

The morphosyntax inference module (`batchalign/inference/morphosyntax.py`)
detects several classes of Stanza misbehavior:

| Anomaly | Detection |
|---------|-----------|
| Bogus lemma | Lemma is pure punctuation for a word with letters (e.g. 哎呀 → 》) |
| Sentence count mismatch | Stanza returned a different number of sentences than input utterances |
| Batch failure | Stanza raised an exception on a batch of items |

When detected, these are logged at `WARNING` level. The bogus-lemma check is
in `_is_bogus_lemma()` and triggers substitution with a `"?"` lemma rather
than propagating the bad value.
