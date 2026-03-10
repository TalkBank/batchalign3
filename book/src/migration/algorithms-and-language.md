# Algorithms, Language, and Alignment Migration

**Status:** Current
**Last updated:** 2026-03-15

Comparison anchors:

- `batchalign2` baseline `84ad500b09e52a82aca982c41a8ccd46b01f4f2c` (2026-01-09)
- later released `batchalign2` master-branch point
  `e8f8bfada6170aa0558a638e5b73bf2c3675fe6d` (2026-02-09) where relevant
- current `batchalign3`

This page documents durable algorithmic and data-structure changes only.
Temporary migration-branch experiments do not belong here.

For user-facing command and output consequences, start with
[User Workflow Migration](user-migration.md). This page explains the mechanism
behind those differences.

## Comparison discipline

Algorithmic/output parity claims in this page are anchored to the correct Jan 9
baseline for the material under test:

- core / non-HK claims: Jan 9 `batchalign2-master`
- HK / Cantonese claims: Jan 9 `BatchalignHK`

- Later Feb 9 BA2 behavior can still be useful secondary evidence when the
  question is specifically about the last released BA2 master branch.
- later Python operational builds are not the migration baseline for the
  algorithmic claims documented here.

Use the repo comparison harnesses with the historically correct
`84ad500...`-pinned runner when validating these deltas in practice. For HK
material, that means `batchalignhk`, not stock `batchalign`.

## 1) CHAT parser/validator/AST/serialization as the central algorithmic change

The most important migration is architectural and algorithmic: parsing and
validating CHAT is now a typed, structured pipeline. This reduces failure modes
caused by line-oriented text surgery and makes downstream alignment and `%mor`/`%gra`
operations operate on stable structure.

Implication for contributors: if a change can be expressed as AST transform, do
that first; avoid direct string hacking.

## 1.1) Why this changed correctness, not just implementation language

The main gain is not "Rust is faster." The gain is that parsing, validation,
injection, and serialization now operate on stable typed structure.

That directly changes user-visible correctness:

- fewer opportunities for `%mor`/`%gra` drift from line-oriented text surgery,
- fewer silent remap choices after tokenization/alignment divergence,
- stronger validation before invalid CHAT is written back out.

This is a fundamental redesign, not incremental cleanup: ad-hoc string
manipulation and parallel-array patching are replaced by principled typed
structure with explicit provenance — and that structural shift is what drives
both the correctness and efficiency gains throughout the migration.

That pattern shows up repeatedly across the migration:

- UTR uses global Hirschberg DP alignment (the same proven approach as old
  batchalign, now in Rust),
- FA now carries explicit word identity and timing-mode metadata,
- retokenization uses deterministic range/index mapping and AST rebuilds,
- `%gra` generation uses explicit chunk/head validation rather than positional
  guesswork.
- `utseg` now treats constituency parsing and assignment computation as separate
  steps instead of flattening subtree leaves and DP-aligning them back to forms.
- `coref` now carries typed sentence/chain structure instead of detokenized text
  plus DP remap back to utterance positions.

For migration purposes, separate:

| Stage | Algorithm/data-structure shift |
|---|---|
| Jan 9 BA2 -> Feb 9 BA2 | released BA2 already improved DP behavior, caching boundaries, and a number of robustness/performance details inside the Python architecture |
| Feb 9 BA2 -> current BA3 | current code goes further by moving key remap/injection/validation logic into Rust-owned chat-ops and typed orchestration |

## 2) Dynamic programming: what was removed, what remains

### Narrowed or removed from runtime remap paths

- Retokenize char-level DP fallback mapping path was removed, replaced by
  deterministic interval/index mapping with length-aware monotonic fallback.
- FA response handling uses indexed word timings or deterministic token
  stitching in `fa/alignment.rs` — no DP.

For FA, the precise current claim is narrower:

- current Rust FA response handling uses indexed word timings or
  deterministic token stitching in `fa/alignment.rs`;
- it does not use the Jan 9 / Feb 9 BA2 broad transcript-wide remap policy;
- a shared Hirschberg DP library still exists in-tree, so the accurate migration
  claim is that runtime remap policy was narrowed, not that every DP use
  disappeared.

### UTR: global DP is the steady-state correctness boundary

UTR timing recovery (`fa/utr.rs`) uses a single global Hirschberg DP alignment
of all document words against all ASR tokens.

This is the correct steady-state algorithm for the 407-style hand-edited
transcript case: transcript words and ASR tokens are two independent
full-document sequences, so the matcher must reason globally rather than
utterance by utterance. The Rust Hirschberg implementation is O(mn) time,
O(min(m,n)) space, and runs 10-50x faster than the old Python implementation.

This fixes token-starvation failures where local matching consumed tokens too
early. It does not solve every alignment pathology: dense `&*` overlap and
larger text/audio order divergence still remain a limitation of any monotonic
aligner.

### Intentionally retained

- model-internal alignment/decoding internals (e.g., CTC/Whisper internals)
- evaluation/edit-distance style metrics (WER and similar analysis tooling)

Policy: runtime user-output remap should not silently reintroduce global DP tie
ambiguity in paths where deterministic mapping is available.

This is a durable migration boundary. DP is legitimate when aligning two
genuinely independent sequences (UTR: transcript words vs ASR tokens; WER:
hypothesis vs reference). It is a regression when runtime output reconstruction
uses global DP to paper over mismatches that should be handled by deterministic
identity/index mapping (retokenization, FA injection, `%mor`/`%gra` attachment).

## 3) Realign-after-edit consequences

When transcript edits occur after initial alignment:

- deterministic ID/index matching preserves timing slots where provenance remains,
- bounded window policies prevent cross-utterance remap jumps,
- unresolved ambiguity yields explicit unassigned outcomes (not hidden remap).

This is the intended operational tradeoff: transparent uncertainty over unstable
auto-corrections.

For migrators, this means some BA2-to-BA3 output differences are expected:

- Jan 9 / Feb 9 BA2 output may have looked "more complete" because ambiguous words were forced
  into a global remap anyway;
- current BA3 output may leave some timing/provenance unresolved explicitly;
- that is a correctness choice, not a missing feature.

### 3.1) `align` improvements

Feb 9 BA2 already improved cache use, DP edge cases, and FA failure handling.
BA3 goes further by moving FA grouping, timing injection, `%wor`,
monotonicity, and overlap cleanup into Rust orchestration with typed FA
payloads and deterministic transfer rules.

One especially important `align` sub-change is UTR:

- released BA2 recovered utterance timing via a single global Hirschberg DP
  alignment of all transcript words against all ASR tokens;
- BA3 now uses the same global Hirschberg approach (in Rust), preserving the
  correct full-document alignment model while moving the implementation onto
  the typed Rust chat-ops boundary.
- This fixes the 407-style token-starvation regression class, but it does not
  make UTR non-monotonic. Files with dense overlap and text/audio reordering
  can still remain only partially recoverable.

## 4) Retokenization and Stanza multi-token outputs

Batchalign3 accounts for multi-word token expansion and tokenization divergence
with deterministic interval/index mapping logic, preserving monotonic ordering.

Practical outcomes:

- one source token yielding multiple UD tokens is handled through explicit mapping,
- merged/split forms are attached by deterministic policy rather than global
  string-level DP reconciliation,
- divergence remains visible and testable in golden fixtures.

This directly addresses the "multiple tokens from Stanza" migration concern:
token expansion is treated as structured provenance mapping, not as text that
must later be globally realigned by DP.

## 4.1) Morphotag and `%gra` correctness consequences

Durable migration-relevant correctness changes include:

- `%gra` root attachment now follows standard root-head semantics instead of
  self-referential root indices;
- reflexive pronoun suffix handling was corrected;
- MWT chunk mapping avoids brittle positional assumptions;
- tokenizer-generated divergence is either deterministically mapped back or left
  explicit, rather than silently "fixed" by a global text remap.

Concrete currently tested consequences include:

- ROOT must attach to virtual root `0`, not to itself;
- invalid root/head/chunk-count combinations are rejected;
- MWT expansions produce per-component `%gra` relations;
- `@c` and `@s` special forms are mapped explicitly rather than relying on
  placeholder leakage;
- `xbxxx` placeholders are restored back to the original form in retokenized
  output;
- reflexive pronouns explicitly emit `reflx`;
- retokenization can split contractions structurally, while `retokenize=false`
  preserves original tokenization.

Important comparison nuance:

- reflexive `reflx`, special-form handling, and `xbxxx` restoration were not
  invented only in current BA3; older BA2 already had versions of those
  behaviors in Python `ud.py`;
- the more durable current shift is that ROOT/head/chunk semantics and
  retokenization behavior are now enforced through explicit mapping logic and
  tests rather than left to positional array repair.

For corpus maintainers, this means BA3 `%mor`/`%gra` diffs against BA2 should be
reviewed as likely corrections first.

## 4.2) From positional repair to principled indexing

The durable algorithmic shift is away from workflows like:

- flatten text,
- keep parallel arrays,
- patch indices after skips/merges,
- run broad DP when the arrays drift.

Toward workflows like:

- carry stable word identity,
- maintain explicit utterance/word/chunk indices,
- iterate AST content directly,
- use deterministic local fallback only where provenance is missing.

Concrete command-level examples:

- `transcribe`:
  - BA2 Python built transcript structure while it was still normalizing token
    strings and punctuation;
  - BA3 separates tagged raw ASR payloads from Rust normalization, Rust
    postprocess, and Rust CHAT assembly.
- `translate`:
  - BA2 translated utterance strings and then relied on Python generation to
    materialize output tiers;
  - BA3 extracts utterance payloads from the AST and injects `%xtra` back by
    line index.
- `utseg`:
  - BA2 flattened constituency subtrees to strings, aligned them back to form
    arrays, then rebuilt utterances;
  - BA3 returns raw tree strings, computes assignment vectors, then splits AST
    utterances by index.
- `coref`:
  - BA2 flattened the document to one detokenized string and DP-mapped chain
    payloads back to `(utterance, form)` slots;
  - BA3 uses sentence arrays and typed chain refs, then injects sparse
    `%xcoref` by validated sentence/line mapping.

This matters because it reduces accidental correctness:

- fewer outputs that "look plausible" only because a later repair pass guessed
  the intended alignment,
- more outputs whose correctness follows from preserved structure and validated
  index relationships.

## 4.3) Cantonese / HK ASR tokenization

HK / Cantonese parity has one additional algorithmic wrinkle that deserves to
be called out explicitly.

- Cantonese material must be compared against the Jan 9 `BatchalignHK` baseline.
- The relevant preserved legacy command is `batchalignhk`.
- For `yue`, semantically correct ASR text can still benchmark badly if the
  runtime keeps long Han-script chunks as one giant token instead of splitting
  them into character tokens before retokenization and scoring.

Current `batchalign3` now handles this in the Rust-owned ASR post-process path:

- Cantonese text is normalized to HK traditional form,
- Han-script `yue` ASR chunks are split with the shared
  `cantonese_char_tokens()` helper,
- ASCII/code-switched tokens are left intact,
- punctuation-based utterance retokenization then runs on those normalized
  tokens.

This matters because WER/compare behavior for Cantonese is sensitive to token
granularity. A transcript can be visibly "close" while still scoring as a large
regression if the main path presents only a few giant tokens to the scorer.

## 5) Japanese and multilingual preprocessing/postprocessing

### Japanese verb-form and POS overrides

`nlp/lang_ja.rs` (460 lines, ported from Python `ja/verbforms.py`) applies 50+
ordered override rules that run before UD→CHAT POS mapping. These correct Stanza
outputs for colloquial Japanese forms that the model frequently misclassifies:

- **Subordinating conjunctions:** contracted conditionals (ちゃ→ば, なきゃ,
  じゃ→ちゃ, たら, たっ, で) reclassified from VERB/AUX to SCONJ.
- **Auxiliary verbs:** colloquial endings (れる→られる, よう→おう, だら→たら,
  だ→た, 無い→ない, せる→させる, なさい→為さい) with corrected lemmas.
- **Interjections:** backchannels and fillers (はい, うん, おっ, ほら, ヤッホー,
  ただいま) reclassified from NOUN/VERB to INTJ.
- **Verb lemma corrections:** specific kanji verbs (撮る, 貼る, 混ぜる, 釣る,
  降りる/降る, 載せる, 帰る, 舐める, etc.) that Stanza assigns wrong lemmas.
- **Noun/pronoun fixes:** colloquial forms (あたし→PRON, バツ, ブラシ,
  引き出し, クシャミ) and onomatopoeia (ゴロンっ, モチーンっ).

### Japanese Stanza processor configuration

Japanese requires the `combined` Stanza processor package (tokenize+pos+lemma+depparse
in one model), not separate processors. This is enforced in
`test_stanza_config_parity.py` to prevent misconfiguration that causes silent
accuracy degradation.

### Multilingual safeguards

- Multilingual routing and normalization safeguards for code-switch contexts.
- Stronger consistency between preprocessing and serializer outputs.

## 5.1) Performance consequences of the algorithm shift

The performance wins that belong in migration documentation are the durable ones:

- deterministic mapping avoids some expensive reconstruction work that used to
  happen after engine calls;
- better cache boundaries mean repeated morphosyntax/alignment work is skipped
  more often;
- batching and warm workers reduce per-file startup overhead.

Point-in-time benchmark spikes do not belong here unless they became a durable
property of the released architecture.

## 6) Overlap and rapid interleaving speech

A known limit remains: rapidly overlapping/interleaving speaker turns are still
hard for perfect automatic assignment. The migration improves this by local
window constraints and deterministic fallback, but does not claim complete
disambiguation in all overlap-heavy audio.

Mitigation strategy:

- preserve all candidate structure and timings,
- avoid global crossing remaps,
- expose unresolved slots for explicit review tools.

## 7) Regression governance

Algorithmic migrations are now defended by:

- perturbation and golden test matrices
  (`batchalign/tests/test_dp_broad_validation.py`, `batchalign/tests/golden/`),
- no-DP-runtime allowlist tests
  (`batchalign/tests/test_dp_allowlist.py` — 3 tests: Rust PyO3 call sites,
  chat-ops call sites, Python inference zero-DP),
- corpus-level A/B validation tests
  (`test_dp_broad_validation.py` — metrics computed in-memory per test run),
- tracing instrumentation for mapping-mode divergence
  (`retokenize/mapping.rs` line 64: `warn!` when falling back to
  length-aware monotonic mapping).

That governance change is itself part of the migration: the codebase is less
willing to accept "looks plausible on a few files" as evidence that an
algorithmic rewrite is safe.
