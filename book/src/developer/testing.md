# Testing

**Status:** Current
**Last updated:** 2026-03-16

Batchalign has Python tests, Rust workspace tests, PyO3 tests, and a small set
of release-hygiene checks.

Rust test binaries standardize on `cargo nextest run`, including the separate
`pyo3/` crate.

For the standard local pass, start with the repo-native entrypoints:

```bash
make test
make lint
make ci-local
```

## Python tests

```bash
uv run pytest
uv run pytest batchalign/tests/test_pipeline_api.py -v
uv run pytest batchalign/tests/test_worker_ipc.py -v
uv run pytest batchalign/tests/cli/test_cli_e2e.py -v
```

If you changed `pyo3/` or shared Rust crates that feed `batchalign_core`,
rebuild the extension before running Python tests that import it:

```bash
make build-python
# or:
uv run maturin develop
```

Prefer explicit fake seams over `monkeypatch` when touching production code.
If a test needs to replace runtime behavior, the first question should be
whether the production boundary wants a typed injected dependency instead.
The remaining worker seams should follow the same rule: keep runtime boundaries
typed and injectable instead of patching module globals.

The staged worker-protocol V2 boundary has its own cross-language drift suite:

```bash
uv run pytest batchalign/tests/test_worker_protocol_v2_types.py -q
uv run pytest batchalign/tests/test_worker_protocol_v2_artifacts.py -q
uv run pytest batchalign/tests/test_worker_fa_v2.py -q
cargo nextest run -p batchalign-app --test worker_protocol_v2_compat
cargo nextest run -p batchalign-app -E 'test(fa_result_v2)'
cargo nextest run -p batchalign-app --test worker_v2_fa_roundtrip
```

Those tests read the same fixture set under
`tests/fixtures/worker_protocol_v2/` so the Rust and Python schema models stay
aligned before the transport migration goes live. The artifact-reader test
adds a second guarantee: once Rust starts materializing prepared audio and
prepared JSON payloads, the Python side has a thin, deterministic reader layer
instead of ad hoc file handling in each worker path.

## Rust tests

PyO3 extension:

```bash
cargo nextest run --manifest-path pyo3/Cargo.toml
```

Root workspace:

```bash
cargo nextest run --workspace
```

Useful focused suites:

```bash
cargo nextest run -p batchalign-cli --test cli
cargo nextest run -p batchalign-cli --test ci_checks
cargo nextest run -p batchalign-cli --test e2e
cargo nextest run -p batchalign-app --test integration
cargo nextest run -p batchalign-app --test worker_integration
cargo nextest run -p batchalign-app --test json_compat
```

The real-server `batchalign-app` integration suite is intentionally serialized
inside its harness, even under Rust's default parallel test runner. Those
tests spin up full HTTP servers plus Python worker subprocesses, so allowing
many of them to overlap creates resource-contention flakes rather than useful
coverage.

Golden suites remain separate because they are slower and may depend on real
model behavior:

```bash
uv run pytest -m golden -v
cargo nextest run -p batchalign-app --test golden
```

## Type checking

The current repo-native mypy gate is:

```bash
uv run mypy
# or together with clippy:
make lint
```

## Dashboard Playwright tests

The standard entrypoint for dashboard browser tests is the `frontend/` package,
not the `frontend/e2e/` subdirectory directly:

```bash
cd frontend
npm run e2e:install
npm run test:e2e
```

If Chromium has not been installed on the machine yet:

```bash
cd frontend
npm run test:e2e:setup
```

That setup keeps Playwright dependency management and the human-facing command
surface rooted at the main frontend package, while the actual tests and lockfile
remain isolated under `frontend/e2e/`.

## CI hygiene

The release-facing CI checks currently cover:

- CLI/package version sync
- stale legacy-term detection in active docs and code
- retired package/path checks
- targeted integration coverage for command execution paths

Run the hygiene suite locally with:

```bash
cargo nextest run -p batchalign-cli --test ci_checks
make ci-local
```

## Coverage metrics

There is an official coverage workflow in `.github/workflows/test.yml`, but it
is currently a **manual audit tool**, not a release gate.

Current signoff snapshot from the 2026-03-16 hardening pass:

- the broad non-integration Python coverage run now reports `90%` total across
  `batchalign/` (`678 passed`, `3 skipped`, `27 deselected`)
- the full Python inference adapter surface is now fully covered in focused
  runs:
  - `batchalign/inference/*.py`
  - `batchalign/inference/hk/*.py`
- the remaining low-coverage Python areas are no longer the thin model adapter
  boundary; they are training, worker bootstrap, runtime, and test-only helper
  surfaces
- the inference-specific coverage work also exposed two dead/unreachable
  branches, which were removed (`utseg.py` and `hk/_tencent_api.py`), plus one
  real uncovered-branch bug in the Whisper beam-index timestamp path
  (`audio.py`) that is now fixed

When refreshing the broad Python snapshot locally, prefer the non-integration
pass unless you explicitly need environment-dependent engine coverage:

```bash
uv run --no-sync pytest -n0 --cov=batchalign --cov-report=term \
  --disable-pytest-warnings -m 'not integration' -q batchalign/tests
```

The workflow generates LCOV artifacts with the repo-native coverage commands:

```bash
uv run --no-sync pytest --cov=batchalign --cov-report=lcov:lcov-python.info \
  batchalign --disable-pytest-warnings -k 'not test_whisper_fa_pipeline'

cargo llvm-cov nextest --manifest-path pyo3/Cargo.toml \
  --lcov --output-path lcov-rust.info

cargo llvm-cov nextest --workspace \
  --lcov --output-path lcov-rust-workspace.info
```

Current limitations:

- the coverage job runs only on `workflow_dispatch`
- there is no enforced minimum threshold in CI yet
- there is no published dashboard or badge yet
- Codecov upload is still pending repository token/config setup
