.PHONY: build dev-ready build-rust build-python build-python-full build-dashboard sync check test test-quick test-rust test-python test-workers test-ml test-affected lint lint-rust lint-python lint-affected clean ci-local ci-full install-hooks generate-ipc-types check-ipc-drift check-affected

# ---------------------------------------------------------------------------
# FAST DEV LOOP (< 10 seconds incremental)
# ---------------------------------------------------------------------------

# Default: check compilation only. This is what you run after every edit.
check:
	cargo check -p batchalign-cli -q

# Quick test: library tests only. No Python, no ML, no OOM risk. ~3 seconds.
test: test-quick

test-quick:
	cargo test --workspace --lib -q

# Houjun-path dev build: fast extension rebuild + Rust CLI (debug), no dashboard.
dev-ready: build-python build-rust

# ---------------------------------------------------------------------------
# BUILD TARGETS
# ---------------------------------------------------------------------------

# Full dev build: dashboard (embedded in binary) + PyO3 + Rust CLI (debug).
build: build-dashboard build-python build-rust

# Rust CLI binary (debug for fast incremental builds).
build-rust:
	cargo build -p batchalign-cli

# Rust CLI binary (release, for large-scale work).
build-release:
	cargo build --release -p batchalign-cli

# Fast local-dev rebuild: extension-only PyO3.
build-python:
	uv run maturin develop -m pyo3/Cargo.toml -F pyo3/extension-module

# Full packaged rebuild: extension + CLI binary copied into package.
build-python-full: build-rust
	cp target/debug/batchalign3 batchalign/_bin/batchalign3
	uv run maturin develop -m pyo3/Cargo.toml -F pyo3/extension-module

# Build React dashboard and deploy to ~/.batchalign3/dashboard/.
build-dashboard:
	bash scripts/build_react_dashboard.sh

# Install/update Python deps + rebuild batchalign_core from scratch.
sync:
	uv sync --group dev

# ---------------------------------------------------------------------------
# EXTENDED TEST TARGETS (opt-in, not part of the default dev loop)
# ---------------------------------------------------------------------------

# Full safe Rust tests — library + safe integration tests. ~5 seconds.
test-rust:
	cargo test --workspace --lib -q
	cargo test -p batchalign-app --test json_compat --test workflow_helpers -q

# Python tests (pytest). Slow — run before pushing, not every edit.
test-python:
	uv run pytest --ignore=_private

# Worker tests — spawns Python test-echo workers (no ML models).
# Memory guard protects against OOM. Still, run only when needed.
test-workers:
	cargo test -p batchalign-app --test worker_integration -- --test-threads=1
	cargo test -p batchalign-app --test worker_protocol_matrix -- --test-threads=1
	cargo test -p batchalign-app --test gpu_concurrent_dispatch -- --test-threads=1
	cargo test -p batchalign-app --test integration -- --test-threads=1

# ML golden tests — loads real ML models (2-15 GB each).
# ONLY run on net (256 GB). See docs/memory-safety.md.
test-ml:
	@echo "WARNING: ML tests load Whisper/Stanza models (2-15 GB each)."
	@echo "Only run on machines with 256+ GB RAM (e.g., net)."
	@echo "Press Ctrl-C to abort, or wait 3 seconds to continue..."
	@sleep 3
	cargo test -p batchalign-app --test ml_golden -- --test-threads=1

test-affected:
	cargo xtask affected-rust test

# ---------------------------------------------------------------------------
# LINT (run before pushing, not every edit)
# ---------------------------------------------------------------------------

# Rust only — faster than full lint.
lint-rust:
	cargo clippy -p batchalign-types -p batchalign-app -p batchalign-cli --lib -- -D warnings

# Full lint (Rust + Python types). Slow.
lint:
	cargo clippy --workspace --all-targets -- -D warnings
	uv run mypy

# Python types only.
lint-python:
	uv run mypy

lint-affected:
	cargo xtask affected-rust clippy

check-affected:
	cargo xtask affected-rust check

# Regenerate Python Pydantic models from Rust IPC types via JSON Schema.
# Run after changing any Rust struct/enum with JsonSchema that crosses the Python boundary.
generate-ipc-types:
	bash scripts/generate_ipc_types.sh

# Check that Rust IPC schemas are up to date. Exits non-zero on drift.
check-ipc-drift:
	bash scripts/check_ipc_type_drift.sh

# Fast local CI: fmt + affected compile checks.
ci-local:
	@echo "==> fmt check"
	cargo fmt --all -- --check
	@echo "==> affected compile check"
	cargo xtask affected-rust check
	@echo "==> dashboard API drift"
	bash scripts/check_dashboard_api_drift.sh
	@echo "✓ ci-local passed"

# Full local CI: mirrors the strict CI-style gate and keeps heavy checks explicit.
ci-full:
	@echo "==> fmt check"
	cargo fmt --all -- --check
	@echo "==> clippy"
	cargo clippy --workspace --all-targets -- -D warnings
	@echo "==> compile check"
	cargo check --workspace
	@echo "==> mypy (strict)"
	uv run mypy
	@echo "==> IPC schema drift"
	bash scripts/check_ipc_type_drift.sh
	@echo "==> dashboard API drift"
	bash scripts/check_dashboard_api_drift.sh
	@echo "✓ ci-full passed"

# Install git hooks (pre-push).
install-hooks:
	ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
	@echo "✓ pre-push hook installed"

# Remove the release binary from the default Cargo target dir.
clean:
	rm -f target/release/batchalign3
