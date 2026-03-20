.PHONY: build build-rust build-python build-dashboard sync test test-rust test-python lint clean ci-local install-hooks generate-ipc-types check-ipc-drift

# Full dev build: dashboard (embedded in binary) + PyO3 + Rust CLI (debug).
# Dashboard must be built first — Rust compilation embeds frontend/dist/ into the binary.
build: build-dashboard build-python build-rust

# Rust CLI binary (debug for fast incremental builds).
build-rust:
	cargo build -p batchalign-cli

# Rust CLI binary (release, for large-scale work).
build-release:
	cargo build --release -p batchalign-cli

# Rebuild the batchalign_core PyO3 extension into the active venv.
build-python:
	uv run maturin develop -m pyo3/Cargo.toml

# Build React dashboard and deploy to ~/.batchalign3/dashboard/.
build-dashboard:
	bash scripts/build_react_dashboard.sh

# Install/update Python deps + rebuild batchalign_core from scratch.
sync:
	uv sync --group dev

# Run all tests.
test: test-python test-rust

test-python:
	uv run pytest --ignore=_private

test-rust:
	cargo nextest run --manifest-path pyo3/Cargo.toml
	cargo nextest run --workspace

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	uv run mypy

# Regenerate Python Pydantic models from Rust IPC types via JSON Schema.
# Run after changing any Rust struct/enum with JsonSchema that crosses the Python boundary.
generate-ipc-types:
	bash scripts/generate_ipc_types.sh

# Check that Rust IPC schemas are up to date. Exits non-zero on drift.
check-ipc-drift:
	bash scripts/check_ipc_type_drift.sh

# Fast local CI: checks that mirror the CI pipeline (no tests).
ci-local:
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
	@echo "✓ ci-local passed"

# Install git hooks (pre-push).
install-hooks:
	ln -sf ../../scripts/pre-push.sh .git/hooks/pre-push
	@echo "✓ pre-push hook installed"

# Remove the release binary from the default Cargo target dir.
clean:
	rm -f target/release/batchalign3
