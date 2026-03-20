#!/usr/bin/env bash
# Pre-push hook: fast local checks that mirror CI gates.
# Install: make install-hooks
set -euo pipefail

echo "==> pre-push: fmt check"
cargo fmt --all -- --check

echo "==> pre-push: clippy"
cargo clippy --workspace --all-targets -- -D warnings

echo "==> pre-push: dashboard API drift check"
bash scripts/check_dashboard_api_drift.sh

echo "✓ All pre-push checks passed"
# Note: mypy is in 'make ci-local' but not here (too slow + transitive import noise)
