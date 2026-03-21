#!/usr/bin/env bash
# Pre-push hook: fast local checks that mirror CI gates.
# Install: make install-hooks
set -euo pipefail

echo "==> pre-push: fmt check"
cargo fmt --all -- --check

echo "==> pre-push: affected compile check"
cargo xtask affected-rust check

echo "==> pre-push: dashboard API drift check"
bash scripts/check_dashboard_api_drift.sh

if [[ "${BATCHALIGN_PRE_PUSH_CLIPPY:-0}" == "1" ]]; then
  echo "==> pre-push: affected clippy"
  cargo xtask affected-rust clippy
fi

if [[ "${BATCHALIGN_PRE_PUSH_MYPY:-0}" == "1" ]]; then
  echo "==> pre-push: mypy"
  uv run mypy
fi

echo "✓ All pre-push checks passed"
# Heavy gates remain available explicitly:
#   BATCHALIGN_PRE_PUSH_CLIPPY=1 git push
#   BATCHALIGN_PRE_PUSH_MYPY=1 git push
