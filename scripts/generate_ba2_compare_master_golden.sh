#!/usr/bin/env bash
# Generate compare golden reference outputs from live batchalign2-master.
#
# This script runs the current `batchalign2-master` compare command on a small
# curated fixture set and writes committed oracle artifacts under:
#   batchalign/tests/golden/ba2_reference/compare/
#
# Requirements:
# - a batchalign2-master checkout (default: ~/batchalign2-master)
# - a working `batchalign` console script with BA2 dependencies installed
#
# Usage:
#   bash scripts/generate_ba2_compare_master_golden.sh
#   BATCHALIGN2_MASTER_DIR=/path/to/batchalign2-master bash scripts/generate_ba2_compare_master_golden.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="$REPO_ROOT/batchalign/tests/support/parity"
GOLDEN="$REPO_ROOT/batchalign/tests/golden/ba2_reference/compare"
BA2_MASTER_DIR="${BATCHALIGN2_MASTER_DIR:-$HOME/batchalign2-master}"
BA2_BIN="${BATCHALIGN2_BIN:-$(command -v batchalign || true)}"

[[ -d "$BA2_MASTER_DIR" ]] || {
    echo "ERROR: batchalign2-master checkout not found: $BA2_MASTER_DIR" >&2
    exit 1
}
[[ -n "$BA2_BIN" ]] || {
    echo "ERROR: batchalign console script not found; set BATCHALIGN2_BIN" >&2
    exit 1
}

mkdir -p "$GOLDEN"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FIXTURES_LIST=(
    eng_compare_exact
    eng_compare_multi_exact
)

echo "=== Generating BA2-master compare goldens ==="
echo "BA2 master: $BA2_MASTER_DIR"
echo "batchalign : $BA2_BIN"

for fixture in "${FIXTURES_LIST[@]}"; do
    echo ""
    echo "--- compare $fixture ---"
    rm -rf "$WORK/in" "$WORK/out"
    mkdir -p "$WORK/in" "$WORK/out"
    cp "$FIXTURES/${fixture}.cha" "$WORK/in/"
    cp "$FIXTURES/${fixture}.gold.cha" "$WORK/in/"

    if PYTHONPATH="$BA2_MASTER_DIR" "$BA2_BIN" compare "$WORK/in" "$WORK/out" --lang eng; then
        cp "$WORK/out/${fixture}.cha" "$GOLDEN/${fixture}.master.cha"
        cp "$WORK/out/${fixture}.compare.csv" "$GOLDEN/${fixture}.master.compare.csv"
        echo "    OK"
    else
        echo "    ERROR: compare failed for $fixture" >&2
        exit 1
    fi
done

echo ""
echo "=== Done ==="
find "$GOLDEN" -maxdepth 1 -type f | sort
