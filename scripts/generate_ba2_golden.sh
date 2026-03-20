#!/usr/bin/env bash
# Generate BA2 golden reference outputs for parity testing.
#
# Runs batchalignjan9 (canonical Jan 9 baseline) on curated CHAT fixtures.
#
# Usage: bash scripts/generate_ba2_golden.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURES="$REPO_ROOT/batchalign/tests/support/parity"
GOLDEN="$REPO_ROOT/batchalign/tests/golden/ba2_reference"

command -v batchalignjan9 >/dev/null || { echo "ERROR: batchalignjan9 not on PATH" >&2; exit 1; }

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

run_ba2() {
    # run_ba2 <ba2_command> <fixture_name> [extra_args...]
    local cmd=$1 fixture=$2
    shift 2

    rm -rf "$WORK/in" "$WORK/out"
    mkdir -p "$WORK/in" "$WORK/out"
    cp "$FIXTURES/${fixture}.cha" "$WORK/in/"

    echo "  [jan9] $cmd $fixture $*"
    if batchalignjan9 "$cmd" "$WORK/in" "$WORK/out" "$@" 2>/dev/null; then
        if [[ -f "$WORK/out/${fixture}.cha" ]]; then
            mkdir -p "$GOLDEN/$cmd"
            cp "$WORK/out/${fixture}.cha" "$GOLDEN/$cmd/${fixture}.jan9.cha"
            echo "    OK"
            return 0
        fi
    fi
    echo "    WARN: no output or failed"
    return 1
}

FIXTURES_LIST=(
    eng_disfluency
    eng_multi_speaker
    eng_overlap_ca
    eng_clinical_aphasia
    eng_retokenize
    eng_bilingual
    eng_complex_tiers
    spa_simple
    spa_clinical
    fra_simple
    deu_clinical
    jpn_clinical
    yue_timed
)

declare -A LANG_MAP=(
    [eng_disfluency]=eng [eng_multi_speaker]=eng [eng_overlap_ca]=eng
    [eng_clinical_aphasia]=eng [eng_retokenize]=eng [eng_bilingual]=eng
    [eng_complex_tiers]=eng [spa_simple]=spa [spa_clinical]=spa
    [fra_simple]=fra [deu_clinical]=deu [jpn_clinical]=jpn [yue_timed]=yue
)

echo "=== Generating BA2 golden outputs ==="

# --- morphotag (NO --lang flag — BA2 reads @Languages) ---
echo ""
echo "--- morphotag ---"
for f in "${FIXTURES_LIST[@]}"; do
    run_ba2 morphotag "$f" || true
done

# --- morphotag with retokenize (English only) ---
echo ""
echo "--- morphotag --retokenize ---"
for f in "${FIXTURES_LIST[@]}"; do
    lang="${LANG_MAP[$f]}"
    [[ "$lang" != "eng" ]] && continue
    rm -rf "$WORK/in" "$WORK/out"
    mkdir -p "$WORK/in" "$WORK/out"
    cp "$FIXTURES/${f}.cha" "$WORK/in/"
    echo "  [jan9] morphotag --retokenize $f"
    if batchalignjan9 morphotag "$WORK/in" "$WORK/out" --retokenize 2>/dev/null; then
        mkdir -p "$GOLDEN/morphotag_retok"
        [[ -f "$WORK/out/${f}.cha" ]] && cp "$WORK/out/${f}.cha" "$GOLDEN/morphotag_retok/${f}.jan9.cha" && echo "    OK"
    else
        echo "    WARN: retokenize failed"
    fi
done

# --- utseg (uses --lang) ---
echo ""
echo "--- utseg ---"
for f in "${FIXTURES_LIST[@]}"; do
    lang="${LANG_MAP[$f]}"
    run_ba2 utseg "$f" --lang "$lang" || true
done

# --- translate (NO --lang flag — reads @Languages) ---
echo ""
echo "--- translate ---"
for f in "${FIXTURES_LIST[@]}"; do
    run_ba2 translate "$f" || true
done

# --- coref (English only, no --lang flag) ---
echo ""
echo "--- coref ---"
for f in "${FIXTURES_LIST[@]}"; do
    lang="${LANG_MAP[$f]}"
    [[ "$lang" != "eng" ]] && continue
    run_ba2 coref "$f" || true
done

echo ""
echo "=== Done ==="
echo "Golden outputs in: $GOLDEN"
find "$GOLDEN" -name "*.cha" | sort
