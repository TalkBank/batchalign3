#!/usr/bin/env bash
# Install batchalign3 worker launchd agents (macOS).
#
# Usage:
#   bash deploy/scripts/install_launchd_workers.sh [--dry-run]
#
# Installs plist files into ~/Library/LaunchAgents/ with the correct
# batchalign3 binary path and state directory substituted in.
# Then loads the agents with launchctl.

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Resolve batchalign3 binary.
BA3_BIN="$(command -v batchalign3 2>/dev/null || true)"
if [[ -z "$BA3_BIN" ]]; then
    echo "ERROR: batchalign3 not found in PATH." >&2
    echo "Install it first: uv tool install batchalign3" >&2
    exit 1
fi

# Resolve state directory.
STATE_DIR="${BATCHALIGN_STATE_DIR:-$HOME/.batchalign3}"
mkdir -p "$STATE_DIR/logs"

PLIST_DIR="$(cd "$(dirname "$0")/../launchd" && pwd)"
INSTALL_DIR="$HOME/Library/LaunchAgents"
mkdir -p "$INSTALL_DIR"

echo "batchalign3 binary: $BA3_BIN"
echo "State directory:    $STATE_DIR"
echo "Plist source:       $PLIST_DIR"
echo "Install target:     $INSTALL_DIR"
echo

for plist in "$PLIST_DIR"/*.plist; do
    filename="$(basename "$plist")"
    label="${filename%.plist}"
    dest="$INSTALL_DIR/$filename"

    echo "Installing $filename..."

    if $DRY_RUN; then
        echo "  [dry-run] Would copy $plist -> $dest"
        echo "  [dry-run] Would substitute __BATCHALIGN3_BIN__ -> $BA3_BIN"
        echo "  [dry-run] Would substitute __STATE_DIR__ -> $STATE_DIR"
        echo "  [dry-run] Would run: launchctl load $dest"
        continue
    fi

    # Unload existing agent if loaded.
    launchctl unload "$dest" 2>/dev/null || true

    # Copy and substitute placeholders.
    sed -e "s|__BATCHALIGN3_BIN__|$BA3_BIN|g" \
        -e "s|__STATE_DIR__|$STATE_DIR|g" \
        "$plist" > "$dest"

    # Load the agent.
    launchctl load "$dest"
    echo "  Loaded: $label"
done

echo
echo "Done. Check status with:"
echo "  launchctl list | grep batchalign3"
echo "  batchalign3 worker list"
