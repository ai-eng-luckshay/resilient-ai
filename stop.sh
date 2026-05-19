#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Resilient AI — stop.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_FILE=".pids"
GREEN='\033[0;32m'
NC='\033[0m'

log() { echo -e "${GREEN}[stop]${NC} $*"; }

if [ ! -f "$PID_FILE" ]; then
    log "No .pids file found — nothing to stop."
    exit 0
fi

while read -r name pid; do
    if kill -0 "$pid" 2>/dev/null; then
        log "Stopping $name (PID $pid)…"
        kill "$pid" 2>/dev/null || true
    else
        log "$name (PID $pid) is already stopped."
    fi
done < "$PID_FILE"

rm -f "$PID_FILE"
log "All services stopped."
