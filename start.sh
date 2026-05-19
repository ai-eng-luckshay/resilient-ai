#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Resilient AI — start.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
PID_FILE=".pids"
LOG_DIR="logs"
API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[start]${NC} $*"; }
warn() { echo -e "${YELLOW}[start]${NC} $*"; }

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment…"
    python -m venv "$VENV_DIR"
fi

# Prepend venv bin to PATH directly — avoids sourcing activate (needs cygpath)
VENV_BIN="$VENV_DIR/Scripts"
[ -d "$SCRIPT_DIR/$VENV_BIN" ] || VENV_BIN="$VENV_DIR/bin"
export PATH="$SCRIPT_DIR/$VENV_BIN:$PATH"
log "Virtual environment ready."

# ── Dependencies ──────────────────────────────────────────────────────────────
if ! command -v pip-compile &>/dev/null; then
    log "Installing pip-tools…"
    pip install --quiet "pip<26" pip-tools
fi

if [ ! -f "requirements.txt" ] || [ "requirements.in" -nt "requirements.txt" ]; then
    log "Compiling requirements.in → requirements.txt…"
    pip-compile requirements.in --output-file requirements.txt --quiet --annotation-style line
fi

log "Syncing dependencies…"
pip-sync requirements.txt --quiet

log "Installing project in editable mode…"
pip install -e . --quiet --no-deps

# ── Services ──────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
: > "$PID_FILE"

log "Starting backend (port $API_PORT)…"
PYTHONUNBUFFERED=1 uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" --no-access-log \
    > >(tee "$LOG_DIR/api.log") 2>&1 &
echo "api $!" >> "$PID_FILE"

API_PID=$(grep "^api" "$PID_FILE" | awk '{print $2}')
connected=false

log "Waiting for backend to be ready (up to 30s)…"
for i in $(seq 1 30); do
    # Fail fast if the process already died (import error, port conflict, etc.)
    if ! kill -0 "$API_PID" 2>/dev/null; then
        warn "Backend process exited unexpectedly. Last log output:"
        echo "──────────────────────────────────────────"
        tail -30 "$LOG_DIR/api.log" 2>/dev/null || true
        echo "──────────────────────────────────────────"
        exit 1
    fi

    if curl -sf "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
        connected=true
        break
    fi

    printf "${GREEN}[start]${NC} [$i/30] still starting…\r"
    sleep 1
done
echo ""

if [ "$connected" = false ]; then
    warn "Backend did not respond after 30s. Last log output:"
    echo "──────────────────────────────────────────"
    tail -30 "$LOG_DIR/api.log" 2>/dev/null || true
    echo "──────────────────────────────────────────"
    exit 1
fi

log "Backend ready → http://localhost:$API_PORT"

log "Starting dashboard (port $UI_PORT)…"
streamlit run app/ui/streamlit_app.py \
    --server.port "$UI_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --logger.level error \
    > "$LOG_DIR/ui.log" 2>&1 &
echo "ui $!" >> "$PID_FILE"

log "Dashboard ready → http://localhost:$UI_PORT"
echo ""
echo -e "${GREEN}┌─────────────────────────────────────────────────┐${NC}"
echo -e "${GREEN}│  ✅  All services are up and running!           │${NC}"
echo -e "${GREEN}│                                                  │${NC}"
echo -e "${GREEN}│  👉  Open the demo dashboard in your browser:   │${NC}"
echo -e "${GREEN}│      http://localhost:$UI_PORT                   │${NC}"
echo -e "${GREEN}│                                                  │${NC}"
echo -e "${GREEN}│  🔌  REST API  → http://localhost:$API_PORT           │${NC}"
echo -e "${GREEN}│  🤖  A2A       → http://localhost:$API_PORT/agent/a2a │${NC}"
echo -e "${GREEN}│                                                  │${NC}"
echo -e "${GREEN}│  Press Ctrl+C to stop all services.             │${NC}"
echo -e "${GREEN}└─────────────────────────────────────────────────┘${NC}"
echo ""

trap "./stop.sh" INT TERM
wait
