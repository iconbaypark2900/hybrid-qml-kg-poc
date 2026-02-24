#!/usr/bin/env bash
# Launch the Streamlit dashboard on the first available port.
# Usage: ./scripts/launch_dashboard.sh [--start-port 8501]

set -euo pipefail

START_PORT="${1:-8501}"
if [[ "$START_PORT" == "--start-port" ]]; then
    START_PORT="${2:-8501}"
fi

MAX_PORT=$((START_PORT + 50))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD="$PROJECT_ROOT/benchmarking/dashboard.py"

if [[ ! -f "$DASHBOARD" ]]; then
    echo "Error: dashboard not found at $DASHBOARD" >&2
    exit 1
fi

port_in_use() {
    # Returns 0 (true) if the port is occupied.
    if command -v ss &>/dev/null; then
        ss -tlnH "sport = :$1" 2>/dev/null | grep -q "$1"
    elif command -v lsof &>/dev/null; then
        lsof -iTCP:"$1" -sTCP:LISTEN -t &>/dev/null
    elif command -v netstat &>/dev/null; then
        netstat -tlnp 2>/dev/null | grep -q ":$1 "
    else
        # Last resort: try to bind briefly with Python.
        python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(('127.0.0.1', $1))
    s.close()
    sys.exit(1)   # port is free
except OSError:
    sys.exit(0)   # port is in use
" 2>/dev/null
    fi
}

PORT="$START_PORT"
while port_in_use "$PORT"; do
    echo "Port $PORT is in use, trying next..."
    PORT=$((PORT + 1))
    if [[ "$PORT" -gt "$MAX_PORT" ]]; then
        echo "Error: no open port found between $START_PORT and $MAX_PORT" >&2
        exit 1
    fi
done

echo "Starting dashboard on port $PORT"
echo "  Local URL:   http://localhost:$PORT"
echo ""

exec streamlit run "$DASHBOARD" \
    --server.port "$PORT" \
    --server.headless true \
    --server.address 0.0.0.0
