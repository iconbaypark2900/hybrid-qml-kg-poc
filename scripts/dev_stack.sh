#!/usr/bin/env bash
# Start FastAPI (middleware) and Next.js together with matching API URL.
# Usage: ./scripts/dev_stack.sh
#        API_PORT=9000 ./scripts/dev_stack.sh
#
# - API:  http://127.0.0.1:${API_PORT:-8000}  (default 8000, matches uvicorn default + frontend/.env.example)
# - Next: http://localhost:3000 (default; override with FRONTEND_PORT)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
API_PORT="${API_PORT:-8000}"
API_HOST="${API_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
API_URL="http://${API_HOST}:${API_PORT}"

usage() {
    cat <<'EOF'
Start FastAPI and Next.js with matching NEXT_PUBLIC_API_URL.

Usage: ./scripts/dev_stack.sh
       API_PORT=9000 FRONTEND_PORT=3001 ./scripts/dev_stack.sh

Environment:
  API_PORT       API listen port (default: 8000)
  API_HOST       API bind address (default: 127.0.0.1)
  FRONTEND_PORT  Next.js dev port (default: 3000)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

cd "$PROJECT_ROOT"

if [[ ! -f middleware/api.py ]]; then
    echo "Error: run from repo root (middleware/api.py missing)." >&2
    exit 1
fi

if [[ ! -d frontend/node_modules ]]; then
    echo "Hint: run 'cd frontend && pnpm install' first." >&2
fi

if [[ -x "$PROJECT_ROOT/.venv/bin/uvicorn" ]]; then
    UVICORN="$PROJECT_ROOT/.venv/bin/uvicorn"
elif command -v uvicorn &>/dev/null; then
    UVICORN="uvicorn"
else
    echo "Error: uvicorn not found. Activate .venv or install dependencies." >&2
    exit 1
fi

if ! command -v pnpm &>/dev/null; then
    echo "Error: pnpm not found. Install pnpm or use corepack enable." >&2
    exit 1
fi

API_PID=""
cleanup() {
    if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
        echo ""
        echo "Stopping API (pid ${API_PID})..."
        kill "${API_PID}" 2>/dev/null || true
        wait "${API_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting FastAPI at ${API_URL} ..."
"$UVICORN" middleware.api:app --reload --host "${API_HOST}" --port "${API_PORT}" &
API_PID=$!

echo "Waiting for API /status ..."
ready=0
for _ in $(seq 1 60); do
    if curl -sf "${API_URL}/status" >/dev/null 2>&1; then
        ready=1
        break
    fi
    sleep 0.25
done

if [[ "${ready}" -ne 1 ]]; then
    echo "Error: API did not become ready at ${API_URL}/status" >&2
    exit 1
fi

echo "API OK. Starting Next.js on port ${FRONTEND_PORT} (NEXT_PUBLIC_API_URL=${API_URL})"
cd "$PROJECT_ROOT/frontend"
export NEXT_PUBLIC_API_URL="${API_URL}"
# Use `pnpm exec` so we pass `-p` to `next` directly. `pnpm dev -- -p …` becomes
# `next dev -- -p …` and Next treats `-p` as the project directory.
pnpm exec next dev -p "${FRONTEND_PORT}"
