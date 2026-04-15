#!/usr/bin/env bash
# Run Hugging Face lite Gradio UI (hf_space/app.py) against the in-process FastAPI app.
# Usage: ./scripts/run_hf_lite.sh
# Requires: repository root, PYTHONPATH=., dependencies (see hf_space/requirements.txt).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f hf_space/app.py ]]; then
  echo "Error: hf_space/app.py missing (run from repo root)." >&2
  exit 1
fi

PY=python3
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PY="$PROJECT_ROOT/.venv/bin/python"
fi

echo "Starting HF lite Gradio at http://127.0.0.1:${PORT:-7860} (PYTHONPATH=$PROJECT_ROOT)"
exec "$PY" hf_space/app.py
