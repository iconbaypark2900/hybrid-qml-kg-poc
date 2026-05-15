#!/usr/bin/env bash
# launch_jupyter.sh — Launch Jupyter Lab for the playbooks directory.
set -euo pipefail
PORT="${JUPYTER_PORT:-8888}"
echo "Starting Jupyter Lab on port $PORT …"
jupyter lab --no-browser --ip=0.0.0.0 --port="$PORT" --notebook-dir=playbooks
