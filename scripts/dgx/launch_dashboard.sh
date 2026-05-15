#!/usr/bin/env bash
# launch_dashboard.sh — Launch the Streamlit evidence dashboard.
set -euo pipefail
PORT="${DASHBOARD_PORT:-8501}"
echo "Starting Streamlit dashboard on port $PORT …"
streamlit run benchmarking/dashboard.py --server.port "$PORT" --server.address 0.0.0.0
