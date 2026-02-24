#!/usr/bin/env bash
set -euo pipefail

RELATION="${1:-CtD}"
RESULTS_DIR="${2:-results}"
BACKEND_NAME="${3:-ibm_brisbane}"          # ibm_fez | ibm_brisbane
BACKEND_CONFIG="${4:-config/quantum_config_hardware.yaml}"
shift 4 || true

EXTRA_ARGS=("$@")

export IBM_BACKEND="$BACKEND_NAME"

echo "Running HARDWARE benchmark (backend: $IBM_BACKEND, config: $BACKEND_CONFIG)..."
python3 scripts/run_optimized_pipeline.py \
  --relation "$RELATION" \
  --results_dir "$RESULTS_DIR" \
  --quantum_config_path "$BACKEND_CONFIG" \
  "${EXTRA_ARGS[@]}"
