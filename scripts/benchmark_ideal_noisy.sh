#!/usr/bin/env bash
set -euo pipefail

RELATION="${1:-CtD}"
RESULTS_DIR="${2:-results}"
shift 2 || true

EXTRA_ARGS=("$@")

echo "Running IDEAL simulator benchmark..."
python3 scripts/run_optimized_pipeline.py \
  --relation "$RELATION" \
  --results_dir "$RESULTS_DIR" \
  --quantum_config_path config/quantum_config_ideal.yaml \
  "${EXTRA_ARGS[@]}"

echo "Running NOISY simulator benchmark..."
python3 scripts/run_optimized_pipeline.py \
  --relation "$RELATION" \
  --results_dir "$RESULTS_DIR" \
  --quantum_config_path config/quantum_config_noisy.yaml \
  "${EXTRA_ARGS[@]}"
