#!/usr/bin/env bash
# Multi-seed evaluation: 5 seeds for arXiv v2
# Runs sequentially; each seed does a fresh train/test split + QSVC kernel
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PY="$PROJECT_ROOT/.venv/bin/python"
RESULTS_DIR="$PROJECT_ROOT/results/multiseed"
mkdir -p "$RESULTS_DIR"

BASE_ARGS=(
  --relation CtD --full_graph_embeddings
  --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200
  --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli
  --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24
  --run_ensemble --ensemble_method stacking --tune_classical --fast_mode
  --use_cached_embeddings
)

SEEDS=(42 7 13 99 2026)

for seed in "${SEEDS[@]}"; do
  echo ""
  echo "======================================================"
  echo "Starting seed=${seed} at $(date)"
  echo "======================================================"
  "$PY" scripts/run_optimized_pipeline.py \
    "${BASE_ARGS[@]}" \
    --random_state "$seed" \
    --results_dir "$RESULTS_DIR" \
    2>&1
  echo "Finished seed=${seed} at $(date) — exit code $?"
  echo ""
done

echo "======================================================"
echo "All 5 seeds complete at $(date)"
echo "Results in: $RESULTS_DIR"
echo "======================================================"
