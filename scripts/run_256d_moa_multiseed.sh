#!/usr/bin/env bash
# 256D RotatE + MoA + full classical + stacking (best classical pick for ensemble).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate
export HYBRID_QML_SYSTEM=dgx
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="$PROJECT_ROOT/data"
OUT_DIR="results/rerun_256d_moa"
mkdir -p "$OUT_DIR"

ln -sf rotate_256d_entity_embeddings.npy "$DATA_DIR/rotate_128d_entity_embeddings.npy"
ln -sf rotate_256d_entity_ids.json "$DATA_DIR/rotate_128d_entity_ids.json"
ln -sf rotate_256d_relation_embeddings.npy "$DATA_DIR/rotate_128d_relation_embeddings.npy"

SEEDS=(42 7 13 99 2026)
LOG="$PROJECT_ROOT/$OUT_DIR/run.log"

echo "=== 256D + MoA multiseed started at $(date) ===" | tee "$LOG"

for seed in "${SEEDS[@]}"; do
  if compgen -G "$OUT_DIR/seed_$seed/optimized_results_*.json" > /dev/null; then
    echo "--- seed $seed skipped (results exist) at $(date) ---" | tee -a "$LOG"
    continue
  fi
  echo "--- seed $seed at $(date) ---" | tee -a "$LOG"
  mkdir -p "$OUT_DIR/seed_$seed"
  python3 scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --embedding_method RotatE \
    --embedding_dim 256 \
    --embedding_epochs 200 \
    --negative_sampling hard \
    --use_moa_features \
    --tune_classical \
    --skip_vqc \
    --skip_svm_rbf \
    --qml_dim 16 \
    --qml_feature_map Pauli \
    --qml_feature_map_reps 2 \
    --qsvc_C 0.1 \
    --qml_pre_pca_dim 24 \
    --optimize_feature_map_reps \
    --run_ensemble \
    --ensemble_method stacking \
    --use_cached_embeddings \
    --allow_cached_embeddings_with_holdout \
    --qsvc_nystrom_m 200 \
    --random_state "$seed" \
    --results_dir "$OUT_DIR/seed_$seed" \
    --quantum_config_path config/quantum_config_dgx.yaml \
    2>&1 | tee -a "$LOG"
  echo "--- seed $seed finished at $(date) ---" | tee -a "$LOG"
done

python3 scripts/aggregate_multiseed.py \
  --results-dir "$OUT_DIR" \
  --seeds "${SEEDS[@]}" \
  --out "$OUT_DIR/summary.json" \
  2>&1 | tee -a "$LOG"

echo "=== 256D + MoA multiseed complete at $(date) ===" | tee -a "$LOG"
