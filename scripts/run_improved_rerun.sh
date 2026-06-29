#!/usr/bin/env bash
# Re-run Tier 1 CtD with settings closer to README best (0.7987) + MoA.
# Differences from run_tier1_cached_sequential.sh multiseed leg:
#   - --use_moa_features
#   - --optimize_feature_map_reps (README best; do NOT use --fast_mode — it drops HistGBDT/SVM)
#   - --skip_vqc
#   - --qsvc_nystrom_m 200 on multiseed queue (full kernel ~5 min/seed, no QSVC gain)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate
export HYBRID_QML_SYSTEM=dgx
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="$PROJECT_ROOT/data"
mkdir -p results/rerun_improved

# Keep cached 256d artifacts available under 128d paths (same as Tier 1 queue).
ln -sf rotate_256d_entity_embeddings.npy "$DATA_DIR/rotate_128d_entity_embeddings.npy"
ln -sf rotate_256d_entity_ids.json "$DATA_DIR/rotate_128d_entity_ids.json"
ln -sf rotate_256d_relation_embeddings.npy "$DATA_DIR/rotate_128d_relation_embeddings.npy"

SEEDS=(42 7 13 99 2026)
LOG="$PROJECT_ROOT/results/rerun_improved/run.log"

echo "=== Improved rerun started at $(date) ===" | tee "$LOG"

for seed in "${SEEDS[@]}"; do
  echo "--- seed $seed at $(date) ---" | tee -a "$LOG"
  mkdir -p "results/rerun_improved/seed_$seed"
  python3 scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --embedding_method RotatE \
    --embedding_dim 128 \
    --embedding_epochs 200 \
    --negative_sampling hard \
    --qml_dim 16 \
    --qml_feature_map Pauli \
    --qml_feature_map_reps 2 \
    --qsvc_C 0.1 \
    --optimize_feature_map_reps \
    --run_ensemble \
    --ensemble_method stacking \
    --tune_classical \
    --qml_pre_pca_dim 24 \
    --use_moa_features \
    --use_cached_embeddings \
    --allow_cached_embeddings_with_holdout \
    --skip_vqc \
    --skip_svm_rbf \
    --qsvc_nystrom_m 200 \
    --random_state "$seed" \
    --results_dir "results/rerun_improved/seed_$seed" \
    --quantum_config_path config/quantum_config_dgx.yaml \
    2>&1 | tee -a "$LOG"
  echo "--- seed $seed finished at $(date) ---" | tee -a "$LOG"
done

python3 scripts/aggregate_multiseed.py \
  --results-dir results/rerun_improved \
  --seeds "${SEEDS[@]}" \
  --out results/rerun_improved/summary.json \
  2>&1 | tee -a "$LOG"

echo "=== Improved rerun complete at $(date) ===" | tee -a "$LOG"
