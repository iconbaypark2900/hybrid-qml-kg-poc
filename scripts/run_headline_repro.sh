#!/usr/bin/env bash
# Reproduce README headline (Ensemble PR-AUC ~0.7987, RF ~0.7838).
# Locked config: utils/preregistered_constants.py + README "Reproduce the best result".
# Do NOT use --use_moa_features (headline run was without MoA).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODE="${1:-cached}"   # cached | fresh
SEED="${2:-42}"
RESULTS_SUBDIR="${3:-results/rerun_headline}"

mkdir -p "$RESULTS_SUBDIR/seed_${SEED}"
LOG="$RESULTS_SUBDIR/seed_${SEED}_run.log"

COMMON=(
  --relation CtD
  --full_graph_embeddings
  --embedding_method RotatE
  --embedding_dim 128
  --embedding_epochs 200
  --negative_sampling hard
  --qml_dim 16
  --qml_feature_map Pauli
  --qml_feature_map_reps 2
  --qsvc_C 0.1
  --optimize_feature_map_reps
  --run_ensemble
  --ensemble_method stacking
  --tune_classical
  --qml_pre_pca_dim 24
  --fast_mode
  --skip_vqc
  --random_state "$SEED"
  --results_dir "$RESULTS_SUBDIR/seed_${SEED}"
  --quantum_config_path config/quantum_config_ideal.yaml
)

echo "=== Headline repro mode=$MODE seed=$SEED at $(date) ===" | tee "$LOG"

if [[ "$MODE" == "cached" ]]; then
  export HYBRID_QML_SYSTEM=dgx
  DATA_DIR="$PROJECT_ROOT/data"
  ln -sf rotate_256d_entity_embeddings.npy "$DATA_DIR/rotate_128d_entity_embeddings.npy"
  ln -sf rotate_256d_entity_ids.json "$DATA_DIR/rotate_128d_entity_ids.json"
  ln -sf rotate_256d_relation_embeddings.npy "$DATA_DIR/rotate_128d_relation_embeddings.npy"
  python3 scripts/run_optimized_pipeline.py \
    "${COMMON[@]}" \
    --use_cached_embeddings \
    --allow_cached_embeddings_with_holdout \
    2>&1 | tee -a "$LOG"
elif [[ "$MODE" == "fresh" ]]; then
  export HYBRID_QML_SYSTEM=dgx
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  # fast_mode forces num_epochs=50 in pipeline; headline used 200-epoch RotatE.
  # Train embeddings + full pipeline without fast_mode (RF/ET/QSVC/stacking).
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
    --skip_vqc \
    --random_state "$SEED" \
    --results_dir "$RESULTS_SUBDIR/seed_${SEED}" \
    --quantum_config_path config/quantum_config_ideal.yaml \
    2>&1 | tee -a "$LOG"
else
  echo "Unknown mode: $MODE (use cached or fresh)" >&2
  exit 1
fi

echo "=== Done at $(date) ===" | tee -a "$LOG"
