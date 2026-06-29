#!/usr/bin/env bash
# Tier 1: MoA → CpD → multi-seed, using cached RotatE embeddings (no re-train).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate
export HYBRID_QML_SYSTEM=dgx
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

DATA_DIR="$PROJECT_ROOT/data"
mkdir -p results/moa_benchmark results/cpd_run results/multiseed

# 128d config loads rotate_128d_*; GPU run saved rotate_256d_* (complex→real).
ln -sf rotate_256d_entity_embeddings.npy "$DATA_DIR/rotate_128d_entity_embeddings.npy"
ln -sf rotate_256d_entity_ids.json "$DATA_DIR/rotate_128d_entity_ids.json"
ln -sf rotate_256d_relation_embeddings.npy "$DATA_DIR/rotate_128d_relation_embeddings.npy"

CACHE_FLAGS=(--use_cached_embeddings --allow_cached_embeddings_with_holdout)
EMBED_FLAGS=(--full_graph_embeddings --embedding_method RotatE --embedding_dim 128 --negative_sampling hard)
QML_FLAGS=(
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1
  --qml_pre_pca_dim 24 --run_ensemble --ensemble_method stacking --tune_classical
)

echo "=== Tier 1 cached queue started at $(date) ==="
ls -lh "$DATA_DIR"/rotate_128d_entity_embeddings.npy

echo ""
echo "=== [1/3] MoA benchmark (CtD + MoA features) ==="
python3 scripts/run_optimized_pipeline.py \
  --relation CtD \
  "${EMBED_FLAGS[@]}" \
  "${QML_FLAGS[@]}" \
  "${CACHE_FLAGS[@]}" \
  --use_moa_features \
  --results_dir results/moa_benchmark
echo "MoA done at $(date)"

echo ""
echo "=== [2/3] CpD relation ==="
python3 scripts/run_optimized_pipeline.py \
  --relation CpD \
  "${EMBED_FLAGS[@]}" \
  "${QML_FLAGS[@]}" \
  "${CACHE_FLAGS[@]}" \
  --results_dir results/cpd_run
echo "CpD done at $(date)"

echo ""
echo "=== [3/3] Multi-seed (Nyström m=200) ==="
for seed in 42 7 13 99 2026; do
  echo "--- seed $seed at $(date) ---"
  mkdir -p "results/multiseed/seed_$seed"
  python3 scripts/run_optimized_pipeline.py \
    --relation CtD \
    "${EMBED_FLAGS[@]}" \
    "${QML_FLAGS[@]}" \
    "${CACHE_FLAGS[@]}" \
    --qsvc_nystrom_m 200 \
    --random_state "$seed" \
    --results_dir "results/multiseed/seed_$seed"
  echo "--- seed $seed finished ---"
done

python3 scripts/aggregate_multiseed.py \
  --results-dir results/multiseed \
  --seeds 42 7 13 99 2026 \
  --out results/multiseed/summary.json

echo ""
echo "=== Tier 1 cached queue complete at $(date) ==="
