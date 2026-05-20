#!/usr/bin/env bash
# §4 GPU-accelerated simulation + full ensemble — docs/roadmap/04_quantum_scaling.md
# Prefer RUN_ON_DGX=1 (quantum_config_dgx.yaml). Do not combine with --gpu on the same command.
# For a non-DGX NVIDIA host: USE_GPU_FLAG=1 (selects config/quantum_config_gpu.yaml via --gpu).
#
# Usage:
#   ./scripts/run_gpu_ensemble_smoke.sh
#   USE_GPU_FLAG=1 ./scripts/run_gpu_ensemble_smoke.sh
#   RUN_ON_DGX=1 EXECUTE=1 ./scripts/run_gpu_ensemble_smoke.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE="${EXECUTE:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/gpu_run}"

if [[ "${RUN_ON_DGX:-}" == "1" && "${USE_GPU_FLAG:-}" == "1" ]]; then
  echo "ERROR: use RUN_ON_DGX=1 xor USE_GPU_FLAG=1 (not both). See docs/roadmap/04_quantum_scaling.md" >&2
  exit 2
fi

if [[ "${RUN_ON_DGX:-}" == "1" ]]; then
  PYLAUNCH=(env HYBRID_QML_SYSTEM=dgx python)
  GPU_FLAGS=()
elif [[ "${USE_GPU_FLAG:-}" == "1" ]]; then
  PYLAUNCH=(python)
  GPU_FLAGS=(--gpu)
else
  PYLAUNCH=(python)
  GPU_FLAGS=()
fi

if [[ "${RUN_FAST_MODE:-}" == "1" ]]; then
  FAST_FLAGS=(--fast_mode)
else
  FAST_FLAGS=()
fi

COMMON=(
  "${PYLAUNCH[@]}" scripts/run_optimized_pipeline.py --relation CtD
  --full_graph_embeddings --embedding_method RotatE
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard
  "${FAST_FLAGS[@]}"
  "${GPU_FLAGS[@]}"
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2
  --qsvc_C 0.1 --qml_pre_pca_dim 24
  --run_ensemble --ensemble_method stacking --tune_classical
  --results_dir "${RESULTS_DIR}"
)

invoke() {
  if [[ "${EXECUTE}" == "1" ]]; then
    echo ">>> GPU/cuStateVec ensemble run -> ${RESULTS_DIR} ..."
    "${COMMON[@]}"
  else
    echo "# Ensemble + quantum dry-run (EXECUTE=1 | RUN_ON_DGX=1 | USE_GPU_FLAG=1 | RESULTS_DIR=)"
    printf '%q ' "${COMMON[@]}"
    echo
    echo
  fi
}

echo "cwd: ${ROOT}"
echo "EXECUTE=${EXECUTE} RESULTS_DIR=${RESULTS_DIR} RUN_ON_DGX=${RUN_ON_DGX:-0} USE_GPU_FLAG=${USE_GPU_FLAG:-0}"
echo

invoke
