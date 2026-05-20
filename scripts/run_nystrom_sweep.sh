#!/usr/bin/env bash
# §1 Nyström landmark sweep — docs/roadmap/04_quantum_scaling.md
# Prints one pipeline command per m (50..800 by default) unless EXECUTE=1.
#
# Usage:
#   ./scripts/run_nystrom_sweep.sh
#   M_VALUES="100 200" ./scripts/run_nystrom_sweep.sh
#   RUN_ON_DGX=1 EXECUTE=1 ./scripts/run_nystrom_sweep.sh
#   RUN_FAST_MODE=1 EXECUTE=1 ./scripts/run_nystrom_sweep.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE="${EXECUTE:-0}"
RELATION="${RELATION:-CtD}"
RESULTS_PARENT="${RESULTS_PARENT:-results/nystrom_sweep}"
M_VALUES="${M_VALUES:-50 100 200 400 800}"

if [[ "${RUN_ON_DGX:-}" == "1" ]]; then
  PYLAUNCH=(env HYBRID_QML_SYSTEM=dgx python)
else
  PYLAUNCH=(python)
fi

if [[ "${RUN_FAST_MODE:-}" == "1" ]]; then
  FAST_FLAGS=(--fast_mode)
else
  FAST_FLAGS=()
fi

COMMON_HEAD=(
  "${PYLAUNCH[@]}" scripts/run_optimized_pipeline.py --relation "${RELATION}"
  --full_graph_embeddings --embedding_method RotatE
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard
  "${FAST_FLAGS[@]}"
  --quantum_only --qml_dim 16 --qml_feature_map Pauli
  --qml_feature_map_reps 2 --qsvc_C 0.1
)

invoke_m() {
  local m="$1"
  local out="${RESULTS_PARENT}_m${m}"
  if [[ "${EXECUTE}" == "1" ]]; then
    echo ">>> m=${m} -> ${out}"
    "${COMMON_HEAD[@]}" --qsvc_nystrom_m "${m}" --results_dir "${out}"
  else
    echo "# m=${m} (${out}) dry-run; EXECUTE=1 | RUN_ON_DGX=1 | M_VALUES='...'"
    printf '%q ' "${COMMON_HEAD[@]}" --qsvc_nystrom_m "${m}" --results_dir "${out}"
    echo
    echo
  fi
}

echo "cwd: ${ROOT}"
echo "EXECUTE=${EXECUTE} RELATION=${RELATION} RESULTS_PARENT=${RESULTS_PARENT}"
echo "M_VALUES=${M_VALUES} RUN_ON_DGX=${RUN_ON_DGX:-0} RUN_FAST_MODE=${RUN_FAST_MODE:-0}"
echo

for m in ${M_VALUES}; do
  invoke_m "${m}"
done
