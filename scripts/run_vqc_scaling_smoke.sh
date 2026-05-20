#!/usr/bin/env bash
# §2 VQC scaling (iteration + ansatz depth) — docs/roadmap/04_quantum_scaling.md
# Uses --qml_max_iter and --vqc_ansatz_reps (see run_optimized_pipeline.py --help).
#
# Usage:
#   ./scripts/run_vqc_scaling_smoke.sh
#   QML_MAX_ITER=300 RUN_ON_DGX=1 EXECUTE=1 ./scripts/run_vqc_scaling_smoke.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE="${EXECUTE:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/vqc_200iter}"
QML_MAX_ITER="${QML_MAX_ITER:-200}"
VQC_ANSATZ_REPS="${VQC_ANSATZ_REPS:-6}"

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

COMMON=(
  "${PYLAUNCH[@]}" scripts/run_optimized_pipeline.py --relation CtD
  --full_graph_embeddings --embedding_method RotatE
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard
  "${FAST_FLAGS[@]}"
  --vqc_only --qml_dim 8 --vqc_optimizer SPSA
  --qml_max_iter "${QML_MAX_ITER}"
  --vqc_ansatz_type RealAmplitudes --vqc_ansatz_reps "${VQC_ANSATZ_REPS}"
  --results_dir "${RESULTS_DIR}"
)

invoke() {
  if [[ "${EXECUTE}" == "1" ]]; then
    echo ">>> VQC scaling run (qml_max_iter=${QML_MAX_ITER}, vqc_ansatz_reps=${VQC_ANSATZ_REPS}) ..."
    "${COMMON[@]}"
  else
    echo "# VQC dry-run (EXECUTE=1 | RUN_ON_DGX=1 | QML_MAX_ITER= | VQC_ANSATZ_REPS=)"
    printf '%q ' "${COMMON[@]}"
    echo
    echo
  fi
}

echo "cwd: ${ROOT}"
echo "EXECUTE=${EXECUTE} QML_MAX_ITER=${QML_MAX_ITER} VQC_ANSATZ_REPS=${VQC_ANSATZ_REPS}"
echo

invoke
