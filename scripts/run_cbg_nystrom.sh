#!/usr/bin/env bash
# §1 CbG + Nyström — docs/roadmap/04_quantum_scaling.md
# Adjust --qsvc_nystrom_m after the CtD sweep picks m (default 400).
#
# Usage:
#   ./scripts/run_cbg_nystrom.sh
#   NYSTROM_M=200 RUN_ON_DGX=1 EXECUTE=1 ./scripts/run_cbg_nystrom.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE="${EXECUTE:-0}"
NYSTROM_M="${NYSTROM_M:-400}"
RESULTS_DIR="${RESULTS_DIR:-results/cbg_nystrom}"

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
  "${PYLAUNCH[@]}" scripts/run_optimized_pipeline.py --relation CbG
  --full_graph_embeddings --embedding_method RotatE
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard
  "${FAST_FLAGS[@]}"
  --quantum_only --qml_dim 16 --qml_feature_map Pauli
  --qml_feature_map_reps 2 --qsvc_C 0.1
  --qsvc_nystrom_m "${NYSTROM_M}"
  --results_dir "${RESULTS_DIR}"
)

invoke() {
  if [[ "${EXECUTE}" == "1" ]]; then
    echo ">>> CbG Nyström (m=${NYSTROM_M}) ..."
    "${COMMON[@]}"
  else
    echo "# CbG + --qsvc_nystrom_m ${NYSTROM_M} (dry-run; EXECUTE=1 | NYSTROM_M= | RUN_ON_DGX=1)"
    printf '%q ' "${COMMON[@]}"
    echo
    echo
  fi
}

echo "cwd: ${ROOT}"
echo "EXECUTE=${EXECUTE} NYSTROM_M=${NYSTROM_M} RESULTS_DIR=${RESULTS_DIR}"
echo

invoke
