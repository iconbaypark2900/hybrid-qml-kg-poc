#!/usr/bin/env bash
# Campaign 3 fair classical vs quantum comparison (docs/roadmap/02_scientific_gaps.md §1).
# By default prints the four condition commands without running them (safe for CI / discovery).
#
# Usage:
#   ./scripts/run_campaign3_ablation.sh           # dry-run all (A,B,C,D)
#   ./scripts/run_campaign3_ablation.sh B C       # dry-run selected conditions only
#   EXECUTE=1 ./scripts/run_campaign3_ablation.sh B   # actually run condition B (~hours each)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE="${EXECUTE:-0}"
RESULTS_PARENT="${RESULTS_PARENT:-results/campaign3_ablation}"

if [[ "${RUN_FAST_MODE:-}" == "1" ]]; then
  FAST_FLAGS=(--fast_mode)
else
  FAST_FLAGS=()
fi

COMMON=(
  python scripts/run_optimized_pipeline.py --relation CtD
  --full_graph_embeddings --embedding_method RotatE
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard
  "${FAST_FLAGS[@]}"
)

invoke() {
  local label="$1"
  shift
  if [[ "${EXECUTE}" == "1" ]]; then
    echo ">>> Executing condition ${label} ..."
    "$@"
  else
    echo "# Condition ${label} (dry-run; set EXECUTE=1 to run)"
    printf '%q ' "$@"
    echo
    echo
  fi
}

run_A() {
  invoke A "${COMMON[@]}" \
    --classical_only \
    --results_dir "${RESULTS_PARENT}_A"
}

run_B() {
  invoke B "${COMMON[@]}" \
    --classical_only --restrict_classical_to_qml_dim --qml_dim 16 \
    --results_dir "${RESULTS_PARENT}_B"
}

run_C() {
  invoke C "${COMMON[@]}" \
    --quantum_only --qml_dim 16 --qml_feature_map Pauli \
    --qml_feature_map_reps 2 --qsvc_C 0.1 \
    --results_dir "${RESULTS_PARENT}_C"
}

run_D() {
  invoke D "${COMMON[@]}" \
    --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2 \
    --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
    --results_dir "${RESULTS_PARENT}_D"
}

dispatch_condition() {
  local u="${1^^}"
  case "${u}" in
    A) run_A ;;
    B) run_B ;;
    C) run_C ;;
    D) run_D ;;
    *)
      echo "Unknown condition '${1}'. Use one of: A B C D" >&2
      exit 2
      ;;
  esac
}

if [[ $# -eq 0 ]]; then
  ORDER=(A B C D)
else
  ORDER=("$@")
fi

echo "cwd: ${ROOT}"
echo "EXECUTE=${EXECUTE} RESULTS_PARENT_BASE=${RESULTS_PARENT}"
if [[ ${#FAST_FLAGS[@]} -gt 0 ]]; then
  echo "FAST_FLAGS: ${FAST_FLAGS[*]} (RUN_FAST_MODE=1)"
else
  echo "FAST_FLAGS: none (set RUN_FAST_MODE=1 to add --fast_mode)"
fi
echo

for letter in "${ORDER[@]}"; do
  dispatch_condition "${letter}"
done
