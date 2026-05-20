#!/usr/bin/env bash
# §2 "Practical solution — Nyström approximation for CV" in docs/roadmap/02_scientific_gaps.md
# Prints the feasibility command without running unless EXECUTE=1 (same ergonomics as
# scripts/run_campaign3_ablation.sh). Optional RUN_FAST_MODE=1 adds --fast_mode for local iteration.
#
# Usage:
#   ./scripts/run_cv_feasibility_smoke.sh           # dry-run (quoted command lines)
#   EXECUTE=1 ./scripts/run_cv_feasibility_smoke.sh
#   EXECUTE=1 RUN_FAST_MODE=1 ./scripts/run_cv_feasibility_smoke.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE="${EXECUTE:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/cv_quantum_smoke}"

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
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2
  --qsvc_C 0.1 --qml_pre_pca_dim 24
  --qsvc_nystrom_m 200
  --run_ensemble --ensemble_method stacking --tune_classical
  --use_cv_evaluation --cv_folds 5
  --quantum_config_path config/quantum_config_ideal.yaml
  --results_dir "${RESULTS_DIR}"
)

invoke() {
  if [[ "${EXECUTE}" == "1" ]]; then
    echo ">>> Executing CV feasibility run (Nyström + 5-fold) ..."
    "${COMMON[@]}"
  else
    echo "# Dry-run CV quantum + ensemble (--use_cv_evaluation --cv_folds 5 --qsvc_nystrom_m 200)"
    echo "# set EXECUTE=1 to run | RUN_FAST_MODE=1 for --fast_mode | RESULTS_DIR=... to relocate"
    printf '%q ' "${COMMON[@]}"
    echo
    echo
  fi
}

invoke
