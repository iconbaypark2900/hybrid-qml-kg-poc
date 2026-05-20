#!/usr/bin/env bash
# §3 Noisy Simulator Benchmark — docs/roadmap/02_scientific_gaps.md
# Uses config/quantum_config_noisy.yaml (depolarizing + simulator ZNE / readout stubs).
#
# Prints the invocation without executing unless EXECUTE=1 (same ergonomics as
# scripts/run_cv_feasibility_smoke.sh and scripts/run_campaign3_ablation.sh).
# For iteration: RUN_FAST_MODE=1 forwards --fast_mode into run_optimized_pipeline.py.
#
# Usage:
#   ./scripts/run_noisy_sim_smoke.sh
#   EXECUTE=1 RUN_FAST_MODE=1 ./scripts/run_noisy_sim_smoke.sh
#   RESULTS_DIR=results/noisy_sim_quick EXECUTE=1 ./scripts/run_noisy_sim_smoke.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXECUTE="${EXECUTE:-0}"
RESULTS_DIR="${RESULTS_DIR:-results/noisy_sim}"

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
  --quantum_only --qml_dim 16 --qml_feature_map Pauli
  --qml_feature_map_reps 2 --qsvc_C 0.1
  --quantum_config_path config/quantum_config_noisy.yaml
  --results_dir "${RESULTS_DIR}"
)

invoke() {
  if [[ "${EXECUTE}" == "1" ]]; then
    echo ">>> Executing noisy-simulator QSVC benchmark ..."
    "${COMMON[@]}"
  else
    echo "# Dry-run noisy simulator (ideal vs noisy tiers must not be mixed — see roadmap §3)"
    echo "# set EXECUTE=1 | RUN_FAST_MODE=1 | RESULTS_DIR=..."
    printf '%q ' "${COMMON[@]}"
    echo
    echo
  fi
}

invoke
