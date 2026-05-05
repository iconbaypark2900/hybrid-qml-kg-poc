#!/usr/bin/env bash
# H1/H1b paired-bootstrap CI on real Hetionet — tuned for NVIDIA DGX Spark / cuStateVec.
# Verifies the GPU stack first (qiskit-aer-gpu via scripts/verify_qiskit_gpu.py),
# then runs scripts/run_bootstrap_ci.py --gpu over 5 stratified folds.
#
# Usage (from repo root):
#   ./scripts/run_bootstrap_ci_dgx.sh
#   LOG_PATH=/path/to/run.log ./scripts/run_bootstrap_ci_dgx.sh
#   RESUME_FROM_CACHE=1 ./scripts/run_bootstrap_ci_dgx.sh   # skip CV training, re-emit report
#
# Optional environment overrides:
#   N_FOLDS, N_RESAMPLES, CACHE_DIR, RESUME_FROM_CACHE,
#   SKIP_QSVC, SKIP_ENSEMBLE, RESULTS_DIR, PYTHON (python binary)
#
# See docs/deployment/DGX_BOOTSTRAP_CI.md for the full workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY="${PYTHON:-}"
if [[ -z "$PY" ]] && [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PY="$PROJECT_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="python3"
fi

N_FOLDS="${N_FOLDS:-5}"
N_RESAMPLES="${N_RESAMPLES:-10000}"
CACHE_DIR="${CACHE_DIR:-results/cv_predictions}"
RESULTS_DIR="${RESULTS_DIR:-results}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/$RESULTS_DIR/bootstrap_ci_dgx_${STAMP}.log}"
mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$PROJECT_ROOT/$CACHE_DIR"

echo "=== Bootstrap CI (DGX-oriented, cuStateVec) ==="
echo "Repo:        $PROJECT_ROOT"
echo "Python:      $PY"
echo "Log:         $LOG_PATH"
echo "Folds:       $N_FOLDS | resamples=$N_RESAMPLES | cache=$CACHE_DIR"
if [[ -n "${RESUME_FROM_CACHE:-}" ]]; then
  echo "Mode:        resume from cache (skip CV training, re-emit report)"
fi
if [[ -n "${SKIP_QSVC:-}" ]]; then
  echo "Skipping:    QSVC (debug)"
fi
if [[ -n "${SKIP_ENSEMBLE:-}" ]]; then
  echo "Skipping:    stacking ensemble (debug)"
fi
echo ""

if command -v nvidia-smi &>/dev/null; then
  echo "--- nvidia-smi (summary) ---"
  nvidia-smi -L 2>/dev/null || true
  echo ""
fi

# Step 1: verify qiskit-aer-gpu / cuStateVec is wired up before launching a
# multi-hour-equivalent run on a misconfigured stack. The verify script
# exits non-zero with concrete remediation on any check failure; --gpu in
# the driver also has a strict gate that aborts on the same condition.
# We skip the GPU verification when SKIP_QSVC is set (no QSVC = no GPU need).
if [[ -z "${SKIP_QSVC:-}" ]]; then
  echo "--- Step 1/2: verify qiskit-aer-gpu / cuStateVec ---"
  "$PY" scripts/verify_qiskit_gpu.py
  echo ""
fi

echo "--- Step 2/2: bootstrap CI driver (tee to log) ---"

# Build the driver argv. --gpu only when QSVC is in scope; skip flags
# pass through; resume mode skips CV entirely.
DRIVER_ARGS=(
  --n_folds "$N_FOLDS"
  --n_resamples "$N_RESAMPLES"
  --cache_dir "$CACHE_DIR"
)
if [[ -z "${SKIP_QSVC:-}" ]]; then
  DRIVER_ARGS+=(--gpu)
else
  DRIVER_ARGS+=(--skip_qsvc)
fi
if [[ -n "${SKIP_ENSEMBLE:-}" ]]; then
  DRIVER_ARGS+=(--skip_ensemble)
fi
if [[ -n "${RESUME_FROM_CACHE:-}" ]]; then
  DRIVER_ARGS+=(--resume_from_cache)
fi

set +e
"$PY" scripts/run_bootstrap_ci.py "${DRIVER_ARGS[@]}" 2>&1 | tee "$LOG_PATH"
exit "${PIPESTATUS[0]}"
