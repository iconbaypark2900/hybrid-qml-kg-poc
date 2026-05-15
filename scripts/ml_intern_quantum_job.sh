#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a quantum job through ml-intern using the repo-owned deterministic runner.

Usage:
  ./scripts/ml_intern_quantum_job.sh <recipe> [runner args...]

Recipes:
  simulator-smoke
  simulator-full
  heron-dry-run
  heron-run

Add --dry-run to print the ml-intern prompt without invoking ml-intern.
Real Heron hardware jobs still require --confirm-hardware in the runner args.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

RECIPE="$1"
shift

DRY_RUN=0
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

quote_args() {
  local quoted=""
  for arg in "$@"; do
    printf -v piece "%q" "$arg"
    quoted+="${piece} "
  done
  printf "%s" "${quoted% }"
}

RUNNER_ARGS="$(quote_args "$@")"
RUNNER_COMMAND="python3 scripts/quantum_job_runner.py ${RECIPE}"
if [[ -n "$RUNNER_ARGS" ]]; then
  RUNNER_COMMAND+=" ${RUNNER_ARGS}"
fi

PROMPT="$(cat <<EOF
You are orchestrating a controlled quantum job for the Hybrid QML-KG project.

Run only this command from the repository root:

${RUNNER_COMMAND}

Rules:
- Do not print IBM Quantum tokens or environment variables containing secrets.
- Do not paste, request, or echo IBM credentials.
- Do not submit real IBM hardware work unless the command already includes --confirm-hardware.
- If the command fails, report the failing command, exit code, and relevant non-secret log lines.
- If it succeeds, inspect the configured results directory when available and summarize output artifacts.
- Do not broaden scope beyond this single job.
EOF
)"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "ml-intern dry-run"
  echo "Command:"
  echo "${RUNNER_COMMAND}"
  echo ""
  echo "Prompt:"
  echo "${PROMPT}"
  exit 0
fi

if ! command -v ml-intern >/dev/null 2>&1; then
  echo "Error: ml-intern not found on PATH. Install it first from https://github.com/huggingface/ml-intern." >&2
  exit 127
fi

ml-intern "${PROMPT}"
