#!/usr/bin/env bash
# Wrapper for flyctl that disables Depot remote builders by default.
#
# Fly may route builds through Depot ("Waiting for depot builder..."). If that
# step fails with 403 / internal error, building without Depot avoids it.
#
# Usage (from repo root):
#   ./scripts/fly_deploy.sh deploy -a hybrid-qml-kg-poc
#   ./scripts/fly_deploy.sh deploy --build-only --push -a hybrid-qml-kg-poc --image-label my-label
#
# To use Depot again explicitly: FLY_USE_DEPOT=1 fly deploy ...

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${FLY_USE_DEPOT:-}" == "1" ]]; then
  exec fly "$@"
fi

# Prepend --depot=false for deploy subcommand when not already set
args=("$@")
if [[ "${1:-}" == "deploy" ]]; then
  has_depot=false
  for a in "${args[@]}"; do
    if [[ "$a" == --depot=* ]] || [[ "$a" == --depot ]]; then
      has_depot=true
      break
    fi
  done
  if [[ "$has_depot" == false ]]; then
    args=(deploy --depot=false "${args[@]:1}")
  fi
fi

exec fly "${args[@]}"
