#!/usr/bin/env bash
# run_full_repurposing_pipeline.sh — Full end-to-end drug repurposing pipeline.
# Usage: ./scripts/dgx/run_full_repurposing_pipeline.sh [--mode kg-only|kg+omics] [--validate]
set -euo pipefail

MODE="kg+omics"
VALIDATE=""

for arg in "$@"; do
    case $arg in
        --mode) shift; MODE="$1"; shift ;;
        --mode=*) MODE="${arg#*=}" ;;
        --validate) VALIDATE="--validate" ;;
    esac
done

echo "=== Full Repurposing Pipeline ==="
echo "Mode: $MODE"
echo ""

python3 scripts/run_full_repurposing_pipeline.py --mode "$MODE" $VALIDATE

echo ""
echo "=== Pipeline complete. Artifacts in artifacts/ ==="
