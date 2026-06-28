#!/usr/bin/env bash
# Check progress of Tier 1 experiment runs.
# Usage: ./scripts/check_tier1_progress.sh
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=============================================================================="
echo "TIER 1 RUN PROGRESS  ($(date '+%Y-%m-%d %H:%M %Z'))"
echo "=============================================================================="

declare -A RUNS=(
    ["MoA benchmark"]="results/moa_benchmark/run.log"
    ["CpD relation"]="results/cpd_run/run.log"
    ["Multi-seed seed42"]="results/multiseed/seed_42.log"
)

for label in "MoA benchmark" "CpD relation" "Multi-seed seed42"; do
    log="${RUNS[$label]}"
    dir=$(dirname "$log")
    echo ""
    echo "--- $label ---"
    if [ -f "$log" ]; then
        pct=$(grep -oP 'Training epochs on cpu:\s+\K\d+%' "$log" | tail -1)
        echo "  progress: ${pct:-starting...}"
        loss=$(grep -oP 'loss=[\d.]+' "$log" | tail -1)
        [ -n "$loss" ] && echo "  $loss"
        prauc=$(grep -oP 'Test PR-AUC:\s+[\d.]+' "$log" | tail -5)
        [ -n "$prauc" ] && echo "  $prauc"
    else
        echo "  No log found at $log"
    fi
    result=$(ls -1 "$dir"/optimized_results_*.json 2>/dev/null | head -1)
    if [ -n "$result" ]; then
        echo "  RESULT: $result"
    else
        echo "  (no results JSON yet)"
    fi
done

echo ""
echo "=============================================================================="
echo "RUNNING PROCESSES"
echo "=============================================================================="
ps -o pid,etimes,pcpu,rss,comm -C python 2>/dev/null || echo "  (none)"
