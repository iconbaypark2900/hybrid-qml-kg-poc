#!/usr/bin/env python3
"""Pipeline smoke test: invokes run_optimized_pipeline with --cheap_mode. Target: < 5 min."""

import sys
import os
import subprocess
import time
import json
import glob
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _ROOT / "results"


def main():
    print("Pipeline smoke test: run_optimized_pipeline --cheap_mode")
    start = time.time()

    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "run_optimized_pipeline.py"),
        "--relation", "CtD",
        "--cheap_mode",
    ]

    result = subprocess.run(cmd, cwd=str(_ROOT), capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nPipeline smoke failed (exit {result.returncode}) after {elapsed:.1f}s")
        return 1

    # Verify JSON output exists and has expected structure
    pat = str(RESULTS_DIR / "optimized_results_*.json")
    matches = sorted(glob.glob(pat), reverse=True)
    if not matches:
        print(f"\nPipeline ran but no optimized_results_*.json found in {RESULTS_DIR}")
        return 1

    with open(matches[0]) as f:
        data = json.load(f)
    if "ranking" not in data and "config" not in data:
        print("\nPipeline JSON missing expected keys (ranking or config)")
        return 1

    print(f"\nPipeline smoke passed in {elapsed:.1f}s")
    print(f"  JSON: {matches[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
