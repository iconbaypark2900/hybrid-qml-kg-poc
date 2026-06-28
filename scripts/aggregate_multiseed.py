#!/usr/bin/env python3
"""Aggregate multi-seed pipeline results into mean ± std for Table 3.

Reads results/multiseed/seed_*/optimized_results_*.json and computes
mean ± std PR-AUC for RF, ET, QSVC, and ensemble across all seeds.

Run:  .venv/bin/python scripts/aggregate_multiseed.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MULTISEED_DIR = REPO_ROOT / "results" / "multiseed"


def _extract_metrics(result: dict) -> dict[str, float]:
    metrics = {}
    classical = result.get("classical_results", {})
    for name in ("RandomForest-Optimized", "RandomForest", "ExtraTrees-Optimized", "ExtraTrees"):
        if name in classical:
            entry = classical[name]
            if isinstance(entry, dict) and "pr_auc" in entry:
                metrics["RF" if "RandomForest" in name else "ET"] = float(entry["pr_auc"])
                break
    for name in ("ExtraTrees-Optimized", "ExtraTrees"):
        if name in classical:
            entry = classical[name]
            if isinstance(entry, dict) and "pr_auc" in entry:
                metrics["ET"] = float(entry["pr_auc"])
                break

    quantum = result.get("quantum_results", {})
    for name in ("QSVC-Optimized", "QSVC", "QSVC-Optimized-Pauli"):
        if name in quantum:
            entry = quantum[name]
            if isinstance(entry, dict) and "pr_auc" in entry:
                metrics["QSVC"] = float(entry["pr_auc"])
                break

    ensemble = result.get("ensemble_results", {})
    for name in ("stacking", "Ensemble-QC-stacking", "stacking_ensemble"):
        if name in ensemble:
            entry = ensemble[name]
            if isinstance(entry, dict) and "pr_auc" in entry:
                metrics["Ensemble"] = float(entry["pr_auc"])
                break

    return metrics


def main() -> int:
    seed_dirs = sorted(MULTISEED_DIR.glob("seed_*"))
    if not seed_dirs:
        print(f"No seed directories found in {MULTISEED_DIR}", file=sys.stderr)
        return 1

    all_metrics: dict[str, list[float]] = {"RF": [], "ET": [], "QSVC": [], "Ensemble": []}
    seeds_found = []

    for sd in seed_dirs:
        result_files = sorted(sd.glob("optimized_results_*.json"))
        if not result_files:
            print(f"WARNING: no result JSON in {sd.name}", file=sys.stderr)
            continue
        result = json.loads(result_files[-1].read_text(encoding="utf-8"))
        m = _extract_metrics(result)
        seeds_found.append(sd.name)
        for key in all_metrics:
            if key in m:
                all_metrics[key].append(m[key])

    print(f"Seeds found: {len(seeds_found)} — {seeds_found}\n")

    summary = {}
    for model, vals in all_metrics.items():
        if vals:
            arr = np.array(vals)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            summary[model] = {
                "mean": mean,
                "std": std,
                "n": len(vals),
                "values": [float(v) for v in arr],
                "formatted": f"{mean:.4f} ± {std:.4f}",
            }
            print(f"{model:12s}  PR-AUC = {mean:.4f} ± {std:.4f}  (n={len(vals)}, vals={[f'{v:.4f}' for v in arr]})")
        else:
            summary[model] = {"mean": None, "std": None, "n": 0, "values": [], "formatted": "N/A"}
            print(f"{model:12s}  PR-AUC = N/A (no data)")

    out = MULTISEED_DIR / "multiseed_summary.json"
    out.write_text(json.dumps({"seeds": seeds_found, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
