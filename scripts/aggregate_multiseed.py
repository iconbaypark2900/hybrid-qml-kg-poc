#!/usr/bin/env python3
"""Aggregate multi-seed pipeline results into mean ± std for Table 3.

Reads seed_*/optimized_results_*.json under a multiseed results directory and
computes mean ± std PR-AUC for RF, ET, QSVC, and ensemble.

Run:
  .venv/bin/python scripts/aggregate_multiseed.py
  .venv/bin/python scripts/aggregate_multiseed.py \\
      --results-dir results/multiseed --seeds 42 7 13 99 2026 \\
      --out results/multiseed/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MULTISEED_DIR = REPO_ROOT / "results" / "multiseed"


def _pr_auc(entry: dict) -> float | None:
    """Test PR-AUC from a classical/quantum/ensemble result entry."""
    if not isinstance(entry, dict) or entry.get("status") != "success":
        return None
    test_metrics = entry.get("test_metrics")
    if isinstance(test_metrics, dict) and "pr_auc" in test_metrics:
        return float(test_metrics["pr_auc"])
    if "pr_auc" in entry:
        return float(entry["pr_auc"])
    return None


def _extract_metrics(result: dict) -> dict[str, float]:
    metrics: dict[str, float] = {}
    classical = result.get("classical_results", {})

    for name in ("RandomForest-Optimized", "RandomForest"):
        if name in classical:
            value = _pr_auc(classical[name])
            if value is not None:
                metrics["RF"] = value
                break

    for name in ("ExtraTrees-Optimized", "ExtraTrees"):
        if name in classical:
            value = _pr_auc(classical[name])
            if value is not None:
                metrics["ET"] = value
                break

    if "HistGBDT" in classical:
        value = _pr_auc(classical["HistGBDT"])
        if value is not None:
            metrics["HistGBDT"] = value

    quantum = result.get("quantum_results", {})
    for name in ("QSVC-Optimized", "QSVC", "QSVC-Optimized-Pauli"):
        if name in quantum:
            value = _pr_auc(quantum[name])
            if value is not None:
                metrics["QSVC"] = value
                break

    ensemble = result.get("ensemble_results", {})
    for name in ("Ensemble-QC-stacking", "stacking", "stacking_ensemble"):
        if name in ensemble:
            value = _pr_auc(ensemble[name])
            if value is not None:
                metrics["Ensemble"] = value
                break

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed PR-AUC results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_MULTISEED_DIR,
        help="Directory containing seed_<N>/ subdirs (default: results/multiseed)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Optional seed list; only aggregate these seeds (default: all seed_* dirs)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <results-dir>/multiseed_summary.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    multiseed_dir = args.results_dir if args.results_dir.is_absolute() else REPO_ROOT / args.results_dir
    out_path = args.out
    if out_path is None:
        out_path = multiseed_dir / "multiseed_summary.json"
    elif not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    if args.seeds:
        seed_dirs = [multiseed_dir / f"seed_{s}" for s in args.seeds]
    else:
        seed_dirs = sorted(multiseed_dir.glob("seed_*"))

    seed_dirs = [sd for sd in seed_dirs if sd.is_dir()]
    if not seed_dirs:
        print(f"No seed directories found in {multiseed_dir}", file=sys.stderr)
        return 1

    all_metrics: dict[str, list[float]] = {
        "HistGBDT": [],
        "RF": [],
        "ET": [],
        "QSVC": [],
        "Ensemble": [],
    }
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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": seeds_found, "summary": summary}, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
