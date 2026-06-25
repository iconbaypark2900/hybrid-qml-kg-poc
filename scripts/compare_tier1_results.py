"""Compare Tier 1 experiment results against the baseline run.

Usage:
    .venv/bin/python scripts/compare_tier1_results.py
"""
from __future__ import annotations
import json
from pathlib import Path

BASELINE = Path("results/optimized_results_20260216-100431.json")
RUNS = {
    "MoA benchmark": Path("results/moa_benchmark"),
    "CpD relation": Path("results/cpd_run"),
    "Multi-seed (seed 42)": Path("results/multiseed/seed_42"),
}


def find_result_json(dir_path: Path) -> Path | None:
    if not dir_path.exists():
        return None
    candidates = sorted(dir_path.glob("optimized_results_*.json"), reverse=True)
    return candidates[0] if candidates else None


def extract_ranking(path: Path) -> list[dict]:
    with open(path) as f:
        d = json.load(f)
    return d.get("ranking", [])


def fmt(pr_auc):
    if pr_auc is None:
        return "   N/A"
    return f"{pr_auc:.4f}"


def main():
    print("=" * 78)
    print("TIER 1 RESULTS COMPARISON")
    print("=" * 78)

    baseline_ranking = extract_ranking(BASELINE) if BASELINE.exists() else []
    baseline_best = baseline_ranking[0] if baseline_ranking else {}

    print(f"\nBaseline: {BASELINE.name}")
    if baseline_ranking:
        for i, m in enumerate(baseline_ranking[:5], 1):
            print(f'  {i}. {m["name"]:30s} ({m["type"]:9s}) PR-AUC={fmt(m["pr_auc"])}')

    results = {}
    for label, dir_path in RUNS.items():
        path = find_result_json(dir_path)
        if path is None:
            print(f"\n{label}: NOT YET COMPLETE (no results JSON found in {dir_path})")
            results[label] = None
            continue
        ranking = extract_ranking(path)
        results[label] = ranking
        print(f"\n{label}: {path.name}")
        for i, m in enumerate(ranking[:5], 1):
            print(f'  {i}. {m["name"]:30s} ({m["type"]:9s}) PR-AUC={fmt(m["pr_auc"])}')

    print("\n" + "=" * 78)
    print("DELTA vs BASELINE (best overall PR-AUC)")
    print("=" * 78)
    base_pr = baseline_best.get("pr_auc")
    print(f'  Baseline best: {baseline_best.get("name", "?")} = {fmt(base_pr)}')
    for label, ranking in results.items():
        if ranking:
            best = ranking[0]
            delta = (best["pr_auc"] - base_pr) if base_pr else None
            sign = "+" if delta and delta >= 0 else ""
            if delta is not None:
                print(f'  {label:25s} best={best["name"]:30s} {fmt(best["pr_auc"])}  ({sign}{delta:.4f})')
            else:
                print(f'  {label:25s} best={best["name"]:30s} {fmt(best["pr_auc"])}')

    print("\n" + "=" * 78)
    print("BY MODEL TYPE")
    print("=" * 78)
    for label, ranking in results.items():
        if not ranking:
            continue
        by_type = {}
        for m in ranking:
            t = m["type"]
            if t not in by_type or m["pr_auc"] > by_type[t]["pr_auc"]:
                by_type[t] = m
        print(f"\n{label}:")
        for t in ["classical", "quantum", "ensemble"]:
            if t in by_type:
                m = by_type[t]
                print(f'  best {t:10s}: {m["name"]:30s} PR-AUC={fmt(m["pr_auc"])}')


if __name__ == "__main__":
    main()
