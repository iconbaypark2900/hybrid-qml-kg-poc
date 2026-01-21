#!/usr/bin/env python3
"""
Compare ideal vs noisy simulator runs from experiment_history.csv.
"""

import argparse
from pathlib import Path
from typing import List

import pandas as pd


def build_comparison_table(df_history: pd.DataFrame, metric: str) -> pd.DataFrame:
    required_cols = {"execution_mode", "noise_model", "backend_label", metric}
    missing = required_cols - set(df_history.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df_history = df_history.copy()
    df_history["run_index"] = df_history.index
    group_cols: List[str] = ["execution_mode", "noise_model", "backend_label"]

    # Take the latest row per execution group
    latest = (
        df_history.sort_values("run_index")
        .groupby(group_cols, dropna=False)
        .tail(1)
        .reset_index(drop=True)
    )

    columns = group_cols + ["run_index", metric]
    if "classical_pr_auc" in latest.columns:
        columns.append("classical_pr_auc")
    if "quantum_pr_auc" in latest.columns and metric != "quantum_pr_auc":
        columns.append("quantum_pr_auc")

    return latest[columns].sort_values(group_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ideal vs noisy runs.")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--metric", type=str, default="quantum_pr_auc", help="Metric to compare")
    parser.add_argument("--out_csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    history_path = results_dir / "experiment_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing {history_path}")

    df_history = pd.read_csv(history_path)
    comparison = build_comparison_table(df_history, args.metric)

    out_csv = Path(args.out_csv) if args.out_csv else results_dir / "ideal_vs_noisy_comparison.csv"
    comparison.to_csv(out_csv, index=False)
    print(f"Wrote comparison to {out_csv}")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
