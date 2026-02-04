#!/usr/bin/env python3
"""
Run a small sweep over embedding + QML settings and summarize results.

This wrapper calls scripts/run_optimized_pipeline.py repeatedly and collects
the latest_run.csv metrics into a single summary CSV for quick ranking.
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
LATEST_RUN = RESULTS_DIR / "latest_run.csv"


def _split_list(val: str) -> List[str]:
    return [v.strip() for v in val.split(",") if v.strip()]


def _read_latest_run() -> Dict[str, Any]:
    if not LATEST_RUN.exists():
        return {}
    import pandas as pd
    df = pd.read_csv(LATEST_RUN)
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return row


def _run_once(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        return {"status": "failed", "returncode": proc.returncode}
    row = _read_latest_run()
    row["status"] = "ok"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep embedding + QML settings and summarize results.")
    parser.add_argument("--relation", type=str, default="CtD", help="Target relation (default: CtD)")
    parser.add_argument("--embedding_methods", type=str, default="ComplEx,RotatE",
                        help="Comma-separated embedding methods.")
    parser.add_argument("--embedding_dims", type=str, default="64,128",
                        help="Comma-separated embedding dims.")
    parser.add_argument("--qml_encodings", type=str, default="hybrid,optimized_diff,tensor_product",
                        help="Comma-separated QML encoding strategies.")
    parser.add_argument("--qml_pre_pca_dim", type=int, default=128, help="Pre-PCA dimension for QML.")
    parser.add_argument("--qml_feature_selection_method", type=str, default="f_classif",
                        choices=["mutual_info", "f_classif", "variance", "none"])
    parser.add_argument("--qml_feature_select_k_mult", type=float, default=6.0)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--negative_sampling", type=str, default="hard")
    parser.add_argument("--full_graph_embeddings", action="store_true", help="Use full-graph embeddings.")
    parser.add_argument("--pos_edge_sample", type=int, default=0, help="Optional positive edge sample size.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="", help="Output CSV path.")
    args = parser.parse_args()

    methods = _split_list(args.embedding_methods)
    dims = [int(x) for x in _split_list(args.embedding_dims)]
    encodings = _split_list(args.qml_encodings)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_csv = Path(args.out_csv) if args.out_csv else RESULTS_DIR / f"qml_grid_{ts}.csv"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for method in methods:
        for dim in dims:
            for enc in encodings:
                cmd = [
                    sys.executable, "scripts/run_optimized_pipeline.py",
                    "--relation", args.relation,
                    "--embedding_method", method,
                    "--embedding_dim", str(dim),
                    "--qml_encoding", enc,
                    "--qml_pre_pca_dim", str(args.qml_pre_pca_dim),
                    "--qml_feature_selection_method", str(args.qml_feature_selection_method),
                    "--qml_feature_select_k_mult", str(args.qml_feature_select_k_mult),
                    "--neg_ratio", str(args.neg_ratio),
                    "--negative_sampling", str(args.negative_sampling),
                    "--random_state", str(args.random_state),
                ]
                if args.full_graph_embeddings:
                    cmd.append("--full_graph_embeddings")
                if args.pos_edge_sample and args.pos_edge_sample > 0:
                    cmd.extend(["--pos_edge_sample", str(args.pos_edge_sample)])

                print(f"\n=== Running: method={method} dim={dim} enc={enc} ===")
                result = _run_once(cmd)
                result.update({
                    "embedding_method": method,
                    "embedding_dim": dim,
                    "qml_encoding": enc,
                    "qml_pre_pca_dim": args.qml_pre_pca_dim,
                    "qml_feature_selection_method": args.qml_feature_selection_method,
                    "qml_feature_select_k_mult": args.qml_feature_select_k_mult,
                    "neg_ratio": args.neg_ratio,
                    "negative_sampling": args.negative_sampling,
                    "pos_edge_sample": args.pos_edge_sample,
                })
                rows.append(result)

    # Write summary CSV
    if rows:
        keys = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote summary → {out_csv}")
    else:
        print("No results produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
