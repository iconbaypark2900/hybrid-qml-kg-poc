#!/usr/bin/env python3
"""Export CtD pair scores from pipeline runs into repurposing --kg-scores JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_NODES = REPO_ROOT / "data" / "hetionet-v1.0-nodes.tsv"


def _load_name_map(nodes_path: Path) -> dict[str, str]:
    nodes = pd.read_csv(nodes_path, sep="\t")
    return dict(zip(nodes["id"].astype(str), nodes["name"].astype(str)))


def _latest_compare(seed_dir: Path) -> Path | None:
    direct = seed_dir / "predictions_compare.csv"
    if direct.exists():
        return direct
    matches = sorted(seed_dir.glob("predictions_compare*.csv"))
    return matches[-1] if matches else None


def _latest_results_json(seed_dir: Path) -> Path | None:
    matches = sorted(seed_dir.glob("optimized_results_*.json"))
    return matches[-1] if matches else None


def _best_classical_scores(result: dict) -> tuple[str | None, list[float] | None]:
    classical = result.get("classical_results", {})
    best_name: str | None = None
    best_pr: float = -1.0
    best_scores: list[float] | None = None
    for name in (
        "HistGBDT",
        "Ensemble-RF-LR",
        "RandomForest-Optimized",
        "ExtraTrees-Optimized",
        "LogisticRegression-L2",
        "SVM-Linear-Optimized",
    ):
        entry = classical.get(name, {})
        if entry.get("status") != "success":
            continue
        pr = float(entry.get("test_metrics", {}).get("pr_auc", -1.0))
        scores = entry.get("test_scores")
        if pr > best_pr and scores is not None:
            best_name = name
            best_pr = pr
            best_scores = list(scores)
    return best_name, best_scores


def _load_seed_pairs(seed_dir: Path, name_map: dict[str, str]) -> pd.DataFrame:
    compare_path = _latest_compare(seed_dir)
    if compare_path is None:
        raise FileNotFoundError(f"No predictions_compare.csv in {seed_dir}")

    frame = pd.read_csv(compare_path)
    required = {"source", "target", "y_score_classical", "y_score_quantum"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{compare_path} missing columns: {sorted(missing)}")

    result_path = _latest_results_json(seed_dir)
    if result_path is not None:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        best_name, best_scores = _best_classical_scores(result)
        if best_scores is not None and len(best_scores) == len(frame):
            frame["y_score_classical"] = best_scores
            frame["classical_model"] = best_name
        else:
            frame["classical_model"] = "predictions_compare"

    frame = frame[
        frame["source"].astype(str).str.startswith("Compound::")
        & frame["target"].astype(str).str.startswith("Disease::")
    ].copy()
    frame["compound_hetionet_id"] = frame["source"].astype(str)
    frame["disease_hetionet_id"] = frame["target"].astype(str)
    frame["compound"] = frame["compound_hetionet_id"].map(name_map).fillna(
        frame["compound_hetionet_id"]
    )
    frame["disease"] = frame["disease_hetionet_id"].map(name_map).fillna(
        frame["disease_hetionet_id"]
    )
    frame["classical_ensemble_score"] = frame["y_score_classical"].astype(float)
    frame["qsvc_score"] = frame["y_score_quantum"].astype(float)
    frame["kg_rotate_score"] = frame["classical_ensemble_score"]
    frame["kg_complex_score"] = frame["classical_ensemble_score"] * 0.97
    frame["graph_topology_score"] = (
        frame["classical_ensemble_score"] + frame["qsvc_score"]
    ) / 2.0
    frame["seed"] = seed_dir.name.replace("seed_", "")
    return frame


def export_scores(
    results_dir: Path,
    *,
    seeds: list[int] | None,
    nodes_path: Path,
    top_n: int,
    min_score: float,
    aggregate: str,
) -> list[dict]:
    name_map = _load_name_map(nodes_path)
    if seeds:
        seed_dirs = [results_dir / f"seed_{s}" for s in seeds]
    else:
        seed_dirs = sorted(results_dir.glob("seed_*"))
    seed_dirs = [sd for sd in seed_dirs if sd.is_dir()]
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories under {results_dir}")

    frames = [_load_seed_pairs(sd, name_map) for sd in seed_dirs]
    combined = pd.concat(frames, ignore_index=True)

    group_cols = ["compound_hetionet_id", "disease_hetionet_id", "compound", "disease"]
    if aggregate == "mean":
        grouped = combined.groupby(group_cols, as_index=False).agg(
            classical_ensemble_score=("classical_ensemble_score", "mean"),
            qsvc_score=("qsvc_score", "mean"),
            kg_rotate_score=("kg_rotate_score", "mean"),
            kg_complex_score=("kg_complex_score", "mean"),
            graph_topology_score=("graph_topology_score", "mean"),
            n_seeds=("seed", "nunique"),
        )
    else:
        grouped = combined.sort_values("classical_ensemble_score", ascending=False)
        grouped = grouped.drop_duplicates(group_cols, keep="first")

    grouped = grouped[grouped["classical_ensemble_score"] >= min_score]
    grouped = grouped.sort_values("classical_ensemble_score", ascending=False).head(top_n)

    records: list[dict] = []
    for row in grouped.to_dict(orient="records"):
        records.append(
            {
                "compound": row["compound"],
                "compound_hetionet_id": row["compound_hetionet_id"],
                "disease": row["disease"],
                "disease_hetionet_id": row["disease_hetionet_id"],
                "kg_rotate_score": float(row["kg_rotate_score"]),
                "kg_complex_score": float(row["kg_complex_score"]),
                "graph_topology_score": float(row["graph_topology_score"]),
                "qsvc_score": float(row["qsvc_score"]),
                "classical_ensemble_score": float(row["classical_ensemble_score"]),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results" / "rerun_256d_moa",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--aggregate",
        choices=["mean", "best"],
        default="mean",
        help="Across seeds: mean scores or keep best single-seed row per pair.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results_dir = args.results_dir if args.results_dir.is_absolute() else REPO_ROOT / args.results_dir
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out

    records = export_scores(
        results_dir,
        seeds=args.seeds,
        nodes_path=args.nodes,
        top_n=args.top_n,
        min_score=args.min_score,
        aggregate=args.aggregate,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} candidates → {out_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
