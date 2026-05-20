#!/usr/bin/env python3
"""
§5 Score-validity inversion — quantitative metrics (docs/roadmap/02_scientific_gaps.md).

Computes Spearman correlation between model scores and ClinicalTrials-style trial counts,
plus simple top-K coverage stats. For paper-grade ρ on all test predictions, join
prediction exports with trial counts before passing --scores-csv (see roadmap).

Pairs with Figure 3: figures/fig3_clinical.py (same curated rows as docs/RESULTS_EVIDENCE.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

# Mirrors figures/fig3_clinical.py (clinical validation points for §7 narrative).
FIG3_PUBLISHED_ROWS: Tuple[Tuple[float, int], ...] = (
    (0.793, 0),  # Abacavir → ocular cancer
    (0.693, 0),  # Ezetimibe → gout
    (0.597, 0),  # Ramipril → stomach cancer
    (0.528, 7),  # Losartan → atherosclerosis
    (0.525, 7),  # Mitomycin → liver cancer
    (0.520, 0),  # Salmeterol → liver cancer
)


def _load_scores_trials_np(
    scores: Sequence[float],
    trials: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    t = np.asarray(trials, dtype=int)
    if s.ndim != 1 or t.ndim != 1 or s.shape != t.shape:
        raise ValueError("scores and trials must be 1D arrays of equal length")
    if s.size < 2:
        raise ValueError("need at least 2 paired observations for Spearman")
    return s, t


def _read_csv_pairs(path: Path, score_col: str, trials_col: str) -> Tuple[List[float], List[int]]:
    scores: List[float] = []
    trials: List[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header row in {path}")
        missing = [c for c in (score_col, trials_col) if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing columns {missing}; have {reader.fieldnames}")
        for row in reader:
            if row.get(score_col) in (None, "") or row.get(trials_col) in (None, ""):
                continue
            scores.append(float(row[score_col]))
            trials.append(int(float(row[trials_col])))
    return scores, trials


def _read_json_array(path: Path) -> Tuple[List[float], List[int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("JSON root must be a list of objects")
    scores: List[float] = []
    trials: List[int] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "score" not in item or "trial_count" not in item:
            raise ValueError('Each object needs "score" and "trial_count"')
        scores.append(float(item["score"]))
        trials.append(int(item["trial_count"]))
    return scores, trials


def top_k_stats(scores: np.ndarray, trials: np.ndarray, k: int) -> Tuple[float, float]:
    """Fraction of top-k (by descending score) with >=1 trial, and fraction with 0 trials."""
    if scores.size == 0 or k <= 0:
        return float("nan"), float("nan")
    k_eff = min(k, scores.size)
    order = np.argsort(-scores)[:k_eff]
    t = trials[order]
    frac_ge1 = float(np.mean(t >= 1))
    frac_z = float(np.mean(t == 0))
    return frac_ge1, frac_z


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Spearman ρ and top-K inversion stats for score vs trial-count pairs."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--fig3-published",
        action="store_true",
        help="Use the six clinically validated points plotted in Figure 3 / RESULTS_EVIDENCE.md.",
    )
    src.add_argument(
        "--scores-csv",
        type=Path,
        metavar="PATH",
        help='CSV with header; use --score-col and --trial-col (e.g. "ensemble_score,trial_count").',
    )
    src.add_argument(
        "--scores-json",
        type=Path,
        metavar="PATH",
        help='JSON list of {"score": float, "trial_count": int}',
    )

    parser.add_argument("--score-col", default="score", help="CSV column name for model score.")
    parser.add_argument("--trial-col", default="trial_count", help="CSV column name for trial count.")

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Compute fraction of top-k predictions with trials (default 10; capped by n).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print ρ and two-column p-value on one line.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.fig3_published:
        pairs = FIG3_PUBLISHED_ROWS
        scores_list = [p[0] for p in pairs]
        trials_list = [p[1] for p in pairs]
    elif args.scores_csv is not None:
        scores_list, trials_list = _read_csv_pairs(args.scores_csv, args.score_col, args.trial_col)
    else:
        assert args.scores_json is not None
        scores_list, trials_list = _read_json_array(args.scores_json)

    s, t = _load_scores_trials_np(scores_list, trials_list)
    rho, pval = spearmanr(s, t)

    k = args.top_k
    frac_ge1, frac_z = top_k_stats(s, t, k)

    if args.quiet:
        print(f"{rho:.6f} {pval:.6g}")
        return 0

    print(f"n_pairs={s.size}")
    print(f"Spearman rho={rho:.4f}, p-value={pval:.4g}")
    print(f"top_{min(k, s.size)}: fraction_with_trials_ge1={frac_ge1:.3f}, fraction_zero_trials={frac_z:.3f}")
    if args.fig3_published:
        print(
            "(Curated Figure 3 panel; cite alongside full-test ρ once scores are joined "
            "with ClinicalTrials counts — roadmap §5.)",
            file=sys.stderr,
        )

    payload: dict[str, Any] = {
        "n_pairs": int(s.size),
        "spearman_rho": float(rho) if not np.isnan(rho) else None,
        "spearman_p": float(pval) if not np.isnan(pval) else None,
        f"top_{min(k, s.size)}_frac_trials_ge1": frac_ge1,
        f"top_{min(k, s.size)}_frac_zero_trials": frac_z,
    }
    print("json_metrics:", json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
