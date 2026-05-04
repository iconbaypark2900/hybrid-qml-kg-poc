"""Paired-bootstrap confidence intervals for PR-AUC differences across baselines.

Implements the H1 decision rule from
``preregistration/osf_preregistration_v1.md`` §8.1: paired-bootstrap CI on
per-fold PR-AUC differences (model A − model B), with the conjunction-across-
baselines rule deciding overall H1 support.

Bootstrap parameters (seed, n_resamples, confidence) default to the locked
values in ``utils.preregistered_constants``. Override only with a deviation
log entry.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from utils.preregistered_constants import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
)
from utils.reproducibility import get_rng


def paired_bootstrap_pr_auc_difference(
    scores_a: np.ndarray,
    labels_a: np.ndarray,
    scores_b: np.ndarray,
    labels_b: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Paired-bootstrap CI on PR-AUC(model_a) − PR-AUC(model_b).

    Both models score the same evaluation instances, so labels_a and labels_b
    must be identical (the function checks this). Each bootstrap resample
    samples instances with replacement, computes PR-AUC for each model on the
    resampled set, and records the difference.

    Args:
        scores_a, scores_b: (n,) per-instance positive-class probabilities.
        labels_a, labels_b: (n,) ground-truth labels (must be identical).
        n_resamples: number of bootstrap resamples.
        confidence: confidence level (e.g. 0.95).
        seed: deterministic seed for the bootstrap RNG.

    Returns:
        (point_estimate, ci_low, ci_high) where point_estimate is the
        observed PR-AUC(a) − PR-AUC(b) on the original sample and
        (ci_low, ci_high) is the percentile bootstrap CI.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    labels_a = np.asarray(labels_a, dtype=int)
    labels_b = np.asarray(labels_b, dtype=int)

    if scores_a.shape != scores_b.shape:
        raise ValueError(
            f"scores_a and scores_b must have same shape; got "
            f"{scores_a.shape} vs {scores_b.shape}"
        )
    if not np.array_equal(labels_a, labels_b):
        raise ValueError(
            "paired bootstrap requires identical labels for both models "
            "(scoring the same evaluation instances)"
        )
    n = scores_a.shape[0]
    if n < 2:
        raise ValueError(f"need at least 2 instances for bootstrap; got {n}")

    point_estimate = float(
        average_precision_score(labels_a, scores_a)
        - average_precision_score(labels_b, scores_b)
    )

    rng = get_rng(seed)
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        ya = labels_a[idx]
        # If the resample is degenerate (all one class), PR-AUC is undefined;
        # skip by drawing again. Bounded retries to avoid infinite loop.
        retries = 0
        while ya.sum() == 0 or ya.sum() == n:
            idx = rng.integers(0, n, size=n)
            ya = labels_a[idx]
            retries += 1
            if retries > 50:
                raise RuntimeError(
                    "bootstrap could not draw a non-degenerate resample; "
                    "labels are too imbalanced for n_resamples to converge"
                )
        diffs[i] = (
            average_precision_score(ya, scores_a[idx])
            - average_precision_score(ya, scores_b[idx])
        )

    alpha = 1.0 - confidence
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point_estimate, float(lo), float(hi)


def conjunction_across_baselines(
    qsvc_scores: np.ndarray,
    labels: np.ndarray,
    baseline_scores: Mapping[str, np.ndarray],
    *,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Conjunction-across-baselines decision rule for H1.

    Computes paired-bootstrap CI on PR-AUC(QSVC) − PR-AUC(baseline) for each
    baseline. H1 is supported iff every CI excludes zero in the favorable
    direction (lower bound > 0).

    Args:
        qsvc_scores: (n,) QSVC positive-class probabilities.
        labels: (n,) ground-truth labels.
        baseline_scores: name -> (n,) per-baseline positive-class probabilities.
        n_resamples, confidence, seed: forwarded to
            ``paired_bootstrap_pr_auc_difference``.

    Returns:
        {
            "per_baseline": {name: {point, ci_low, ci_high, supported}},
            "h1_supported": bool,
            "n_baselines_supporting": int,
            "n_baselines_total": int,
        }
    """
    per_baseline: dict[str, dict] = {}
    for name, b_scores in baseline_scores.items():
        point, lo, hi = paired_bootstrap_pr_auc_difference(
            qsvc_scores,
            labels,
            np.asarray(b_scores, dtype=float),
            labels,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        per_baseline[name] = {
            "point": point,
            "ci_low": lo,
            "ci_high": hi,
            "supported": lo > 0.0,
        }

    n_supporting = sum(1 for v in per_baseline.values() if v["supported"])
    n_total = len(per_baseline)
    return {
        "per_baseline": per_baseline,
        "h1_supported": n_supporting == n_total and n_total > 0,
        "n_baselines_supporting": n_supporting,
        "n_baselines_total": n_total,
    }


__all__ = [
    "paired_bootstrap_pr_auc_difference",
    "conjunction_across_baselines",
]
