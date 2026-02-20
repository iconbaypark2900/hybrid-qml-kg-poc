# benchmarking/directional_metrics.py

"""
Falsifiable, split-aware directional validation metrics.
DC = (# edges with correct predicted direction) / (total tested edges)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

DC_THRESHOLD = 0.65
REPLICATION_SPLITS_MIN = 2


def _load_thresholds(config_path: str = "config/hypotheses/metrics_thresholds.yaml") -> Dict:
    path = Path(config_path)
    if not path.exists():
        return {"directional_consistency_min": DC_THRESHOLD, "replication_splits_min": REPLICATION_SPLITS_MIN}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "directional_consistency_min": cfg.get("directional_consistency_min", DC_THRESHOLD),
        "replication_splits_min": cfg.get("replication_splits_min", REPLICATION_SPLITS_MIN),
    }


def compute_directional_consistency(
    predicted_directions: Union[np.ndarray, List[int], List[float]],
    ground_truth_directions: Union[np.ndarray, List[int], List[float]],
) -> float:
    """
    DC = (# edges with correct predicted direction) / (total tested edges).

    Directions: positive = up/positive effect, negative = down/negative effect.
    Sign must match (both positive or both negative) for correct.

    Args:
        predicted_directions: Predicted effect directions (sign: +1, -1, or continuous)
        ground_truth_directions: True effect directions

    Returns:
        DC in [0, 1], or 0.0 if no edges
    """
    pred = np.asarray(predicted_directions).flatten()
    gt = np.asarray(ground_truth_directions).flatten()
    if len(pred) != len(gt) or len(pred) == 0:
        return 0.0
    pred_sign = np.sign(pred)
    gt_sign = np.sign(gt)
    # Treat 0 as neutral; match only when both non-zero and same sign
    valid = (pred_sign != 0) & (gt_sign != 0)
    if not np.any(valid):
        return 0.0
    correct = np.sum((pred_sign == gt_sign) & valid)
    total = np.sum(valid)
    return float(correct / total) if total > 0 else 0.0


def run_permutation_baseline(
    predicted_directions: Union[np.ndarray, List],
    ground_truth_directions: Union[np.ndarray, List],
    n_permutations: int = 100,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Shuffle node/edge labels and recompute DC to get null distribution.

    Shuffles ground_truth_directions (breaks alignment with predictions).

    Args:
        predicted_directions: Predicted directions (unchanged)
        ground_truth_directions: True directions (shuffled per permutation)
        n_permutations: Number of permutations
        random_state: Random seed

    Returns:
        Null distribution of DC values, shape (n_permutations,)
    """
    rng = np.random.default_rng(random_state)
    pred = np.asarray(predicted_directions).flatten()
    gt = np.asarray(ground_truth_directions).flatten()
    n = len(pred)
    null_dc = np.zeros(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        shuffled_gt = rng.permutation(gt)
        null_dc[i] = compute_directional_consistency(pred, shuffled_gt)
    return null_dc


def compare_to_null(
    observed_dc: float,
    null_distribution: np.ndarray,
) -> float:
    """
    Compute p-value: proportion of null distribution >= observed_dc (one-tailed).

    Args:
        observed_dc: Observed directional consistency
        null_distribution: DC values from permutation baseline

    Returns:
        p-value in [0, 1]
    """
    if len(null_distribution) == 0:
        return 1.0
    return float(np.mean(null_distribution >= observed_dc))
