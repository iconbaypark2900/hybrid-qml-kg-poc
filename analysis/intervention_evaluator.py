# analysis/intervention_evaluator.py

"""
Intervention evaluator with weighted composite score.
Score = w1 * mechanism_score + w2 * directional_consistency - w3 * variance_across_splits
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "w1_mechanism_score": 0.5,
    "w2_directional_consistency": 0.4,
    "w3_variance_penalty": 0.1,
}


def load_ranking_weights(config_path: str = "config/hypotheses/ranking_weights.yaml") -> Dict[str, float]:
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_WEIGHTS
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return {
        "w1_mechanism_score": cfg.get("w1_mechanism_score", DEFAULT_WEIGHTS["w1_mechanism_score"]),
        "w2_directional_consistency": cfg.get("w2_directional_consistency", DEFAULT_WEIGHTS["w2_directional_consistency"]),
        "w3_variance_penalty": cfg.get("w3_variance_penalty", DEFAULT_WEIGHTS["w3_variance_penalty"]),
    }


def evaluate_compound(
    mechanism_score: float,
    directional_consistency: float,
    variance_across_splits: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute composite score for an intervention candidate.

    Score = w1 * mechanism_score + w2 * directional_consistency - w3 * variance_across_splits

    Args:
        mechanism_score: BMP-restoration / lysosomal stabilization proxy (0-1)
        directional_consistency: Stability of effect direction across splits (0-1)
        variance_across_splits: Variance to penalize (0-1 or higher)
        weights: Optional override for w1, w2, w3

    Returns:
        Composite score (higher is better)
    """
    w = weights or load_ranking_weights()
    return (
        w["w1_mechanism_score"] * mechanism_score
        + w["w2_directional_consistency"] * directional_consistency
        - w["w3_variance_penalty"] * variance_across_splits
    )
