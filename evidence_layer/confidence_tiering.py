from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLDS = {
    "tier_1_threshold": 0.90,
    "tier_2_threshold": 0.80,
    "tier_3_threshold": 0.70,
    "tier_4_threshold": 0.60,
}


def load_thresholds(config_path: str = "config/evidence_fusion_config.yaml") -> Dict[str, float]:
    """Load confidence tier thresholds from config."""
    p = Path(config_path)
    if not p.exists():
        return dict(_DEFAULT_THRESHOLDS)
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("confidence_tiers", _DEFAULT_THRESHOLDS)


def assign_tier(score: float, thresholds: Optional[Dict[str, float]] = None) -> int:
    """
    Assign confidence tier (1 = highest confidence) based on final score.

    Returns 1–4; score below tier_4_threshold returns 4.
    """
    t = thresholds or _DEFAULT_THRESHOLDS
    if score >= t.get("tier_1_threshold", 0.90):
        return 1
    if score >= t.get("tier_2_threshold", 0.80):
        return 2
    if score >= t.get("tier_3_threshold", 0.70):
        return 3
    return 4
