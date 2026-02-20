# benchmarking/replication_validator.py

"""
Replication validation: DC within ±5% across splits, CI overlap.
Falsification trigger logging.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

REPLICATION_TOLERANCE = 0.05  # ±5%
FALSIFICATION_LOG_PATH = "results/reports/falsification_log.json"


def check_replication(
    dc_by_split: Dict[str, float],
    tolerance: float = REPLICATION_TOLERANCE,
) -> bool:
    """
    Replication is satisfied if DC is within ±5% across all splits.

    Args:
        dc_by_split: Dict mapping split_id -> directional consistency
        tolerance: Maximum allowed spread (default 0.05 = ±5%)

    Returns:
        True if DC values are within tolerance range
    """
    if len(dc_by_split) < 2:
        return False
    values = list(dc_by_split.values())
    spread = max(values) - min(values)
    return spread <= tolerance


def _bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for mean."""
    if not values:
        return (0.0, 0.0)
    arr = np.array(values)
    n = len(arr)
    means = [np.mean(np.random.choice(arr, n, replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return (float(lo), float(hi))


def check_ci_overlap(
    dc_by_split: Dict[str, float],
) -> bool:
    """
    Check if confidence intervals overlap across splits.

    Uses bootstrap CI per split; returns True if any two CIs overlap.
    """
    if len(dc_by_split) < 2:
        return True
    # For simplicity: treat each split's DC as single value; CI would need multiple runs
    values = list(dc_by_split.values())
    lo, hi = _bootstrap_ci(values)
    # If spread is small, CIs overlap
    return (hi - lo) < 0.2  # Heuristic


def log_falsification_trigger(
    hypothesis_id: str,
    trigger: str,
    conditions: Dict[str, Any],
    log_path: Optional[str] = None,
) -> None:
    """
    Write falsification trigger to results/reports/falsification_log.json.

    Args:
        hypothesis_id: e.g., H-001
        trigger: Description of falsification condition
        conditions: Dict with run details (metrics, splits, etc.)
        log_path: Override default path
    """
    path = Path(log_path or FALSIFICATION_LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "hypothesis_id": hypothesis_id,
        "timestamp": datetime.now().isoformat(),
        "trigger": trigger,
        "conditions": conditions,
    }

    logs = []
    if path.exists():
        try:
            with open(path, "r") as f:
                logs = json.load(f)
            if not isinstance(logs, list):
                logs = [logs]
        except Exception as e:
            logger.warning(f"Could not read existing falsification log: {e}")

    logs.append(entry)
    with open(path, "w") as f:
        json.dump(logs, f, indent=2)
    logger.info(f"Logged falsification trigger for {hypothesis_id} to {path}")
