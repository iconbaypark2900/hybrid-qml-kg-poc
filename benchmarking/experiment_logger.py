# benchmarking/experiment_logger.py

"""
Structured experiment logging for reproducibility.
Logs: model config, embedding params, feature flags, random seeds, hardware backend, metrics per split.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

EXPERIMENT_LOGS_DIR = "results/experiment_logs"


def log_experiment_run(
    run_id: str,
    config: Dict[str, Any],
    metrics_by_split: Optional[Dict[str, Dict[str, float]]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    embedding_params: Optional[Dict[str, Any]] = None,
    feature_flags: Optional[Dict[str, bool]] = None,
    random_seed: Optional[int] = None,
    hardware_backend: Optional[str] = None,
    output_dir: str = EXPERIMENT_LOGS_DIR,
) -> str:
    """
    Write structured experiment log to results/experiment_logs/run_{run_id}.json.

    Args:
        run_id: Unique run identifier
        config: Full run configuration
        metrics_by_split: Metrics per split (e.g., {"split_0": {"pr_auc": 0.8}})
        model_config: Model hyperparameters
        embedding_params: Embedding configuration
        feature_flags: e.g., {"use_directional": True, "use_lysosomal": False}
        random_seed: Seed used
        hardware_backend: e.g., "cpu", "cuda"
        output_dir: Directory for log files

    Returns:
        Path to written log file
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"run_{run_id}.json"

    log_entry = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "model_config": model_config or {},
        "embedding_params": embedding_params or {},
        "feature_flags": feature_flags or {},
        "random_seed": random_seed,
        "hardware_backend": hardware_backend or "unknown",
        "metrics_by_split": metrics_by_split or {},
    }

    with open(filepath, "w") as f:
        json.dump(log_entry, f, indent=2)
    logger.info(f"Logged experiment run to {filepath}")
    return str(filepath)
