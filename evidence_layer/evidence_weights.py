from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "kg_rotate_score": 1.5,
    "kg_complex_score": 1.0,
    "graph_topology_score": 0.8,
    "qsvc_score": 1.2,
    "classical_ensemble_score": 1.5,
    "signature_reversal_score": 1.3,
    "cell_type_reversal_score": 1.0,
    "pathway_reversal_score": 0.9,
    "moa_alignment_score": 0.7,
    "clinical_evidence_score": 0.5,
}


def load_weights(config_path: str = "config/evidence_fusion_config.yaml") -> Dict[str, float]:
    """Load per-feature weights from config; fall back to defaults."""
    p = Path(config_path)
    if not p.exists():
        logger.warning(f"Evidence config not found at {p}, using default weights.")
        return dict(_DEFAULT_WEIGHTS)

    with open(p) as f:
        cfg = yaml.safe_load(f)

    weights = cfg.get("evidence_fusion", {}).get("weights", {})
    if not weights:
        return dict(_DEFAULT_WEIGHTS)

    # Fill any missing keys with defaults
    merged = dict(_DEFAULT_WEIGHTS)
    merged.update(weights)
    return merged
