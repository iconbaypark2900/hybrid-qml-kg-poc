# kg_layer/perturbation_encoder.py

"""
Encode perturbation direction as signed feature blocks for mechanism-aware modeling.
Supports: Knockdown, Knockout, Rescue, Overexpression.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Perturbation types and signed encoding (aligned with embedding dimensions)
PERTURBATION_ENCODING = {
    "knockdown": -1.0,   # Down-regulation
    "knockout": -1.0,    # Null/absence
    "rescue": 0.0,       # Restoration
    "overexpression": 1.0,  # Up-regulation
    "down": -1.0,
    "null": -1.0,
    "up": 1.0,
}


def encode_perturbation(
    perturbation_type: str,
    dim: int = 4,
) -> np.ndarray:
    """
    Encode perturbation type as a signed vector aligned with embedding dimensions.

    Args:
        perturbation_type: One of knockdown, knockout, rescue, overexpression (or down, null, up)
        dim: Output dimension (default 4, matches number of perturbation types)

    Returns:
        Signed vector of shape (dim,) with values scaled for feature concatenation
    """
    key = perturbation_type.lower().strip()
    val = PERTURBATION_ENCODING.get(key, 0.0)
    return np.full(dim, val, dtype=np.float32)


def load_perturbation_assays(data_dir: str = "data/perturbation_assays") -> Optional[pd.DataFrame]:
    """
    Load perturbation assay data from CSV if present.

    Expected schema: entity_id, perturbation, direction, magnitude
    Returns None if directory or files do not exist.
    """
    path = Path(data_dir)
    if not path.exists():
        return None
    files = list(path.glob("*.csv"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            required = {"entity_id", "perturbation"}
            if required.issubset(df.columns):
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def build_perturbation_features(
    entity_ids: List[str],
    assay_df: Optional[pd.DataFrame] = None,
    dim: int = 4,
) -> np.ndarray:
    """
    Build perturbation feature block for a list of entity IDs.

    When assay_df is None, returns zeros (no perturbation data).
    When assay_df exists, maps entity_id -> perturbation -> signed encoding.

    Args:
        entity_ids: List of entity IDs (e.g., Compound::..., Gene::...)
        assay_df: Optional DataFrame with entity_id, perturbation columns
        dim: Feature dimension per entity

    Returns:
        Feature matrix of shape (len(entity_ids), dim)
    """
    out = np.zeros((len(entity_ids), dim), dtype=np.float32)
    if assay_df is None or len(assay_df) == 0:
        return out

    entity_to_pert = {}
    for _, row in assay_df.iterrows():
        eid = str(row["entity_id"])
        pert = str(row.get("perturbation", row.get("direction", "null")))
        entity_to_pert[eid] = pert

    for i, eid in enumerate(entity_ids):
        pert = entity_to_pert.get(str(eid), "null")
        out[i] = encode_perturbation(pert, dim)
    return out
