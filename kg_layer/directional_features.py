# kg_layer/directional_features.py

"""
Directional (perturbation) feature block for hypothesis testing.
Encodes up/down regulation evidence and imbalance (e.g., BMP precursor imbalance proxy).
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .evidence_weighting import (
    EvidenceConfigDirectional,
    add_directional_evidence,
    build_directional_gene_maps,
)


def build_directional_features(
    df_pairs: pd.DataFrame,
    comp2g_up: Dict[str, Set[str]],
    comp2g_down: Dict[str, Set[str]],
    dis2g_up: Dict[str, Set[str]],
    dis2g_down: Dict[str, Set[str]],
    source_col: str = "source",
    target_col: str = "target",
) -> np.ndarray:
    """
    Build directional feature block from up/down regulation evidence.

    Returns (n_samples, 4) array:
      - evidence_up: aligned regulation (compound up + disease up, or both down)
      - evidence_down: opposing regulation
      - evidence_balanced: up / (up + down + eps), 0.5 if both zero
      - imbalance: (up - down) / (up + down + eps), proxy for BMP precursor imbalance

    Args:
        df_pairs: DataFrame with compound-disease pairs
        comp2g_up, comp2g_down, dis2g_up, dis2g_down: Directional gene maps
        source_col, target_col: Column names for compound and disease

    Returns:
        Feature matrix of shape (len(df_pairs), 4)
    """
    df = add_directional_evidence(
        df_pairs,
        comp2g_up=comp2g_up,
        comp2g_down=comp2g_down,
        dis2g_up=dis2g_up,
        dis2g_down=dis2g_down,
        source_col=source_col,
        target_col=target_col,
        out_col_up="evidence_up",
        out_col_down="evidence_down",
        out_col_ratio="evidence_balanced",
    )
    up = np.array(df["evidence_up"], dtype=np.float32)
    down = np.array(df["evidence_down"], dtype=np.float32)
    balanced = np.array(df["evidence_balanced"], dtype=np.float32)
    eps = 1e-8
    total = up + down + eps
    imbalance = (up - down) / total
    return np.column_stack([up, down, balanced, imbalance]).astype(np.float32)


def get_directional_feature_names() -> List[str]:
    """Return names for the directional feature block."""
    return ["evidence_up", "evidence_down", "evidence_balanced", "imbalance"]
