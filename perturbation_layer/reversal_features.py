from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_reversal_feature_vector(
    overall_reversal: float,
    cell_type_scores: Optional[Dict[str, float]] = None,
    pathway_scores: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Build a model-ready numpy feature vector from reversal scores.

    Features (in order):
      [0] overall_reversal_score
      [1] mean_cell_type_reversal   (0 if unavailable)
      [2] max_cell_type_reversal    (0 if unavailable)
      [3] mean_pathway_reversal     (0 if unavailable)
      [4] max_pathway_reversal      (0 if unavailable)
      [5] n_pathways_reversed       (count of pathways with score > 0)

    Returns shape (6,) float32 array.
    """
    feats: List[float] = [overall_reversal]

    if cell_type_scores and len(cell_type_scores) > 0:
        vals = list(cell_type_scores.values())
        feats.append(float(np.mean(vals)))
        feats.append(float(np.max(vals)))
    else:
        feats.extend([0.0, 0.0])

    if pathway_scores and len(pathway_scores) > 0:
        vals = list(pathway_scores.values())
        feats.append(float(np.mean(vals)))
        feats.append(float(np.max(vals)))
        feats.append(float(sum(1 for v in vals if v > 0)))
    else:
        feats.extend([0.0, 0.0, 0.0])

    return np.array(feats, dtype=np.float32)
