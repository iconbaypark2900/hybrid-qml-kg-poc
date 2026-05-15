from __future__ import annotations

import logging
from typing import Dict, List, Optional

from perturbation_layer.reversal_score import compute_reversal_score

logger = logging.getLogger(__name__)


def compute_cell_type_reversal_scores(
    cell_type_signatures: Dict[str, Dict],
    drug_up: List[str],
    drug_down: List[str],
) -> Dict[str, float]:
    """
    Compute reversal scores per cell type.

    Args:
        cell_type_signatures: {cell_type: {"up_genes": [...], "down_genes": [...]}}
        drug_up: drug upregulated genes
        drug_down: drug downregulated genes

    Returns:
        {cell_type: reversal_score}
    """
    scores: Dict[str, float] = {}
    for cell_type, sig in cell_type_signatures.items():
        scores[cell_type] = compute_reversal_score(
            sig.get("up_genes", []),
            sig.get("down_genes", []),
            drug_up,
            drug_down,
        )
    logger.debug(f"Cell-type reversal scores: {scores}")
    return scores


def aggregate_cell_type_score(
    cell_type_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Weighted average of per-cell-type reversal scores.

    If weights is None, uses a simple mean.
    """
    if not cell_type_scores:
        return 0.0
    if weights is None:
        return float(sum(cell_type_scores.values()) / len(cell_type_scores))

    total_w = sum(weights.get(ct, 1.0) for ct in cell_type_scores)
    if total_w == 0:
        return 0.0
    return float(
        sum(v * weights.get(ct, 1.0) for ct, v in cell_type_scores.items()) / total_w
    )
