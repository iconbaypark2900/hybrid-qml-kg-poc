from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_reversal_score(
    disease_up: List[str],
    disease_down: List[str],
    drug_up: List[str],
    drug_down: List[str],
) -> float:
    """
    Gene-level reversal score (Spearman-inspired rank overlap).

    A drug reverses a disease signature when it:
      - Upregulates genes the disease downregulates (concordant up-in-drug / down-in-disease)
      - Downregulates genes the disease upregulates (concordant down-in-drug / up-in-disease)

    Score ∈ [-1, 1]:
      +1 = perfect reversal
       0 = no overlap
      -1 = perfect mirroring (same direction = disease-aggravating)
    """
    d_up = set(disease_up)
    d_dn = set(disease_down)
    c_up = set(drug_up)
    c_dn = set(drug_down)

    all_genes = d_up | d_dn | c_up | c_dn
    if not all_genes:
        return 0.0

    # Reversal: drug opposes disease direction
    reversal_hits = len((d_up & c_dn) | (d_dn & c_up))
    # Aggravation: drug mirrors disease direction
    aggravation_hits = len((d_up & c_up) | (d_dn & c_dn))

    denominator = len(d_up | d_dn) + len(c_up | c_dn)
    if denominator == 0:
        return 0.0

    score = (reversal_hits - aggravation_hits) / (denominator / 2)
    return float(np.clip(score, -1.0, 1.0))


def compute_reversal_scores_batch(
    disease_signature: Dict,
    drug_signatures: Dict[str, object],
) -> Dict[str, float]:
    """
    Compute reversal scores for all compounds in drug_signatures against
    a single disease signature.

    Args:
        disease_signature: dict with keys "up_genes", "down_genes"
        drug_signatures: {compound_name: DrugSignature}

    Returns:
        {compound_name: reversal_score}
    """
    d_up = disease_signature.get("up_genes", [])
    d_dn = disease_signature.get("down_genes", [])

    scores: Dict[str, float] = {}
    for compound, sig in drug_signatures.items():
        c_up = getattr(sig, "up_genes", []) if hasattr(sig, "up_genes") else sig.get("up_genes", [])
        c_dn = getattr(sig, "down_genes", []) if hasattr(sig, "down_genes") else sig.get("down_genes", [])
        scores[compound] = compute_reversal_score(d_up, d_dn, c_up, c_dn)

    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    logger.info(f"Top-5 reversal scores: {top5}")
    return scores
