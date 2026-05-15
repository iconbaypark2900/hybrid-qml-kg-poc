from __future__ import annotations

import logging
from typing import Dict, List, Optional

from perturbation_layer.reversal_score import compute_reversal_score

logger = logging.getLogger(__name__)


def compute_pathway_reversal_scores(
    pathway_gene_sets: Dict[str, Dict[str, List[str]]],
    disease_up: List[str],
    disease_down: List[str],
    drug_up: List[str],
    drug_down: List[str],
) -> Dict[str, float]:
    """
    Compute reversal scores at pathway level.

    Restricts disease and drug gene lists to the genes in each pathway
    before computing the reversal score.

    Args:
        pathway_gene_sets: {pathway_name: {"genes": [...]}}
        disease_up / disease_down: disease DE genes
        drug_up / drug_down: drug perturbation genes

    Returns:
        {pathway_name: reversal_score}
    """
    scores: Dict[str, float] = {}
    d_up = set(disease_up)
    d_dn = set(disease_down)
    c_up = set(drug_up)
    c_dn = set(drug_down)

    for pathway, info in pathway_gene_sets.items():
        pw_genes = set(info.get("genes", []))
        if not pw_genes:
            continue

        pw_d_up = list(d_up & pw_genes)
        pw_d_dn = list(d_dn & pw_genes)
        pw_c_up = list(c_up & pw_genes)
        pw_c_dn = list(c_dn & pw_genes)

        if not (pw_d_up or pw_d_dn):
            continue  # disease has no DE genes in this pathway

        scores[pathway] = compute_reversal_score(pw_d_up, pw_d_dn, pw_c_up, pw_c_dn)

    logger.debug(f"Pathway reversal computed for {len(scores)} pathways")
    return scores
