from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.evidence_weights import load_weights
from evidence_layer.confidence_tiering import assign_tier, load_thresholds

logger = logging.getLogger(__name__)


def fuse_evidence(
    candidates: List[EvidenceFeatures],
    config_path: str = "config/evidence_fusion_config.yaml",
    mode: str = "kg+omics",
) -> List[EvidenceFeatures]:
    """
    Compute final_score and confidence_tier for each candidate.

    In 'kg-only' mode, omics features (signature_reversal_score,
    cell_type_reversal_score, pathway_reversal_score) are zeroed out
    before computing the weighted score. This preserves the baseline
    PR-AUC from the KG+QML-only pipeline.

    Args:
        candidates: list of EvidenceFeatures (scores filled in by caller)
        config_path: path to evidence_fusion_config.yaml
        mode: "kg-only" | "kg+omics"

    Returns:
        Same list with final_score, confidence_tier, explanation populated.
    """
    weights = load_weights(config_path)
    thresholds = load_thresholds(config_path)

    omics_fields = {"signature_reversal_score", "cell_type_reversal_score", "pathway_reversal_score"}

    for ef in candidates:
        fv: Dict[str, float] = {
            "kg_rotate_score": ef.kg_rotate_score,
            "kg_complex_score": ef.kg_complex_score,
            "graph_topology_score": ef.graph_topology_score,
            "qsvc_score": ef.qsvc_score,
            "classical_ensemble_score": ef.classical_ensemble_score,
            "signature_reversal_score": ef.signature_reversal_score if mode != "kg-only" else 0.0,
            "cell_type_reversal_score": ef.cell_type_reversal_score if mode != "kg-only" else 0.0,
            "pathway_reversal_score": ef.pathway_reversal_score if mode != "kg-only" else 0.0,
            "moa_alignment_score": ef.moa_alignment_score,
            "clinical_evidence_score": ef.clinical_evidence_score,
        }

        total_w = sum(weights.get(k, 1.0) for k in fv)
        if total_w == 0:
            ef.final_score = 0.0
        else:
            ef.final_score = float(
                sum(v * weights.get(k, 1.0) for k, v in fv.items()) / total_w
            )

        ef.confidence_tier = assign_tier(ef.final_score, thresholds)

    # Sort by final_score descending
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    logger.info(
        f"Evidence fusion complete ({mode}): {len(candidates)} candidates, "
        f"top score={candidates[0].final_score:.4f}" if candidates else "no candidates"
    )
    return candidates
