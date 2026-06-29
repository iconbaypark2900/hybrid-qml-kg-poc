from __future__ import annotations

import logging
from typing import Dict

from evidence_layer.evidence_schema import EvidenceFeatures

logger = logging.getLogger(__name__)

_TIER_LABELS = {1: "Tier 1 (very high)", 2: "Tier 2 (high)", 3: "Tier 3 (moderate)", 4: "Tier 4 (exploratory)"}


def build_explanation(ef: EvidenceFeatures) -> str:
    """
    Generate a human-readable explanation string for a candidate.

    Returns a multi-line string summarising which evidence streams are
    strong, moderate, or weak for this compound-disease pair.
    """
    def _strength(score: float) -> str:
        if score >= 0.7:
            return "strong"
        if score >= 0.4:
            return "moderate"
        return "weak"

    lines = [
        f"Candidate: {ef.compound}",
        f"Disease: {ef.disease}",
        f"Final score: {ef.final_score:.4f}",
        f"Confidence: {_TIER_LABELS.get(ef.confidence_tier, 'Tier 4')}",
        "",
        "Evidence summary:",
        f"  - KG embedding score (RotatE): {_strength(ef.kg_rotate_score)} ({ef.kg_rotate_score:.3f})",
        f"  - QSVC score: {_strength(ef.qsvc_score)} ({ef.qsvc_score:.3f})",
        f"  - Classical ensemble: {_strength(ef.classical_ensemble_score)} ({ef.classical_ensemble_score:.3f})",
        f"  - Disease signature reversal: {_strength(ef.signature_reversal_score)} ({ef.signature_reversal_score:.3f})",
        f"  - Cell-type reversal: not computed (weight=0 in fusion config)",
        f"  - Pathway reversal: not computed (GSEA deferred; weight=0)",
        f"  - Clinical evidence: {_strength(ef.clinical_evidence_score)} ({ef.clinical_evidence_score:.3f})",
    ]
    return "\n".join(lines)


def attach_explanations(candidates: list) -> list:
    """Populate ef.explanation for every candidate in-place."""
    for ef in candidates:
        ef.explanation = build_explanation(ef)
    return candidates
