from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.evidence_weights import load_weights
from evidence_layer.confidence_tiering import assign_tier, load_thresholds

logger = logging.getLogger(__name__)

_OMICS_FIELDS = frozenset({
    "signature_reversal_score",
    "cell_type_reversal_score",
    "pathway_reversal_score",
})
_MATCHED_STATUSES = frozenset({"matched_human", "matched_non_human"})


def _load_matched_omics_policy(config_path: str) -> dict:
    p = Path(config_path)
    if not p.exists():
        return {"zero_unmatched_reversal": True, "signature_reversal_multiplier": 1.0}
    with p.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    return (cfg.get("evidence_fusion") or {}).get(
        "matched_omics",
        {"zero_unmatched_reversal": True, "signature_reversal_multiplier": 1.0},
    )


def fuse_evidence(
    candidates: List[EvidenceFeatures],
    config_path: str = "config/evidence_fusion_config.yaml",
    mode: str = "kg+omics",
    omics_match_status: Optional[List[str]] = None,
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
    matched_policy = _load_matched_omics_policy(config_path)
    zero_unmatched = bool(matched_policy.get("zero_unmatched_reversal", True))
    reversal_multiplier = float(matched_policy.get("signature_reversal_multiplier", 1.0))

    for idx, ef in enumerate(candidates):
        match_status = (
            omics_match_status[idx]
            if omics_match_status is not None and idx < len(omics_match_status)
            else None
        )
        is_matched = match_status in _MATCHED_STATUSES

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

        if mode != "kg-only" and omics_match_status is not None:
            if zero_unmatched and not is_matched:
                for field in _OMICS_FIELDS:
                    fv[field] = 0.0

        effective_weights = dict(weights)
        if (
            mode != "kg-only"
            and is_matched
            and reversal_multiplier != 1.0
        ):
            effective_weights["signature_reversal_score"] = (
                weights.get("signature_reversal_score", 1.3) * reversal_multiplier
            )

        total_w = sum(effective_weights.get(k, 1.0) for k in fv if effective_weights.get(k, 0.0) != 0.0)
        if total_w == 0:
            ef.final_score = 0.0
        else:
            ef.final_score = float(
                sum(
                    v * effective_weights.get(k, 1.0)
                    for k, v in fv.items()
                    if effective_weights.get(k, 0.0) != 0.0
                )
                / total_w
            )

        ef.confidence_tier = assign_tier(ef.final_score, thresholds)

    # Sort by final_score descending
    candidates.sort(key=lambda x: x.final_score, reverse=True)
    logger.info(
        f"Evidence fusion complete ({mode}): {len(candidates)} candidates, "
        f"top score={candidates[0].final_score:.4f}" if candidates else "no candidates"
    )
    return candidates
