# middleware/ranked_mechanisms.py

"""
Mechanism-aware intervention ranking.
Extracts ranking logic for use by orchestrator and API.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from kg_layer.hypothesis_graph import build_mechanism_subgraph

logger = logging.getLogger(__name__)


def rank_mechanism_candidates(
    hypothesis_id: str,
    disease_id: str,
    top_k: int,
    df_edges: pd.DataFrame,
    predictor: Callable[[str, str, str], Dict[str, Any]],
    id_to_name: Optional[Dict[str, str]] = None,
    compound_ids: Optional[List[str]] = None,
    max_compounds: int = 200,
    method: str = "auto",
) -> Dict[str, Any]:
    """
    Rank compound candidates for a disease using mechanism-informed subgraph.

    Args:
        hypothesis_id: H-001, H-002, or H-003
        disease_id: Resolved disease entity ID
        top_k: Number of top candidates to return
        df_edges: Full Hetionet edges
        predictor: Callable(drug_id, disease_id, method) -> dict with link_probability, status, model_used
        id_to_name: Optional mapping from entity ID to display name
        compound_ids: Override compound list (e.g., from embedder.entity_to_id)
        max_compounds: Max compounds to score
        method: Prediction method passed to predictor

    Returns:
        Dict with ranked_candidates, model_used, hypothesis_id, status
    """
    subgraph = build_mechanism_subgraph(df_edges, hypothesis_id)
    compounds = list(subgraph.get_compound_ids())
    if not compounds and compound_ids:
        compounds = compound_ids

    scored: List[Dict[str, Any]] = []
    model_used = "classical"
    id_to_name = id_to_name or {}

    for comp_id in compounds[:max_compounds]:
        try:
            res = predictor(comp_id, disease_id, method)
            if res.get("status") == "success":
                model_used = res.get("model_used", model_used)
                comp_name = id_to_name.get(
                    comp_id,
                    comp_id.split("::")[-1] if "::" in comp_id else comp_id,
                )
                scored.append({
                    "compound_id": comp_id,
                    "compound_name": comp_name,
                    "score": res["link_probability"],
                    "mechanism_summary": f"Shared mechanism subgraph ({hypothesis_id})",
                })
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    ranked = scored[:top_k]

    return {
        "hypothesis_id": hypothesis_id,
        "ranked_candidates": ranked,
        "model_used": model_used,
        "status": "success",
    }
