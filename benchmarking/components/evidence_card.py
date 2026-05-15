from __future__ import annotations

"""Streamlit component: evidence card for a single drug candidate."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_TIER_COLORS = {1: "🟢", 2: "🔵", 3: "🟡", 4: "🟠"}
_TIER_LABELS = {1: "Tier 1 — Very High", 2: "Tier 2 — High", 3: "Tier 3 — Moderate", 4: "Tier 4 — Exploratory"}


def render_evidence_card(candidate: Dict) -> None:
    """
    Render a Streamlit evidence card for a single drug candidate.

    Args:
        candidate: dict with keys matching EvidenceFeatures.to_dict()
    """
    try:
        import streamlit as st
    except ImportError:
        logger.warning("streamlit not installed; cannot render evidence card.")
        return

    tier = int(candidate.get("confidence_tier", 4))
    score = float(candidate.get("final_score", 0.0))
    compound = candidate.get("compound", "Unknown")
    disease = candidate.get("disease", "Unknown")

    with st.container():
        st.markdown(f"### {_TIER_COLORS.get(tier, '⚪')} {compound}")
        st.markdown(f"**Disease:** {disease}  |  **Final score:** `{score:.4f}`  |  **{_TIER_LABELS.get(tier, 'Tier 4')}`")

        cols = st.columns(5)
        score_fields = [
            ("KG (RotatE)", "kg_rotate_score"),
            ("QSVC", "qsvc_score"),
            ("Ensemble", "classical_ensemble_score"),
            ("Reversal", "signature_reversal_score"),
            ("Clinical", "clinical_evidence_score"),
        ]
        for col, (label, key) in zip(cols, score_fields):
            val = float(candidate.get(key, 0.0))
            col.metric(label=label, value=f"{val:.3f}")

        if candidate.get("explanation"):
            with st.expander("Full evidence breakdown"):
                st.code(candidate["explanation"])
