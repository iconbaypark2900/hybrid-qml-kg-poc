from __future__ import annotations

"""Streamlit component: clinical validation status view."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

_STATUS_ICONS = {
    "known": "✅",
    "supported_novel_candidate": "🔵",
    "literature_supported": "📄",
    "exploratory": "🔬",
}


def render_clinical_validation_view(candidates: List[Dict]) -> None:
    """
    Render a table of candidates with clinical validation status badges.
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        logger.warning("streamlit/pandas not available.")
        return

    st.subheader("Clinical Validation")

    rows = []
    for c in candidates:
        status = c.get("validation_status", "exploratory")
        rows.append({
            "Compound": c.get("compound", ""),
            "Disease": c.get("disease", ""),
            "Score": f"{c.get('final_score', 0.0):.3f}",
            "Status": f"{_STATUS_ICONS.get(status, '?')} {status}",
            "Known Indication": "Yes" if c.get("known_indication") else "No",
            "Trial Phase": c.get("trial_phase") or "—",
            "Literature": c.get("literature_support_count", 0),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No validated candidates to display.")
