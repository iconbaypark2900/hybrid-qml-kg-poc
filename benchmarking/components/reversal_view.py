from __future__ import annotations

"""Streamlit component: drug reversal score breakdown."""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def render_reversal_view(
    compound: str,
    overall_score: float,
    cell_type_scores: Optional[Dict[str, float]] = None,
    pathway_scores: Optional[Dict[str, float]] = None,
) -> None:
    """
    Render cell-type and pathway reversal scores as bar charts.
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        logger.warning("streamlit/pandas not available.")
        return

    st.subheader(f"Reversal Analysis — {compound}")
    st.metric("Overall Reversal Score", f"{overall_score:.4f}")

    col1, col2 = st.columns(2)

    with col1:
        if cell_type_scores:
            st.markdown("**Cell-Type Reversal Scores**")
            df = pd.DataFrame(
                list(cell_type_scores.items()), columns=["Cell Type", "Score"]
            ).sort_values("Score", ascending=False)
            st.bar_chart(df.set_index("Cell Type"))

    with col2:
        if pathway_scores:
            st.markdown("**Pathway Reversal Scores**")
            df = pd.DataFrame(
                list(pathway_scores.items()), columns=["Pathway", "Score"]
            ).sort_values("Score", ascending=False).head(15)
            st.bar_chart(df.set_index("Pathway"))
