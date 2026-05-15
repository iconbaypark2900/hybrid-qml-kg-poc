from __future__ import annotations

"""Streamlit component: disease gene signature view."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


def render_signature_view(signature: Dict) -> None:
    """
    Render disease signature: top up/down genes as a table and mini-heatmap.
    """
    try:
        import streamlit as st
        import pandas as pd
    except ImportError:
        logger.warning("streamlit/pandas not available.")
        return

    st.subheader(f"Disease Signature — {signature.get('cell_type', 'all cells')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Up-regulated Genes**")
        up = signature.get("up_genes", [])[:20]
        st.dataframe(pd.DataFrame({"gene": up}), use_container_width=True)

    with col2:
        st.markdown("**Top Down-regulated Genes**")
        dn = signature.get("down_genes", [])[:20]
        st.dataframe(pd.DataFrame({"gene": dn}), use_container_width=True)

    ranked = signature.get("ranked_genes", [])[:30]
    if ranked:
        df = pd.DataFrame(ranked)
        st.markdown("**Top Ranked Genes (by score)**")
        st.dataframe(df, use_container_width=True)

    pathways = signature.get("pathways", [])
    if pathways:
        st.markdown("**Enriched Pathways**")
        st.dataframe(pd.DataFrame(pathways[:10]), use_container_width=True)
