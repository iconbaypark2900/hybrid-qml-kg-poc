from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def build_disease_signature(
    de_df: pd.DataFrame,
    disease_id: str,
    tissue: str = "",
    cell_type: str = "all_cells",
    lfc_threshold: float = 0.5,
    padj_threshold: float = 0.05,
    top_n: int = 250,
) -> Dict:
    """
    Build a disease gene signature dict from a DE results DataFrame.

    Args:
        de_df: output of differential_expression.run_de() (or subset)
        disease_id: Hetionet disease node ID or label
        tissue: tissue descriptor
        cell_type: cell type label (use "all_cells" if not stratified)
        lfc_threshold: minimum |log fold change| to include a gene
        padj_threshold: maximum adjusted p-value
        top_n: maximum genes in up/down lists

    Returns:
        Dict matching the gap-doc JSON schema.
    """
    if de_df.empty:
        return {
            "disease": disease_id, "tissue": tissue, "cell_type": cell_type,
            "up_genes": [], "down_genes": [], "ranked_genes": [], "pathways": [],
        }

    # Filter by significance
    sig = de_df[de_df["pvals_adj"] <= padj_threshold].copy()
    sig = sig[sig["logfoldchanges"].abs() >= lfc_threshold]

    sig = sig.nlargest(top_n * 2, "scores")

    up = sig[sig["logfoldchanges"] > 0].nlargest(top_n, "logfoldchanges")
    dn = sig[sig["logfoldchanges"] < 0].nsmallest(top_n, "logfoldchanges")

    ranked = []
    for _, row in sig.iterrows():
        ranked.append({
            "gene": str(row["names"]),
            "score": round(float(row["scores"]), 4),
            "logfc": round(float(row["logfoldchanges"]), 4),
            "p_adj": round(float(row["pvals_adj"]), 6),
        })

    return {
        "disease": disease_id,
        "tissue": tissue,
        "cell_type": cell_type,
        "up_genes": up["names"].tolist(),
        "down_genes": dn["names"].tolist(),
        "ranked_genes": ranked,
        "pathways": [],  # populated by pathway_enrichment.py
    }


def build_cell_type_signatures(
    de_df: pd.DataFrame,
    disease_id: str,
    tissue: str = "",
    **kwargs,
) -> Dict[str, Dict]:
    """
    Build disease signatures for each cell type present in de_df.

    Returns {cell_type: signature_dict}.
    """
    if "cell_type" not in de_df.columns:
        return {"all_cells": build_disease_signature(de_df, disease_id, tissue, **kwargs)}

    sigs: Dict[str, Dict] = {}
    for ct in de_df["cell_type"].unique():
        subset = de_df[de_df["cell_type"] == ct]
        sigs[ct] = build_disease_signature(subset, disease_id, tissue, cell_type=ct, **kwargs)
    return sigs
