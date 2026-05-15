from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False
    logger.warning("gseapy not installed; pathway enrichment disabled.")


def run_pathway_enrichment(
    gene_list: List[str],
    gene_sets: str = "KEGG_2021_Human",
    organism: str = "human",
    top_n: int = 20,
) -> List[Dict]:
    """
    Over-representation analysis (ORA) against a gene set library.

    Args:
        gene_list: list of gene symbols (e.g. disease up-genes)
        gene_sets: gseapy library name
        organism: "human" | "mouse"
        top_n: number of top pathways to return

    Returns:
        List of {name, direction, score, p_adj, genes} dicts.
        Empty list if gseapy is unavailable or enrichment fails.
    """
    if not GSEAPY_AVAILABLE:
        logger.warning("gseapy not available; skipping pathway enrichment.")
        return []

    if not gene_list:
        return []

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            outdir=None,
        )
        df = enr.results.head(top_n)
        return [
            {
                "name": str(row["Term"]),
                "direction": "up",
                "score": round(float(-row.get("Adjusted P-value", 1.0)), 6),
                "p_adj": round(float(row.get("Adjusted P-value", 1.0)), 6),
                "genes": str(row.get("Genes", "")),
            }
            for _, row in df.iterrows()
        ]
    except Exception as e:
        logger.warning(f"Pathway enrichment failed ({e}); returning empty.")
        return []


def attach_pathways_to_signature(signature: Dict, top_n: int = 20) -> Dict:
    """
    Run ORA on up_genes and down_genes and attach results to the signature dict.

    Modifies signature["pathways"] in-place and returns the signature.
    """
    up_pathways = run_pathway_enrichment(signature.get("up_genes", []), top_n=top_n)
    dn_pathways = run_pathway_enrichment(signature.get("down_genes", []), top_n=top_n)

    for p in dn_pathways:
        p["direction"] = "down"

    signature["pathways"] = up_pathways + dn_pathways
    logger.info(
        f"Pathways attached: {len(up_pathways)} up, {len(dn_pathways)} down"
    )
    return signature
