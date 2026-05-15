from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def find_marker_genes(adata, groupby: str = "leiden", n_genes: int = 25) -> None:
    """
    Wilcoxon rank-sum marker genes per cluster.

    Modifies adata in-place (adds adata.uns['rank_genes_groups']).
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy required. Run: pip install -r requirements-omics.txt")

    if groupby not in adata.obs.columns:
        logger.warning(f"Column '{groupby}' not in adata.obs; skipping marker genes.")
        return

    logger.info(f"Computing marker genes (groupby='{groupby}', n_genes={n_genes}) …")
    sc.tl.rank_genes_groups(adata, groupby=groupby, n_genes=n_genes, method="wilcoxon")
    logger.info("Marker genes computed.")


def export_marker_genes(
    adata,
    out_dir: str = "artifacts/single_cell/cell_states",
    groupby: str = "leiden",
) -> Path:
    """
    Export marker genes per cluster to CSV.

    Returns path to the written CSV.
    """
    import pandas as pd
    import scanpy as sc

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "rank_genes_groups" not in adata.uns:
        logger.warning("rank_genes_groups not found in adata.uns; run find_marker_genes first.")
        return out / "marker_genes_by_cluster.csv"

    result = sc.get.rank_genes_groups_df(adata, group=None)
    out_path = out / "marker_genes_by_cluster.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Marker genes saved to {out_path} ({len(result)} rows)")
    return out_path
