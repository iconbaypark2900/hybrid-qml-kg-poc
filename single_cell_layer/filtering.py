from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def filter_cells_and_genes(adata, config: Optional[Dict] = None) -> Tuple[int, int, int, int]:
    """
    Filter low-quality cells and low-expression genes per config thresholds.

    Applies (in order):
      1. Gene filter: min_cells_per_gene
      2. Cell filter: min_genes_per_cell, max_genes_per_cell, max_mito_pct
      3. Remove predicted doublets (if column present)

    Returns (n_cells_before, n_cells_after, n_genes_before, n_genes_after).
    Modifies adata in-place.
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy required. Run: pip install -r requirements-omics.txt")

    qc_cfg = (config or {}).get("qc", {})
    min_genes = qc_cfg.get("min_genes_per_cell", 200)
    max_genes = qc_cfg.get("max_genes_per_cell", 6000)
    max_mito = qc_cfg.get("max_mito_pct", 20.0)
    min_cells = qc_cfg.get("min_cells_per_gene", 3)

    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars

    # Gene filter
    sc.pp.filter_genes(adata, min_cells=min_cells)
    logger.info(f"After gene filter (min_cells={min_cells}): {adata.n_vars} genes")

    # Cell filter — gene count bounds
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_cells(adata, max_genes=max_genes)
    logger.info(
        f"After cell filter (genes {min_genes}–{max_genes}): {adata.n_obs} cells"
    )

    # Mito filter
    if "pct_counts_mt" in adata.obs.columns:
        before_mito = adata.n_obs
        adata._inplace_subset_obs(adata.obs["pct_counts_mt"] < max_mito)
        removed = before_mito - adata.n_obs
        logger.info(
            f"After mito filter (<{max_mito}%): {adata.n_obs} cells "
            f"({removed} removed)"
        )

    # Doublet filter
    if "predicted_doublet" in adata.obs.columns:
        before_dbl = adata.n_obs
        adata._inplace_subset_obs(~adata.obs["predicted_doublet"])
        logger.info(
            f"After doublet removal: {adata.n_obs} cells "
            f"({before_dbl - adata.n_obs} doublets removed)"
        )

    n_cells_after = adata.n_obs
    n_genes_after = adata.n_vars
    logger.info(
        f"Filtering complete: {n_cells_before}→{n_cells_after} cells, "
        f"{n_genes_before}→{n_genes_after} genes"
    )
    return n_cells_before, n_cells_after, n_genes_before, n_genes_after
