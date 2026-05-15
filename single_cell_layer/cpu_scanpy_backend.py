from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logger.warning("scanpy not installed. Run: pip install -r requirements-omics.txt")


def _require_scanpy() -> None:
    if not SCANPY_AVAILABLE:
        raise ImportError(
            "scanpy is required for the CPU backend. "
            "Run: pip install -r requirements-omics.txt"
        )


def preprocess(adata, config: Optional[Dict] = None):
    """
    Normalize and select highly variable genes.

    Steps: total-count normalize → log1p → highly variable gene selection.
    Modifies adata in-place and returns it.
    """
    _require_scanpy()
    cfg = config or {}
    norm_cfg = cfg.get("normalization", {})
    feat_cfg = cfg.get("feature_selection", {})

    target_sum = norm_cfg.get("target_sum", 10_000)
    log1p = norm_cfg.get("log1p", True)
    n_top_genes = feat_cfg.get("n_top_genes", 3000)

    logger.info(f"Normalizing (target_sum={target_sum}) …")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)

    logger.info(f"Selecting {n_top_genes} highly variable genes …")
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    logger.info(f"After HVG selection: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def run_pca(adata, n_pcs: int = 50):
    """PCA on the preprocessed expression matrix."""
    _require_scanpy()
    logger.info(f"Running PCA (n_comps={n_pcs}) …")
    sc.pp.pca(adata, n_comps=n_pcs)
    return adata


def run_neighbors(adata, n_neighbors: int = 15, n_pcs: int = 50):
    """Build kNN graph from PCA embedding."""
    _require_scanpy()
    logger.info(f"Computing neighbors (k={n_neighbors}) …")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    return adata


def run_umap(adata, min_dist: float = 0.3, n_components: int = 2):
    """UMAP dimensionality reduction."""
    _require_scanpy()
    logger.info("Running UMAP …")
    sc.tl.umap(adata, min_dist=min_dist, n_components=n_components)
    return adata


def run_leiden(adata, resolution: float = 1.0):
    """Leiden community detection."""
    _require_scanpy()
    logger.info(f"Running Leiden clustering (resolution={resolution}) …")
    sc.tl.leiden(adata, resolution=resolution)
    return adata


def run_tsne(adata):
    """t-SNE dimensionality reduction (supplementary)."""
    _require_scanpy()
    logger.info("Running t-SNE …")
    sc.tl.tsne(adata)
    return adata
