from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import rapids_singlecell as rsc
    import cupy as cp
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False
    logger.warning(
        "rapids-singlecell or cupy not available. "
        "GPU backend is inactive; set single_cell.backend=cpu in config."
    )


def _require_rapids() -> None:
    if not RAPIDS_AVAILABLE:
        raise ImportError(
            "RAPIDS-singlecell is required for the GPU backend. "
            "See scripts/dgx/install_gpu_omics.sh for installation."
        )


def preprocess(adata, config: Optional[Dict] = None):
    """GPU-accelerated normalize + log1p + HVG selection via RAPIDS."""
    _require_rapids()
    cfg = config or {}
    norm_cfg = cfg.get("normalization", {})
    feat_cfg = cfg.get("feature_selection", {})

    target_sum = norm_cfg.get("target_sum", 10_000)
    log1p = norm_cfg.get("log1p", True)
    n_top_genes = feat_cfg.get("n_top_genes", 3000)

    logger.info(f"[GPU] Normalizing (target_sum={target_sum}) …")
    rsc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        rsc.pp.log1p(adata)

    logger.info(f"[GPU] Selecting {n_top_genes} highly variable genes …")
    rsc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    logger.info(f"[GPU] After HVG: {adata.n_obs} cells × {adata.n_vars} genes")
    return adata


def run_pca(adata, n_pcs: int = 50):
    """GPU PCA."""
    _require_rapids()
    logger.info(f"[GPU] PCA (n_comps={n_pcs}) …")
    rsc.pp.pca(adata, n_comps=n_pcs)
    return adata


def run_neighbors(adata, n_neighbors: int = 15, n_pcs: int = 50):
    """GPU kNN graph."""
    _require_rapids()
    logger.info(f"[GPU] Neighbors (k={n_neighbors}) …")
    rsc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    return adata


def run_umap(adata, min_dist: float = 0.3, n_components: int = 2):
    """GPU UMAP."""
    _require_rapids()
    logger.info("[GPU] UMAP …")
    rsc.tl.umap(adata, min_dist=min_dist, n_components=n_components)
    return adata


def run_leiden(adata, resolution: float = 1.0):
    """GPU Leiden clustering."""
    _require_rapids()
    logger.info(f"[GPU] Leiden (resolution={resolution}) …")
    rsc.tl.leiden(adata, resolution=resolution)
    return adata


def run_tsne(adata):
    """GPU t-SNE."""
    _require_rapids()
    logger.info("[GPU] t-SNE …")
    rsc.tl.tsne(adata)
    return adata
