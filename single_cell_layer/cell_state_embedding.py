from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from single_cell_layer.backend import get_backend

logger = logging.getLogger(__name__)


def run_umap(adata, config: Optional[Dict] = None):
    """UMAP embedding via the configured backend."""
    config = config or {}
    dim_cfg = config.get("dimensionality", {})
    min_dist = dim_cfg.get("umap_min_dist", 0.3)
    n_components = dim_cfg.get("umap_n_components", 2)

    bk = get_backend(config)
    adata = bk.run_umap(adata, min_dist=min_dist, n_components=n_components)
    logger.info("UMAP complete.")
    return adata


def export_embeddings(adata, out_dir: str = "artifacts/single_cell/cell_states") -> None:
    """
    Export PCA and UMAP embeddings to numpy/CSV artifacts.

    Writes:
      - pca_embeddings.npy
      - umap_embeddings.csv
      - clusters.csv
    """
    import numpy as np
    import pandas as pd

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "X_pca" in adata.obsm:
        np.save(out / "pca_embeddings.npy", adata.obsm["X_pca"])
        logger.info(f"PCA embeddings saved: {adata.obsm['X_pca'].shape}")

    if "X_umap" in adata.obsm:
        umap_df = pd.DataFrame(
            adata.obsm["X_umap"],
            index=adata.obs_names,
            columns=[f"UMAP_{i+1}" for i in range(adata.obsm["X_umap"].shape[1])],
        )
        umap_df.to_csv(out / "umap_embeddings.csv")
        logger.info(f"UMAP embeddings saved: {umap_df.shape}")

    if "leiden" in adata.obs.columns:
        adata.obs[["leiden"]].to_csv(out / "clusters.csv")
        logger.info("Cluster labels saved.")
