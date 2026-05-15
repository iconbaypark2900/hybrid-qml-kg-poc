from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import harmonypy as hm
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False
    logger.warning("harmonypy not installed; batch correction disabled.")


def run_batch_correction(adata, config: Optional[Dict] = None):
    """
    Harmony batch correction on PCA embeddings.

    Reads config keys:
      batch_correction.enabled
      batch_correction.method  ("harmony" | "none")
      batch_correction.batch_key

    Adds 'X_pca_harmony' to adata.obsm and updates adata.obsm["X_pca"]
    so downstream neighbor/UMAP steps use corrected embeddings.

    Modifies adata in-place and returns it.
    """
    config = config or {}
    bc_cfg = config.get("batch_correction", {})

    if not bc_cfg.get("enabled", True):
        logger.info("Batch correction disabled in config.")
        return adata

    method = bc_cfg.get("method", "harmony").lower()
    if method == "none":
        logger.info("Batch correction method=none; skipping.")
        return adata

    batch_key = bc_cfg.get("batch_key", "sample_id")

    if batch_key not in adata.obs.columns:
        logger.warning(
            f"Batch key '{batch_key}' not in adata.obs; skipping batch correction."
        )
        return adata

    if "X_pca" not in adata.obsm:
        logger.warning("PCA not run; skipping batch correction (run run_pca first).")
        return adata

    if not HARMONY_AVAILABLE:
        logger.warning("harmonypy not available; skipping batch correction.")
        return adata

    import numpy as np

    logger.info(f"Running Harmony batch correction (batch_key='{batch_key}') …")
    pca_embeddings = adata.obsm["X_pca"]
    meta_data = adata.obs[[batch_key]].copy()

    ho = hm.run_harmony(pca_embeddings, meta_data, batch_key)
    adata.obsm["X_pca_harmony"] = ho.Z_corr.T
    # Use corrected embeddings for downstream steps
    adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]

    logger.info("Harmony correction complete. X_pca updated with corrected embeddings.")
    return adata
