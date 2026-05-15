from __future__ import annotations

import logging
from typing import Dict, Optional

from single_cell_layer.backend import get_backend

logger = logging.getLogger(__name__)


def run_clustering(adata, config: Optional[Dict] = None):
    """
    Leiden community detection via the configured backend.

    Reads config key: clustering.resolution
    Adds 'leiden' column to adata.obs.
    """
    config = config or {}
    cl_cfg = config.get("clustering", {})
    resolution = cl_cfg.get("resolution", 1.0)

    bk = get_backend(config)
    adata = bk.run_leiden(adata, resolution=resolution)

    n_clusters = adata.obs["leiden"].nunique()
    logger.info(f"Leiden clustering: {n_clusters} clusters (resolution={resolution})")
    return adata
