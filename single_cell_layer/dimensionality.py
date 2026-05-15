from __future__ import annotations

import logging
from typing import Dict, Optional

from single_cell_layer.backend import get_backend

logger = logging.getLogger(__name__)


def run_pca_and_neighbors(adata, config: Optional[Dict] = None):
    """
    PCA followed by kNN graph construction via the configured backend.

    Reads config keys:
      dimensionality.n_pcs
      dimensionality.neighbors
    """
    config = config or {}
    dim_cfg = config.get("dimensionality", {})
    n_pcs = dim_cfg.get("n_pcs", 50)
    n_neighbors = dim_cfg.get("neighbors", 15)

    bk = get_backend(config)
    adata = bk.run_pca(adata, n_pcs=n_pcs)
    adata = bk.run_neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    return adata
