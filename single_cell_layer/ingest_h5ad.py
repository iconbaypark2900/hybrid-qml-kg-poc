from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from single_cell_layer.metadata_schema import validate_anndata

logger = logging.getLogger(__name__)

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("anndata not installed. Run: pip install -r requirements-omics.txt")


def load_h5ad(
    path: str,
    config: Optional[Dict] = None,
    validate: bool = True,
) -> "ad.AnnData":
    """
    Load an .h5ad file and validate its metadata schema.

    Args:
        path: Path to the .h5ad file.
        config: single_cell_config dict (from load_single_cell_config()).
        validate: If True, run metadata schema validation and log warnings.

    Returns:
        AnnData object.

    Raises:
        ImportError: if anndata is not installed.
        FileNotFoundError: if path does not exist.
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError(
            "anndata is required. Install with: pip install -r requirements-omics.txt"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"h5ad file not found: {p}")

    logger.info(f"Loading h5ad: {p}")
    adata = ad.read_h5ad(p)
    logger.info(
        f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes  |  "
        f"obs keys: {list(adata.obs.columns[:8])}"
    )

    if validate and config is not None:
        validate_anndata(adata, config)

    return adata
