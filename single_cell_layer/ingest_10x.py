from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from single_cell_layer.metadata_schema import validate_anndata

logger = logging.getLogger(__name__)

try:
    import scanpy as sc
    SCANPY_AVAILABLE = True
except ImportError:
    SCANPY_AVAILABLE = False
    logger.warning("scanpy not installed. Run: pip install -r requirements-omics.txt")


def load_10x(
    matrix_dir: str,
    config: Optional[Dict] = None,
    validate: bool = True,
    genome: Optional[str] = None,
) -> "sc.AnnData":
    """
    Load a 10x Genomics Cell Ranger output directory (matrix.mtx.gz, barcodes, features).

    Args:
        matrix_dir: Path to the folder containing matrix.mtx.gz / barcodes.tsv.gz / features.tsv.gz.
        config: single_cell_config dict.
        validate: Run metadata schema validation.
        genome: Optional genome prefix (passed to sc.read_10x_mtx).

    Returns:
        AnnData object with var_names set to gene names.
    """
    if not SCANPY_AVAILABLE:
        raise ImportError(
            "scanpy is required. Install with: pip install -r requirements-omics.txt"
        )

    d = Path(matrix_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"10x matrix directory not found: {d}")

    logger.info(f"Loading 10x Genomics data from: {d}")
    adata = sc.read_10x_mtx(str(d), var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()

    logger.info(
        f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes"
    )

    if validate and config is not None:
        validate_anndata(adata, config)

    return adata
