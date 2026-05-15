from __future__ import annotations

import logging
from typing import Dict, Optional

from single_cell_layer.mitochondrial_metrics import compute_mito_metrics
from single_cell_layer.doublet_detection import detect_doublets
from single_cell_layer.filtering import filter_cells_and_genes

logger = logging.getLogger(__name__)


def run_qc(adata, config: Optional[Dict] = None):
    """
    Full QC pipeline: mito metrics → doublet detection → cell/gene filtering.

    Steps are applied in-place and the filtered AnnData is returned.
    All steps respect config thresholds from single_cell_config.yaml.
    """
    config = config or {}
    logger.info(f"Starting QC. Input: {adata.n_obs} cells × {adata.n_vars} genes")

    compute_mito_metrics(adata, config)
    detect_doublets(adata, config)
    n_cb, n_ca, n_gb, n_ga = filter_cells_and_genes(adata, config)

    logger.info(
        f"QC complete. "
        f"Cells: {n_cb} → {n_ca} ({n_cb - n_ca} removed). "
        f"Genes: {n_gb} → {n_ga} ({n_gb - n_ga} removed)."
    )
    return adata
