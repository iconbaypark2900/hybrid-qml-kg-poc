from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def compute_mito_metrics(adata, config: Optional[Dict] = None) -> None:
    """
    Compute mitochondrial gene percentage and attach to adata.obs["pct_counts_mt"].

    Modifies adata in-place. Uses scanpy's pp.calculate_qc_metrics internally.
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy required. Run: pip install -r requirements-omics.txt")

    qc_cfg = (config or {}).get("qc", {})
    mito_prefix = qc_cfg.get("mito_prefix", "MT-")

    adata.var["mt"] = adata.var_names.str.startswith(mito_prefix)
    n_mito = adata.var["mt"].sum()
    logger.info(f"Mitochondrial genes found: {n_mito} (prefix='{mito_prefix}')")

    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    logger.info(
        f"Mito metrics computed. Median pct_counts_mt: "
        f"{adata.obs['pct_counts_mt'].median():.2f}%"
    )
