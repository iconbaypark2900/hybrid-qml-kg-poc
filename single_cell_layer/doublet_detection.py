from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    import scrublet as scr
    SCRUBLET_AVAILABLE = True
except ImportError:
    SCRUBLET_AVAILABLE = False
    logger.warning("scrublet not installed; doublet detection disabled.")


def detect_doublets(adata, config: Optional[Dict] = None) -> None:
    """
    Run Scrublet doublet detection and attach scores to adata.obs.

    Adds columns:
      - doublet_score: float [0, 1]
      - predicted_doublet: bool

    Modifies adata in-place. No-ops if scrublet is unavailable or
    doublet_detection.enabled = false in config.
    """
    dd_cfg = (config or {}).get("doublet_detection", {})
    if not dd_cfg.get("enabled", True):
        logger.info("Doublet detection disabled in config.")
        return

    if not SCRUBLET_AVAILABLE:
        logger.warning("Scrublet not available; skipping doublet detection.")
        return

    import numpy as np

    expected_rate = dd_cfg.get("expected_doublet_rate", 0.06)
    logger.info(f"Running Scrublet (expected_doublet_rate={expected_rate}) …")

    # Scrublet expects a raw (non-normalised) count matrix
    try:
        counts = adata.X
        if hasattr(counts, "toarray"):
            counts = counts.toarray()

        scrub = scr.Scrublet(counts, expected_doublet_rate=expected_rate)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)

        adata.obs["doublet_score"] = doublet_scores
        adata.obs["predicted_doublet"] = predicted_doublets

        n_doublets = predicted_doublets.sum()
        logger.info(
            f"Doublet detection complete: {n_doublets} predicted doublets "
            f"({n_doublets / len(adata) * 100:.1f}% of cells)"
        )
    except Exception as e:
        logger.warning(f"Scrublet failed ({e}); doublet scores not added.")
