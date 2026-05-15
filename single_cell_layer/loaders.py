from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def load_single_cell_config(path: str = "config/single_cell_config.yaml") -> Dict:
    """Load single-cell config from YAML; returns defaults if file not found."""
    import yaml
    p = Path(path)
    if not p.exists():
        logger.warning(f"Config not found at {p}, using defaults.")
        return {
            "single_cell": {"backend": "auto", "prefer_gpu": True, "fallback_to_cpu": True},
            "input": {"format": "h5ad", "condition_key": "condition",
                      "batch_key": "sample_id", "cell_type_key": "cell_type"},
            "qc": {"min_genes_per_cell": 200, "max_genes_per_cell": 6000,
                   "max_mito_pct": 20.0, "min_cells_per_gene": 3, "mito_prefix": "MT-"},
            "output": {"artifacts_dir": "artifacts/single_cell"},
        }
    with open(p) as f:
        return yaml.safe_load(f)


def load(
    path: str,
    config: Optional[Dict] = None,
    validate: bool = True,
):
    """
    Auto-detect format from path and load single-cell data.

    Supported formats:
      - .h5ad files  → ingest_h5ad.load_h5ad()
      - directories  → ingest_10x.load_10x()

    Returns AnnData object.
    """
    if config is None:
        config = load_single_cell_config()

    p = Path(path)

    # Format override from config
    fmt = config.get("input", {}).get("format", "auto")

    if fmt == "h5ad" or (fmt == "auto" and p.suffix.lower() == ".h5ad"):
        from single_cell_layer.ingest_h5ad import load_h5ad
        return load_h5ad(str(p), config=config, validate=validate)

    if fmt == "10x" or (fmt == "auto" and p.is_dir()):
        from single_cell_layer.ingest_10x import load_10x
        return load_10x(str(p), config=config, validate=validate)

    raise ValueError(
        f"Cannot determine single-cell format for '{path}'. "
        "Set input.format to 'h5ad' or '10x' in config."
    )


# ------------------------------------------------------------------
# CLI entry point: python -m single_cell_layer.loaders --file demo.h5ad
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    ap = argparse.ArgumentParser(description="Load and validate a single-cell dataset.")
    ap.add_argument("--file", required=True, help="Path to .h5ad or 10x directory")
    ap.add_argument("--config", default="config/single_cell_config.yaml")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    cfg = load_single_cell_config(args.config)
    adata = load(args.file, config=cfg, validate=not args.no_validate)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    sys.exit(0)
