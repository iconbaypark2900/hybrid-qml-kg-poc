from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def compute_batch_metrics(adata, batch_key: str = "sample_id") -> Dict:
    """
    Compute before/after batch correction mixing metrics.

    Returns a dict with:
      - n_batches: number of unique batches
      - batch_sizes: dict {batch_label: n_cells}
      - has_harmony: whether X_pca_harmony is present in obsm
    """
    if batch_key not in adata.obs.columns:
        return {"error": f"batch_key '{batch_key}' not in obs"}

    batch_counts = adata.obs[batch_key].value_counts().to_dict()
    return {
        "batch_key": batch_key,
        "n_batches": len(batch_counts),
        "batch_sizes": {str(k): int(v) for k, v in batch_counts.items()},
        "has_harmony_embedding": "X_pca_harmony" in adata.obsm,
    }


def write_integration_report(
    adata,
    config: Optional[Dict] = None,
    out_dir: str = "artifacts/single_cell/qc",
) -> Path:
    """Write a batch/integration summary report to out_dir."""
    config = config or {}
    bc_cfg = config.get("batch_correction", {})
    batch_key = bc_cfg.get("batch_key", "sample_id")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = compute_batch_metrics(adata, batch_key=batch_key)
    json_path = out / "integration_report.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    lines = [
        "# Batch Integration Report",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Batch key | `{metrics.get('batch_key', 'N/A')}` |",
        f"| Number of batches | {metrics.get('n_batches', 'N/A')} |",
        f"| Harmony correction applied | {metrics.get('has_harmony_embedding', False)} |",
        "",
        "## Batch Sizes",
        "",
        "| Batch | Cells |",
        "|-------|-------|",
    ]
    for batch, n in metrics.get("batch_sizes", {}).items():
        lines.append(f"| {batch} | {n} |")

    md_path = out / "integration_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Integration report written to {md_path}")
    return md_path
