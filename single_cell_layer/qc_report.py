from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def write_qc_report(
    adata,
    config: Optional[Dict] = None,
    out_dir: str = "artifacts/single_cell/qc",
    save_plots: bool = True,
) -> Path:
    """
    Write QC summary report and optional plots to out_dir.

    Writes:
      - qc_report.md        (human-readable summary)
      - qc_summary.json     (machine-readable stats)
      - qc_plots/           (violin plots if save_plots=True)

    Returns path to qc_report.md.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats = _compute_stats(adata)
    json_path = out / "qc_summary.json"
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    report_path = out / "qc_report.md"
    report_path.write_text(_render_markdown(stats, config), encoding="utf-8")

    if save_plots:
        _save_plots(adata, out / "qc_plots")

    logger.info(f"QC report written to {report_path}")
    return report_path


def _compute_stats(adata) -> dict:
    obs = adata.obs
    stats: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
    }

    for col in ["n_genes_by_counts", "total_counts", "pct_counts_mt", "doublet_score"]:
        if col in obs.columns:
            s = obs[col]
            stats[col] = {
                "mean": round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
            }

    if "predicted_doublet" in obs.columns:
        stats["n_predicted_doublets"] = int(obs["predicted_doublet"].sum())

    if "leiden" in obs.columns:
        stats["n_clusters"] = int(obs["leiden"].nunique())

    return stats


def _render_markdown(stats: dict, config: Optional[Dict]) -> str:
    cfg_str = ""
    if config:
        qc_cfg = config.get("qc", {})
        cfg_str = (
            f"\n| min_genes_per_cell | {qc_cfg.get('min_genes_per_cell', 'N/A')} |\n"
            f"| max_genes_per_cell | {qc_cfg.get('max_genes_per_cell', 'N/A')} |\n"
            f"| max_mito_pct | {qc_cfg.get('max_mito_pct', 'N/A')} |\n"
            f"| min_cells_per_gene | {qc_cfg.get('min_cells_per_gene', 'N/A')} |"
        )

    lines = [
        "# Single-Cell QC Report",
        "",
        f"Generated: {stats.get('generated_at', '')}",
        "",
        "## Dataset Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Cells (post-filter) | {stats['n_cells']} |",
        f"| Genes (post-filter) | {stats['n_genes']} |",
    ]

    if "n_clusters" in stats:
        lines.append(f"| Leiden clusters | {stats['n_clusters']} |")
    if "n_predicted_doublets" in stats:
        lines.append(f"| Predicted doublets (removed) | {stats['n_predicted_doublets']} |")

    if cfg_str:
        lines += ["", "## QC Thresholds Applied", "", "| Parameter | Value |", "|-----------|-------|"]
        lines.append(cfg_str)

    for col, label in [
        ("n_genes_by_counts", "Genes per cell"),
        ("total_counts", "Total counts per cell"),
        ("pct_counts_mt", "Mito % per cell"),
    ]:
        if col in stats:
            s = stats[col]
            lines += [
                "",
                f"## {label}",
                "",
                "| Stat | Value |",
                "|------|-------|",
                f"| Mean | {s['mean']} |",
                f"| Median | {s['median']} |",
                f"| Min | {s['min']} |",
                f"| Max | {s['max']} |",
            ]

    return "\n".join(lines) + "\n"


def _save_plots(adata, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import scanpy as sc
        import matplotlib
        matplotlib.use("Agg")

        plot_cols = [c for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]
                     if c in adata.obs.columns]
        if plot_cols:
            sc.pl.violin(
                adata,
                plot_cols,
                jitter=0.4,
                show=False,
                save=str(plots_dir / "qc_violin.png"),
            )
            logger.info(f"QC violin plot saved to {plots_dir}/qc_violin.png")
    except Exception as e:
        logger.warning(f"Could not save QC plots: {e}")
