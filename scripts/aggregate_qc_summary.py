#!/usr/bin/env python3
"""
aggregate_qc_summary.py — Join per-dataset QC reports into a single table.

Scans `artifacts/single_cell/qc/` (and any directory matching `qc/*_report.md`)
for QC artifacts written by `single_cell_layer.qc_report.write_qc_report`,
extracts the key counts, and emits Table 6 from the paper:

    Dataset | n cells | n genes | mito % threshold | cells removed | doublets flagged

Outputs:
    artifacts/single_cell/qc/qc_summary_table.csv
    artifacts/single_cell/qc/qc_summary_table.md

Usage:
    python scripts/aggregate_qc_summary.py
    python scripts/aggregate_qc_summary.py --qc-dir artifacts/single_cell/qc --output artifacts/single_cell/qc
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("aggregate_qc_summary")


# qc_report.write_qc_report emits both a markdown and a JSON sidecar; prefer
# JSON when it exists. Fall back to regex parsing of the markdown.
_MD_PATTERNS = {
    "dataset": re.compile(r"^\*\*Dataset:\*\*\s*(.+)$", re.MULTILINE),
    "n_cells": re.compile(r"^\*\*Cells:\*\*\s*(\d+)", re.MULTILINE),
    "n_genes": re.compile(r"^\*\*Genes:\*\*\s*(\d+)", re.MULTILINE),
    "mito_threshold": re.compile(r"mito\D*([0-9.]+)\s*%", re.IGNORECASE),
    "cells_removed": re.compile(r"removed:?\s*(\d+)\s*cells", re.IGNORECASE),
    "doublets_flagged": re.compile(r"doublets?\D*(\d+)", re.IGNORECASE),
}


def _parse_md(path: Path) -> Dict[str, Optional[str]]:
    text = path.read_text(encoding="utf-8")
    out: Dict[str, Optional[str]] = {"source_report": str(path)}
    for key, pat in _MD_PATTERNS.items():
        m = pat.search(text)
        out[key] = m.group(1).strip() if m else None
    if not out.get("dataset"):
        # Fall back to filename stem.
        out["dataset"] = path.stem.replace("_qc_report", "")
    return out


def _parse_json(path: Path) -> Dict[str, Optional[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": raw.get("dataset") or path.stem.replace("_qc", ""),
        "n_cells": raw.get("n_cells"),
        "n_genes": raw.get("n_genes"),
        "mito_threshold": raw.get("mito_threshold") or raw.get("max_mito_pct"),
        "cells_removed": raw.get("cells_removed"),
        "doublets_flagged": raw.get("doublets_flagged"),
        "source_report": str(path),
    }


def scan_qc_dir(qc_dir: Path) -> List[Dict]:
    """Find all qc_report.{md,json} files and parse each into a row."""
    rows: List[Dict] = []
    if not qc_dir.exists():
        logger.warning(f"QC directory {qc_dir} does not exist.")
        return rows

    seen_datasets: set = set()
    # Prefer JSON over MD when both exist for the same dataset.
    for json_path in sorted(qc_dir.glob("**/*qc*.json")):
        row = _parse_json(json_path)
        rows.append(row)
        if row.get("dataset"):
            seen_datasets.add(row["dataset"])

    for md_path in sorted(qc_dir.glob("**/*qc_report*.md")):
        row = _parse_md(md_path)
        if row.get("dataset") in seen_datasets:
            continue  # already covered by JSON sidecar
        rows.append(row)

    logger.info(f"Aggregated {len(rows)} QC report(s) from {qc_dir}")
    return rows


def write_outputs(rows: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "qc_summary_table.csv"
    md_path = out_dir / "qc_summary_table.md"

    fieldnames = ["dataset", "n_cells", "n_genes", "mito_threshold",
                  "cells_removed", "doublets_flagged", "source_report"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (row.get(k) if row.get(k) is not None else "—") for k in fieldnames})

    md_lines = [
        "# QC Summary — Table 6",
        "",
        f"Aggregated from {len(rows)} per-dataset QC report(s).",
        "",
        "| Dataset | n cells | n genes | Mito % threshold | Cells removed | Doublets flagged |",
        "|---------|---------|---------|------------------|---------------|------------------|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row.get('dataset', '—')} "
            f"| {row.get('n_cells') or '—'} "
            f"| {row.get('n_genes') or '—'} "
            f"| {row.get('mito_threshold') or '—'} "
            f"| {row.get('cells_removed') or '—'} "
            f"| {row.get('doublets_flagged') or '—'} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    logger.info(f"CSV: {csv_path}")
    logger.info(f"MD:  {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-dir", default="artifacts/single_cell/qc",
                        help="Directory containing qc_report.{md,json} files")
    parser.add_argument("--output", default="artifacts/single_cell/qc",
                        help="Where to write qc_summary_table.{csv,md}")
    args = parser.parse_args()

    rows = scan_qc_dir(Path(args.qc_dir))
    if not rows:
        logger.warning(
            "No QC reports found. Run scripts/dgx/run_single_cell_pipeline.sh "
            "to generate one, or pass --qc-dir <path>."
        )
        # Emit empty outputs so downstream consumers don't break.
    write_outputs(rows, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
