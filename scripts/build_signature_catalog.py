#!/usr/bin/env python3
"""
build_signature_catalog.py — Collect disease signature exports into one catalog.

Scans `artifacts/signatures/` for `disease_signature.json` files and
`cell_type_signatures.json` bundles emitted by
`single_cell_layer.signature_export.export_disease_signature`. Produces
Table 7 from the paper:

    Disease (DOID) | Tissue | n cell types | n up genes | n down genes

Outputs:
    artifacts/signatures/signature_catalog.csv
    artifacts/signatures/signature_catalog.md
    artifacts/signatures/signature_catalog.json   (full nested data)

Usage:
    python scripts/build_signature_catalog.py
    python scripts/build_signature_catalog.py --signatures-dir artifacts/signatures
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("build_signature_catalog")


def _normalise_signature_record(sig: Dict, n_cell_types: int = 1) -> Dict:
    """Coerce a single signature dict (single-cohort) into a catalog row."""
    return {
        "disease": sig.get("disease", "—"),
        "tissue": sig.get("tissue", "—"),
        "n_cell_types": n_cell_types,
        "n_up_genes": len(sig.get("up_genes", []) or []),
        "n_down_genes": len(sig.get("down_genes", []) or []),
        "n_ranked_genes": len(sig.get("ranked_genes", []) or []),
        "n_pathways": len(sig.get("pathways", []) or []),
    }


def scan_signatures(sig_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Walk sig_dir for signature JSONs. Per-disease subdirectories are supported:

        artifacts/signatures/disease_signature.json
        artifacts/signatures/DOID_9352/cell_type_signatures.json
        artifacts/signatures/DOID_9352/disease_signature.json
    """
    rows: List[Dict] = []
    full_data: List[Dict] = []

    if not sig_dir.exists():
        logger.warning(f"Signature directory {sig_dir} does not exist.")
        return rows, full_data

    for path in sorted(sig_dir.rglob("disease_signature.json")):
        try:
            sig = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue
        rows.append(_normalise_signature_record(sig, n_cell_types=1))
        full_data.append({"path": str(path), **sig})

    for path in sorted(sig_dir.rglob("cell_type_signatures.json")):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue
        # bundle = {cell_type: signature_dict}
        if not isinstance(bundle, dict) or not bundle:
            continue
        cell_types = list(bundle.keys())
        # Roll up to a single row per disease using totals across cell types.
        merged_up = set()
        merged_down = set()
        merged_pathways = set()
        any_sig = next(iter(bundle.values()))
        for sig in bundle.values():
            merged_up.update(sig.get("up_genes", []) or [])
            merged_down.update(sig.get("down_genes", []) or [])
            merged_pathways.update(p.get("pathway") if isinstance(p, dict) else str(p)
                                   for p in sig.get("pathways", []) or [])
        rows.append({
            "disease": any_sig.get("disease", "—"),
            "tissue": any_sig.get("tissue", "—"),
            "n_cell_types": len(cell_types),
            "n_up_genes": len(merged_up),
            "n_down_genes": len(merged_down),
            "n_ranked_genes": sum(len(s.get("ranked_genes", []) or []) for s in bundle.values()),
            "n_pathways": len(merged_pathways),
        })
        full_data.append({"path": str(path), "cell_types": cell_types, "bundle": bundle})

    logger.info(f"Catalogued {len(rows)} signature record(s)")
    return rows, full_data  # noqa: returns tuple intentionally


def write_outputs(rows: List[Dict], full_data: List[Dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "signature_catalog.csv"
    md_path = out_dir / "signature_catalog.md"
    json_path = out_dir / "signature_catalog.json"

    fieldnames = ["disease", "tissue", "n_cell_types", "n_up_genes",
                  "n_down_genes", "n_ranked_genes", "n_pathways"]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Disease Signature Catalog — Table 7",
        "",
        f"Catalogued from {len(rows)} signature record(s).",
        "",
        "| Disease (DOID) | Tissue | n cell types | n up genes | n down genes | n ranked | n pathways |",
        "|----------------|--------|--------------|------------|--------------|----------|------------|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['disease']} | {row['tissue']} | {row['n_cell_types']} | "
            f"{row['n_up_genes']} | {row['n_down_genes']} | "
            f"{row['n_ranked_genes']} | {row['n_pathways']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    json_path.write_text(json.dumps(full_data, indent=2), encoding="utf-8")

    logger.info(f"CSV:  {csv_path}")
    logger.info(f"MD:   {md_path}")
    logger.info(f"JSON: {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signatures-dir", default="artifacts/signatures",
                        help="Root directory containing signature JSONs")
    parser.add_argument("--output", default="artifacts/signatures",
                        help="Where to write signature_catalog.{csv,md,json}")
    args = parser.parse_args()

    rows, full = scan_signatures(Path(args.signatures_dir))
    if not rows:
        logger.warning(
            "No signatures found. Run scripts/dgx/run_single_cell_pipeline.sh "
            "or playbook 02 to generate at least one."
        )
    write_outputs(rows, full, Path(args.output))
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
