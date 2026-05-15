from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def export_disease_signature(
    signature: Dict,
    out_dir: str = "artifacts/signatures",
    filename: Optional[str] = None,
) -> Path:
    """
    Write a disease signature dict to JSON.

    Returns path to the written file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fname = filename or "disease_signature.json"
    out_path = out / fname
    out_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")

    logger.info(
        f"Disease signature exported to {out_path} "
        f"({len(signature.get('up_genes', []))} up, "
        f"{len(signature.get('down_genes', []))} down genes)"
    )
    return out_path


def export_cell_type_signatures(
    signatures: Dict[str, Dict],
    out_dir: str = "artifacts/signatures",
) -> Dict[str, Path]:
    """Export one JSON file per cell type."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    for ct, sig in signatures.items():
        safe_ct = ct.replace(" ", "_").replace("/", "-")
        path = export_disease_signature(sig, out_dir=str(out), filename=f"signature_{safe_ct}.json")
        paths[ct] = path

    # Also write a combined manifest
    manifest_path = out / "signature_manifest.json"
    manifest_path.write_text(
        json.dumps({ct: str(p) for ct, p in paths.items()}, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Exported {len(paths)} cell-type signatures to {out}")
    return paths
