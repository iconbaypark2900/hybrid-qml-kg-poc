from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def write_manifest(
    datasets: List[Dict],
    out_dir: str = "artifacts/single_cell/manifests",
    filename: str = "dataset_manifest.json",
) -> Path:
    """
    Write a JSON manifest describing all loaded single-cell datasets.

    Args:
        datasets: list of metadata dicts from DatasetRegistry.list_datasets()
        out_dir: output directory
        filename: manifest filename

    Returns:
        Path to the written manifest.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / filename

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_datasets": len(datasets),
        "datasets": datasets,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Dataset manifest written to {manifest_path} ({len(datasets)} datasets)")
    return manifest_path


def load_manifest(path: str) -> Dict:
    """Load an existing manifest JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
