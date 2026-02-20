#!/usr/bin/env python3
"""
Export reproducibility bundle: config, critical paths, dependency info.
Supports deterministic reproduction of experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = "results/reproducibility_bundle"


def export_bundle(output_dir: str = OUTPUT_DIR, include_embeddings: bool = False) -> str:
    """
    Save config files and critical paths to a reproducibility bundle.

    Args:
        output_dir: Directory to write bundle
        include_embeddings: If True, copy entity_embeddings.npy and entity_ids.json

    Returns:
        Path to bundle directory
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    bundle_info = {
        "timestamp": datetime.now().isoformat(),
        "config_paths": [],
        "critical_paths": {},
    }

    # Copy config files
    config_sources = [
        "config/kg_layer_config.yaml",
        "config/hypotheses/H-001.yaml",
        "config/hypotheses/H-002.yaml",
        "config/hypotheses/H-003.yaml",
        "config/hypotheses/metrics_thresholds.yaml",
        "config/hypotheses/ranking_weights.yaml",
    ]
    for src in config_sources:
        src_path = Path(src)
        if src_path.exists():
            dst = path / "config" / src_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
            bundle_info["config_paths"].append(str(dst))

    # Critical paths
    bundle_info["critical_paths"] = {
        "data_dir": "data",
        "embeddings": "data/entity_embeddings.npy",
        "entity_ids": "data/entity_ids.json",
        "hetionet_edges": "data/hetionet-v1.0-edges.sif",
        "models_dir": "models",
        "results_dir": "results",
    }

    # Optional: copy embeddings
    if include_embeddings:
        for key, rel in [("embeddings", "entity_embeddings.npy"), ("entity_ids", "entity_ids.json")]:
            src = Path("data") / rel
            if src.exists():
                dst = path / "data" / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                bundle_info["critical_paths"][f"{key}_copied"] = str(dst)

    # Write manifest
    manifest_path = path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(bundle_info, f, indent=2)

    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Export reproducibility bundle")
    parser.add_argument("-o", "--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--include-embeddings", action="store_true", help="Copy embedding files")
    args = parser.parse_args()
    out = export_bundle(args.output, include_embeddings=args.include_embeddings)
    print(f"Reproducibility bundle written to {out}")


if __name__ == "__main__":
    main()
