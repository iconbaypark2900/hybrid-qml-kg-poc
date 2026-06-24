#!/usr/bin/env python3
"""Build local target-level structure feature tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structure_layer import (
    build_structure_feature_table,
    load_artifact_registry,
    write_structure_feature_outputs,
)


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/structure_config.yaml")
    parser.add_argument("--registry", default=None, help="Local JSON/CSV artifact registry")
    parser.add_argument("--out-dir", default=None, help="Output directory for feature artifacts")
    parser.add_argument("--low-confidence-plddt-cutoff", type=float, default=None)
    parser.add_argument("--contact-distance-angstrom", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    registry_path = args.registry or config.get("artifact_registry")
    if not registry_path:
        raise SystemExit("No structure artifact registry configured. Pass --registry or set artifact_registry.")

    out_dir = args.out_dir or config.get("output_dir", "results/structure_features")
    low_cutoff = (
        args.low_confidence_plddt_cutoff
        if args.low_confidence_plddt_cutoff is not None
        else float(config.get("low_confidence_plddt_cutoff", 70.0))
    )
    contact_distance = (
        args.contact_distance_angstrom
        if args.contact_distance_angstrom is not None
        else float(config.get("contact_distance_angstrom", 8.0))
    )

    artifacts = load_artifact_registry(registry_path)
    features, provenance = build_structure_feature_table(
        artifacts,
        low_confidence_plddt_cutoff=low_cutoff,
        contact_distance_angstrom=contact_distance,
    )
    outputs = write_structure_feature_outputs(features, provenance, out_dir)
    print(f"Wrote {len(features)} structure feature rows")
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
