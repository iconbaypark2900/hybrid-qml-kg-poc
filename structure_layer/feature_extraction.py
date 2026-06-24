"""Deterministic feature extraction for local structure artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from .artifacts import StructureArtifact
from .provenance import build_feature_provenance


STRUCTURE_FEATURE_SCHEMA: list[dict[str, str]] = [
    {"name": "target_id", "type": "string"},
    {"name": "target_name", "type": "string"},
    {"name": "has_structure", "type": "integer"},
    {"name": "parse_success", "type": "integer"},
    {"name": "missing_structure", "type": "integer"},
    {"name": "residue_count", "type": "integer"},
    {"name": "chain_count", "type": "integer"},
    {"name": "ca_atom_count", "type": "integer"},
    {"name": "expected_residue_count", "type": "integer_or_null"},
    {"name": "residue_coverage", "type": "float_or_null"},
    {"name": "plddt_mean", "type": "float_or_null"},
    {"name": "plddt_median", "type": "float_or_null"},
    {"name": "plddt_low_conf_fraction", "type": "float_or_null"},
    {"name": "contact_count", "type": "integer"},
    {"name": "contact_density", "type": "float_or_null"},
    {"name": "radius_gyration", "type": "float_or_null"},
    {"name": "compactness", "type": "float_or_null"},
    {"name": "source_tool", "type": "string"},
    {"name": "source_version", "type": "string"},
    {"name": "artifact_format", "type": "string"},
    {"name": "license_note", "type": "string"},
]

FEATURE_FIELDNAMES = [field["name"] for field in STRUCTURE_FEATURE_SCHEMA]


def extract_structure_features(
    artifact: StructureArtifact,
    *,
    low_confidence_plddt_cutoff: float = 70.0,
    contact_distance_angstrom: float = 8.0,
) -> dict[str, Any]:
    """Extract fixed-width target-level features from a local artifact."""

    base = _base_feature_row(artifact)
    if not artifact.has_local_file:
        return base

    path = Path(artifact.artifact_path)
    fmt = artifact.artifact_format or path.suffix.lower().lstrip(".")
    try:
        if fmt in {"pdb", "ent"}:
            parsed = _parse_pdb(path)
        else:
            raise ValueError(f"Unsupported structure artifact format: {fmt}")
    except Exception as exc:
        row = dict(base)
        row["parse_error"] = str(exc)
        return row

    ca_coords = parsed["ca_coords"]
    plddt_values = parsed["plddt_values"]
    residue_count = len(parsed["residues"])
    expected_residue_count = _coerce_positive_int(
        artifact.metadata.get("expected_residue_count")
    )
    residue_coverage = (
        residue_count / expected_residue_count
        if expected_residue_count
        else None
    )
    contact_count, contact_density = _contact_summary(
        ca_coords,
        distance_cutoff=contact_distance_angstrom,
    )
    radius_gyration = _radius_gyration(ca_coords)

    row = dict(base)
    row.update(
        {
            "has_structure": 1,
            "parse_success": 1,
            "missing_structure": 0,
            "residue_count": residue_count,
            "chain_count": len(parsed["chains"]),
            "ca_atom_count": len(ca_coords),
            "expected_residue_count": expected_residue_count,
            "residue_coverage": _round_or_none(residue_coverage),
            "plddt_mean": _round_or_none(_mean(plddt_values)),
            "plddt_median": _round_or_none(median(plddt_values) if plddt_values else None),
            "plddt_low_conf_fraction": _round_or_none(
                _fraction_below(plddt_values, low_confidence_plddt_cutoff)
            ),
            "contact_count": contact_count,
            "contact_density": _round_or_none(contact_density),
            "radius_gyration": _round_or_none(radius_gyration),
            "compactness": _round_or_none(
                residue_count / radius_gyration if radius_gyration else None
            ),
        }
    )
    return row


def build_structure_feature_table(
    artifacts: list[StructureArtifact],
    *,
    low_confidence_plddt_cutoff: float = 70.0,
    contact_distance_angstrom: float = 8.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return feature rows and provenance rows for a registry."""

    features = [
        extract_structure_features(
            artifact,
            low_confidence_plddt_cutoff=low_confidence_plddt_cutoff,
            contact_distance_angstrom=contact_distance_angstrom,
        )
        for artifact in artifacts
    ]
    provenance = [
        build_feature_provenance(artifact, feature_row).to_dict()
        for artifact, feature_row in zip(artifacts, features)
    ]
    return features, provenance


def write_structure_feature_outputs(
    features: list[dict[str, Any]],
    provenance: list[dict[str, Any]],
    out_dir: str | Path,
) -> dict[str, Path]:
    """Write CSV, schema JSON, and provenance JSON outputs."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    feature_path = out_path / "target_structure_features.csv"
    schema_path = out_path / "target_structure_features.schema.json"
    provenance_path = out_path / "target_structure_features.provenance.json"

    with feature_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(features)

    schema_doc = {
        "schema_version": "1.0",
        "feature_count": len(FEATURE_FIELDNAMES),
        "fields": STRUCTURE_FEATURE_SCHEMA,
    }
    schema_path.write_text(json.dumps(schema_doc, indent=2) + "\n", encoding="utf-8")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return {
        "features": feature_path,
        "schema": schema_path,
        "provenance": provenance_path,
    }


def _base_feature_row(artifact: StructureArtifact) -> dict[str, Any]:
    return {
        "target_id": artifact.target_id,
        "target_name": artifact.target_name,
        "has_structure": 0,
        "parse_success": 0,
        "missing_structure": 1,
        "residue_count": 0,
        "chain_count": 0,
        "ca_atom_count": 0,
        "expected_residue_count": _coerce_positive_int(
            artifact.metadata.get("expected_residue_count")
        ),
        "residue_coverage": None,
        "plddt_mean": None,
        "plddt_median": None,
        "plddt_low_conf_fraction": None,
        "contact_count": 0,
        "contact_density": None,
        "radius_gyration": None,
        "compactness": None,
        "source_tool": artifact.source_tool,
        "source_version": artifact.source_version,
        "artifact_format": artifact.artifact_format,
        "license_note": artifact.license_note,
    }


def _parse_pdb(path: Path) -> dict[str, Any]:
    residues: set[tuple[str, str, str]] = set()
    chains: set[str] = set()
    ca_coords: list[tuple[float, float, float]] = []
    plddt_values: list[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            insertion = line[26].strip()
            residues.add((chain, resseq, insertion))
            chains.add(chain)

            if atom_name != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            ca_coords.append((x, y, z))
            try:
                plddt_values.append(float(line[60:66]))
            except ValueError:
                pass

    return {
        "residues": residues,
        "chains": chains,
        "ca_coords": ca_coords,
        "plddt_values": plddt_values,
    }


def _contact_summary(
    coords: list[tuple[float, float, float]],
    *,
    distance_cutoff: float,
) -> tuple[int, float | None]:
    if len(coords) < 2:
        return 0, None
    contacts = 0
    total_pairs = 0
    cutoff_sq = distance_cutoff ** 2
    for i, left in enumerate(coords):
        for right in coords[i + 1:]:
            total_pairs += 1
            if _squared_distance(left, right) <= cutoff_sq:
                contacts += 1
    return contacts, contacts / total_pairs if total_pairs else None


def _radius_gyration(coords: list[tuple[float, float, float]]) -> float | None:
    if not coords:
        return None
    centroid = tuple(sum(point[i] for point in coords) / len(coords) for i in range(3))
    mean_sq = sum(_squared_distance(point, centroid) for point in coords) / len(coords)
    return math.sqrt(mean_sq)


def _squared_distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return sum((left[i] - right[i]) ** 2 for i in range(3))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fraction_below(values: list[float], cutoff: float) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value < cutoff) / len(values)


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    return round(value, ndigits) if value is not None else None


def _coerce_positive_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None
