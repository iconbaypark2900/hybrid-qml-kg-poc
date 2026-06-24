"""AlphaFold-like protein structure evidence records from local artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .artifacts import StructureArtifact, load_artifact_registry
from .feature_extraction import build_structure_feature_table


def build_protein_structure_evidence(
    registry: str | Path,
    *,
    low_confidence_plddt_cutoff: float = 70.0,
    contact_distance_angstrom: float = 8.0,
) -> list[dict[str, Any]]:
    """Build viewer-ready protein evidence rows from a local structure registry.

    The records are intentionally local-first: they carry artifact paths,
    feature summaries, confidence summaries, license notes, and viewer hints
    without requiring hosted AlphaFold, BioNeMo, or paid services.
    """

    artifacts = load_artifact_registry(registry)
    features, provenance = build_structure_feature_table(
        artifacts,
        low_confidence_plddt_cutoff=low_confidence_plddt_cutoff,
        contact_distance_angstrom=contact_distance_angstrom,
    )
    return [
        _protein_evidence_row(artifact, feature, provenance_row)
        for artifact, feature, provenance_row in zip(artifacts, features, provenance)
    ]


def filter_protein_structure_evidence(
    protein_evidence: Iterable[dict[str, Any]],
    target_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Return evidence rows matching target IDs, preserving target order."""

    by_target = {str(row.get("target_id")): row for row in protein_evidence}
    return [by_target[target_id] for target_id in target_ids if target_id in by_target]


def resolve_candidate_protein_structure_evidence(
    protein_evidence: Iterable[dict[str, Any]],
    target_ids: Iterable[str],
    *,
    include_missing: bool = True,
    source: str = "candidate_target_map",
) -> list[dict[str, Any]]:
    """Return target-ordered protein evidence, including explicit missing rows."""

    by_target = {str(row.get("target_id")): row for row in protein_evidence}
    rows: list[dict[str, Any]] = []
    for target_id in dict.fromkeys(str(value) for value in target_ids if str(value)):
        row = by_target.get(target_id)
        if row is not None:
            rows.append(row)
        elif include_missing:
            rows.append(missing_protein_structure_evidence(target_id, source=source))
    return rows


def missing_protein_structure_evidence(
    target_id: str,
    *,
    target_name: str | None = None,
    source: str = "candidate_target_map",
) -> dict[str, Any]:
    """Create a viewer-safe structure evidence row for a mapped target with no artifact."""

    display_name = target_name or target_id
    return {
        "target_id": target_id,
        "target_name": display_name,
        "display_name": display_name,
        "sequence_hash": None,
        "source_tool": "not_available",
        "source_version": None,
        "artifact_path": "",
        "artifact_format": None,
        "artifact_available": False,
        "parse_success": False,
        "license_note": "No local PDB/mmCIF/OpenFold artifact was provided for this mapped target.",
        "confidence": {
            "score_type": "pLDDT_or_local_confidence",
            "mean_plddt": None,
            "median_plddt": None,
            "low_confidence_fraction": None,
            "label": "unavailable",
        },
        "feature_summary": {
            "residue_count": None,
            "chain_count": None,
            "ca_atom_count": None,
            "expected_residue_count": None,
            "residue_coverage": None,
            "contact_count": None,
            "contact_density": None,
            "radius_gyration": None,
            "compactness": None,
        },
        "viewer": {
            "kind": "missing_local_structure_artifact",
            "supports_3d": False,
            "preferred_viewer": "3Dmol",
            "color_by": "chain",
            "artifact_path": "",
            "artifact_format": None,
        },
        "provenance": {
            "target_id": target_id,
            "source": source,
            "has_structure": 0,
            "parse_success": 0,
            "missing_structure": 1,
        },
        "claim_policy": "Mapped target has no local structure artifact yet; absence must be shown rather than treated as evidence of efficacy.",
    }


def _protein_evidence_row(
    artifact: StructureArtifact,
    feature: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    parse_success = int(feature.get("parse_success") or 0) == 1
    artifact_path = artifact.artifact_path or ""
    plddt_mean = feature.get("plddt_mean")
    confidence_label = _confidence_label(plddt_mean)
    return {
        "target_id": artifact.target_id,
        "target_name": artifact.target_name or artifact.target_id,
        "display_name": artifact.target_name or artifact.target_id,
        "sequence_hash": artifact.sequence_hash,
        "source_tool": artifact.source_tool,
        "source_version": artifact.source_version,
        "artifact_path": artifact_path,
        "artifact_format": artifact.artifact_format,
        "artifact_available": artifact.has_local_file,
        "parse_success": parse_success,
        "license_note": artifact.license_note,
        "confidence": {
            "score_type": artifact.confidence.get("score_type") or "pLDDT_or_local_confidence",
            "mean_plddt": plddt_mean,
            "median_plddt": feature.get("plddt_median"),
            "low_confidence_fraction": feature.get("plddt_low_conf_fraction"),
            "label": confidence_label,
        },
        "feature_summary": {
            "residue_count": feature.get("residue_count"),
            "chain_count": feature.get("chain_count"),
            "ca_atom_count": feature.get("ca_atom_count"),
            "expected_residue_count": feature.get("expected_residue_count"),
            "residue_coverage": feature.get("residue_coverage"),
            "contact_count": feature.get("contact_count"),
            "contact_density": feature.get("contact_density"),
            "radius_gyration": feature.get("radius_gyration"),
            "compactness": feature.get("compactness"),
        },
        "viewer": {
            "kind": "local_structure_artifact",
            "supports_3d": parse_success and bool(artifact_path),
            "preferred_viewer": "3Dmol",
            "color_by": "confidence" if plddt_mean is not None else "chain",
            "artifact_path": artifact_path,
            "artifact_format": artifact.artifact_format,
        },
        "provenance": provenance,
        "claim_policy": "Protein structure evidence supports local hypothesis review only; it is not clinical or wet-lab validation.",
    }


def _confidence_label(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "unavailable"
    if value >= 90:
        return "very_high"
    if value >= 70:
        return "confident"
    if value >= 50:
        return "low"
    return "very_low"
