"""Provenance records for structure-derived feature rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import StructureArtifact


@dataclass(frozen=True)
class StructureFeatureProvenance:
    target_id: str
    source_tool: str
    source_version: str
    artifact_path: str
    artifact_format: str
    license_note: str
    has_structure: int
    parse_success: int
    parse_error: str
    sequence_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_feature_provenance(
    artifact: StructureArtifact,
    feature_row: dict[str, Any],
) -> StructureFeatureProvenance:
    return StructureFeatureProvenance(
        target_id=artifact.target_id,
        source_tool=artifact.source_tool,
        source_version=artifact.source_version,
        artifact_path=str(Path(artifact.artifact_path)) if artifact.artifact_path else "",
        artifact_format=artifact.artifact_format,
        license_note=artifact.license_note,
        has_structure=int(feature_row.get("has_structure", 0) or 0),
        parse_success=int(feature_row.get("parse_success", 0) or 0),
        parse_error=str(feature_row.get("parse_error", "") or ""),
        sequence_hash=artifact.sequence_hash,
    )
