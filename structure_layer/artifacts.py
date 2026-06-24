"""Structure artifact registry primitives.

The registry describes local files produced by open-source structure tools.
It deliberately stores provenance and licensing notes next to the artifact so
later feature rows can be traced without depending on a hosted service.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class StructureArtifact:
    """Metadata for one local structure artifact."""

    target_id: str
    target_name: str = ""
    sequence_hash: str = ""
    source_tool: str = ""
    source_version: str = ""
    artifact_path: str = ""
    artifact_format: str = ""
    license_note: str = ""
    confidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        row: dict[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "StructureArtifact":
        """Build an artifact from a JSON/CSV row.

        CSV registry fields may encode ``confidence`` and ``metadata`` as JSON
        strings. Relative artifact paths are resolved relative to the registry
        file's directory so fixtures and exported registries remain portable.
        """

        confidence = _coerce_mapping(row.get("confidence", {}))
        metadata = _coerce_mapping(row.get("metadata", {}))
        artifact_path = str(row.get("artifact_path", "") or "")
        if artifact_path and base_dir is not None:
            candidate = Path(artifact_path)
            if not candidate.is_absolute():
                artifact_path = str((base_dir / candidate).resolve())

        return cls(
            target_id=str(row.get("target_id", "") or ""),
            target_name=str(row.get("target_name", "") or ""),
            sequence_hash=str(row.get("sequence_hash", "") or ""),
            source_tool=str(row.get("source_tool", "") or ""),
            source_version=str(row.get("source_version", "") or ""),
            artifact_path=artifact_path,
            artifact_format=str(row.get("artifact_format", "") or "").lower(),
            license_note=str(row.get("license_note", "") or ""),
            confidence=confidence,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_local_file(self) -> bool:
        return bool(self.artifact_path) and Path(self.artifact_path).exists()


def load_artifact_registry(path: str | Path) -> list[StructureArtifact]:
    """Load a local structure artifact registry from JSON or CSV."""

    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Structure artifact registry not found: {registry_path}")

    suffix = registry_path.suffix.lower()
    rows: Iterable[dict[str, Any]]
    if suffix == ".json":
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("artifacts", [])
        if not isinstance(data, list):
            raise ValueError("JSON structure registry must be a list or contain an 'artifacts' list")
        rows = data
    elif suffix == ".csv":
        with registry_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported structure registry format: {suffix}")

    artifacts = [
        StructureArtifact.from_mapping(row, base_dir=registry_path.parent)
        for row in rows
    ]
    missing_ids = [idx for idx, artifact in enumerate(artifacts) if not artifact.target_id]
    if missing_ids:
        raise ValueError(f"Structure registry rows missing target_id: {missing_ids}")
    return artifacts


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Expected mapping-compatible value, got {type(value).__name__}")
