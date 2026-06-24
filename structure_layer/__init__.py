"""Local structure-artifact ingestion and feature extraction."""

from .artifacts import StructureArtifact, load_artifact_registry
from .feature_extraction import (
    STRUCTURE_FEATURE_SCHEMA,
    extract_structure_features,
    build_structure_feature_table,
    write_structure_feature_outputs,
)
from .protein_evidence import (
    build_protein_structure_evidence,
    filter_protein_structure_evidence,
    missing_protein_structure_evidence,
    resolve_candidate_protein_structure_evidence,
)
from .target_mapping import (
    StructureTargetIndex,
    aggregate_pair_structure_features,
    build_pair_structure_feature_matrix,
    build_pair_to_targets_from_edges,
    structure_pair_feature_names,
)

__all__ = [
    "STRUCTURE_FEATURE_SCHEMA",
    "StructureArtifact",
    "StructureTargetIndex",
    "aggregate_pair_structure_features",
    "build_pair_structure_feature_matrix",
    "build_protein_structure_evidence",
    "build_pair_to_targets_from_edges",
    "build_structure_feature_table",
    "extract_structure_features",
    "filter_protein_structure_evidence",
    "missing_protein_structure_evidence",
    "resolve_candidate_protein_structure_evidence",
    "load_artifact_registry",
    "structure_pair_feature_names",
    "write_structure_feature_outputs",
]
