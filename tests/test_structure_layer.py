from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import pandas as pd

from structure_layer import (
    STRUCTURE_FEATURE_SCHEMA,
    aggregate_pair_structure_features,
    build_protein_structure_evidence,
    build_structure_feature_table,
    resolve_candidate_protein_structure_evidence,
    extract_structure_features,
    filter_protein_structure_evidence,
    load_artifact_registry,
    write_structure_feature_outputs,
)
from structure_layer.target_mapping import (
    StructureTargetIndex,
    build_pair_structure_feature_matrix,
    build_pair_to_targets_from_edges,
)


FIXTURE_REGISTRY = Path("tests/fixtures/structure_artifacts/registry.json")


def test_artifact_registry_loads_local_fixture_paths() -> None:
    artifacts = load_artifact_registry(FIXTURE_REGISTRY)

    assert len(artifacts) == 2
    assert artifacts[0].target_id == "Gene::1"
    assert artifacts[0].has_local_file
    assert artifacts[0].confidence["score_type"] == "pLDDT"
    assert artifacts[0].metadata["expected_residue_count"] == 5


def test_extract_pdb_structure_features_are_deterministic() -> None:
    artifact = load_artifact_registry(FIXTURE_REGISTRY)[0]

    features = extract_structure_features(artifact)

    assert features["has_structure"] == 1
    assert features["parse_success"] == 1
    assert features["missing_structure"] == 0
    assert features["residue_count"] == 4
    assert features["chain_count"] == 2
    assert features["ca_atom_count"] == 4
    assert features["residue_coverage"] == pytest.approx(0.8)
    assert features["plddt_mean"] == pytest.approx(72.0)
    assert features["plddt_median"] == pytest.approx(70.5)
    assert features["plddt_low_conf_fraction"] == pytest.approx(0.5)
    assert features["contact_count"] == 4
    assert features["contact_density"] == pytest.approx(4 / 6)
    assert features["radius_gyration"] is not None
    assert features["compactness"] is not None


def test_missing_structure_uses_explicit_missingness_not_zero_confidence() -> None:
    artifact = load_artifact_registry(FIXTURE_REGISTRY)[1]

    features = extract_structure_features(artifact)

    assert features["has_structure"] == 0
    assert features["parse_success"] == 0
    assert features["missing_structure"] == 1
    assert features["expected_residue_count"] == 7
    assert features["plddt_mean"] is None
    assert features["plddt_low_conf_fraction"] is None
    assert features["residue_coverage"] is None


def test_feature_outputs_include_csv_schema_and_provenance(tmp_path: Path) -> None:
    artifacts = load_artifact_registry(FIXTURE_REGISTRY)
    features, provenance = build_structure_feature_table(artifacts)

    outputs = write_structure_feature_outputs(features, provenance, tmp_path)

    with outputs["features"].open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    schema = json.loads(outputs["schema"].read_text(encoding="utf-8"))
    provenance_rows = json.loads(outputs["provenance"].read_text(encoding="utf-8"))

    assert len(rows) == 2
    assert schema["feature_count"] == len(STRUCTURE_FEATURE_SCHEMA)
    assert provenance_rows[0]["target_id"] == "Gene::1"
    assert provenance_rows[0]["parse_success"] == 1
    assert provenance_rows[1]["has_structure"] == 0


def test_protein_structure_evidence_is_viewer_ready_and_local_first() -> None:
    evidence = build_protein_structure_evidence(FIXTURE_REGISTRY)

    assert len(evidence) == 2
    parsed = evidence[0]
    assert parsed["target_id"] == "Gene::1"
    assert parsed["parse_success"] is True
    assert parsed["viewer"]["supports_3d"] is True
    assert parsed["viewer"]["preferred_viewer"] == "3Dmol"
    assert parsed["confidence"]["label"] == "confident"
    assert parsed["feature_summary"]["residue_count"] == 4
    assert "not clinical" in parsed["claim_policy"]

    missing = evidence[1]
    assert missing["target_id"] == "Gene::2"
    assert missing["parse_success"] is False
    assert missing["viewer"]["supports_3d"] is False


def test_protein_structure_evidence_filter_preserves_requested_target_order() -> None:
    evidence = build_protein_structure_evidence(FIXTURE_REGISTRY)

    filtered = filter_protein_structure_evidence(evidence, ["Gene::2", "Gene::1", "Gene::404"])

    assert [row["target_id"] for row in filtered] == ["Gene::2", "Gene::1"]


def test_candidate_protein_structure_evidence_records_missing_targets() -> None:
    evidence = build_protein_structure_evidence(FIXTURE_REGISTRY)

    rows = resolve_candidate_protein_structure_evidence(evidence, ["Gene::1", "Gene::404"])

    assert [row["target_id"] for row in rows] == ["Gene::1", "Gene::404"]
    assert rows[0]["viewer"]["supports_3d"] is True
    assert rows[1]["viewer"]["kind"] == "missing_local_structure_artifact"
    assert rows[1]["viewer"]["supports_3d"] is False
    assert rows[1]["artifact_available"] is False


def test_pair_structure_feature_aggregation_keeps_target_provenance() -> None:
    artifacts = load_artifact_registry(FIXTURE_REGISTRY)
    feature_rows, _ = build_structure_feature_table(artifacts)
    target_features = {row["target_id"]: row for row in feature_rows}

    aggregated = aggregate_pair_structure_features(
        {
            ("Compound::DB1", "Disease::DOID:1"): ["Gene::1", "Gene::2", "Gene::404"],
        },
        target_features,
    )

    row = aggregated[("Compound::DB1", "Disease::DOID:1")]
    assert row["structure_target_count"] == 3
    assert row["structure_feature_target_count"] == 2
    assert row["structure_feature_missing_rate"] == pytest.approx(1 / 3)
    assert row["structure_feature_target_ids"] == "Gene::1|Gene::2|Gene::404"
    assert row["structure_has_structure_mean"] == pytest.approx(0.5)
    assert row["structure_residue_count_max"] == 4


def test_pair_target_resolution_uses_shared_compound_disease_genes() -> None:
    edges = pd.DataFrame(
        [
            ("Compound::DB1", "CbG", "Gene::1"),
            ("Compound::DB1", "CdG", "Gene::2"),
            ("Compound::DB1", "CbG", "Gene::3"),
            ("Disease::DOID:1", "DaG", "Gene::1"),
            ("Disease::DOID:1", "DuG", "Gene::2"),
            ("Disease::DOID:1", "DaG", "Gene::404"),
        ],
        columns=["source", "metaedge", "target"],
    )
    pairs = pd.DataFrame(
        [("Compound::DB1", "Disease::DOID:1")],
        columns=["source", "target"],
    )

    pair_to_targets = build_pair_to_targets_from_edges(pairs, edges)

    assert pair_to_targets[("Compound::DB1", "Disease::DOID:1")] == ["Gene::1", "Gene::2"]


def test_pair_structure_feature_matrix_is_dense_and_ordered() -> None:
    artifacts = load_artifact_registry(FIXTURE_REGISTRY)
    feature_rows, _ = build_structure_feature_table(artifacts)
    target_features = {row["target_id"]: row for row in feature_rows}
    edges = pd.DataFrame(
        [
            ("Compound::DB1", "CbG", "Gene::1"),
            ("Compound::DB1", "CdG", "Gene::2"),
            ("Disease::DOID:1", "DaG", "Gene::1"),
            ("Disease::DOID:1", "DuG", "Gene::2"),
        ],
        columns=["source", "metaedge", "target"],
    )
    pairs = pd.DataFrame(
        [
            ("Compound::DB1", "Disease::DOID:1"),
            ("Compound::DB404", "Disease::DOID:404"),
        ],
        columns=["source", "target"],
    )
    index = StructureTargetIndex.from_edges(edges, target_features)

    matrix, names, provenance = build_pair_structure_feature_matrix(pairs, index)

    assert matrix.shape == (2, len(names))
    assert names[:3] == [
        "structure_target_count",
        "structure_feature_target_count",
        "structure_feature_missing_rate",
    ]
    assert provenance[0] == "Gene::1|Gene::2"
    assert matrix[0, names.index("structure_target_count")] == pytest.approx(2.0)
    assert matrix[0, names.index("structure_has_structure_mean")] == pytest.approx(0.5)
    assert matrix[1, names.index("structure_feature_missing_rate")] == pytest.approx(1.0)
