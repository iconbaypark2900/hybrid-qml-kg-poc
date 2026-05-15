"""Integration tests for the evidence_layer package.

Covers: schema, feature fusion, confidence tiering, explanation builder,
report writer, kg-only vs kg+omics mode equivalence.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from evidence_layer.confidence_tiering import assign_tier, load_thresholds
from evidence_layer.evidence_report import write_evidence_report
from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.explanation_builder import attach_explanations, build_explanation
from evidence_layer.feature_fusion import fuse_evidence


# ----- Schema -------------------------------------------------------------

def test_schema_zero_filled_defaults() -> None:
    ef = EvidenceFeatures(
        compound="x", compound_hetionet_id="Compound::DBX",
        disease="d", disease_hetionet_id="Disease::DOID:1",
    )
    # All score fields default to 0.0 — vector is always dense.
    assert ef.kg_rotate_score == 0.0
    assert ef.signature_reversal_score == 0.0
    assert ef.confidence_tier == 4
    assert ef.explanation == ""


def test_schema_to_dict_round_trip() -> None:
    ef = EvidenceFeatures(
        compound="aspirin", compound_hetionet_id="Compound::DB00945",
        disease="diabetes", disease_hetionet_id="Disease::DOID:9351",
        kg_rotate_score=0.85, qsvc_score=0.72,
    )
    d = ef.to_dict()
    assert d["compound"] == "aspirin"
    assert d["kg_rotate_score"] == 0.85


def test_schema_feature_vector_length() -> None:
    ef = EvidenceFeatures(
        compound="x", compound_hetionet_id=None,
        disease="d", disease_hetionet_id=None,
    )
    fv = ef.feature_vector()
    assert len(fv) == 10  # 10 evidence streams


# ----- Confidence tiering -------------------------------------------------

def test_tier_assignment_thresholds() -> None:
    thresholds = load_thresholds()
    # Tier 1 = highest; Tier 4 = exploratory.
    assert assign_tier(0.95, thresholds) == 1
    assert assign_tier(0.05, thresholds) == 4


def test_tier_monotonicity() -> None:
    """Higher score should never yield a worse (numerically larger) tier."""
    thresholds = load_thresholds()
    tier_high = assign_tier(0.85, thresholds)
    tier_low = assign_tier(0.45, thresholds)
    assert tier_high <= tier_low


# ----- Feature fusion ----------------------------------------------------

def _candidate(compound: str, **scores) -> EvidenceFeatures:
    defaults = dict(
        compound=compound,
        compound_hetionet_id=f"Compound::DB{compound[:5]}",
        disease="test_disease",
        disease_hetionet_id="Disease::DOID:9999",
        kg_rotate_score=0.5,
        qsvc_score=0.5,
        classical_ensemble_score=0.5,
    )
    defaults.update(scores)
    return EvidenceFeatures(**defaults)


def test_fusion_sorts_descending() -> None:
    candidates = [
        _candidate("aspirin", kg_rotate_score=0.6),
        _candidate("metformin", kg_rotate_score=0.9, qsvc_score=0.8, classical_ensemble_score=0.85),
        _candidate("ibuprofen", kg_rotate_score=0.3),
    ]
    fused = fuse_evidence(candidates, mode="kg+omics")
    for i in range(len(fused) - 1):
        assert fused[i].final_score >= fused[i + 1].final_score


def test_fusion_kg_only_zeroes_omics() -> None:
    """kg-only mode must not let omics features influence the final score."""
    c_with_omics = _candidate(
        "metformin",
        kg_rotate_score=0.5, qsvc_score=0.5, classical_ensemble_score=0.5,
        signature_reversal_score=1.0,
        cell_type_reversal_score=1.0,
        pathway_reversal_score=1.0,
    )
    c_without_omics = _candidate(
        "metformin2",
        kg_rotate_score=0.5, qsvc_score=0.5, classical_ensemble_score=0.5,
        signature_reversal_score=0.0,
        cell_type_reversal_score=0.0,
        pathway_reversal_score=0.0,
    )
    fused = fuse_evidence([c_with_omics, c_without_omics], mode="kg-only")
    # In kg-only mode both candidates should get identical final scores.
    assert fused[0].final_score == pytest.approx(fused[1].final_score)


def test_fusion_kg_plus_omics_uses_reversal() -> None:
    """kg+omics mode should rank a high-reversal candidate above an identical
    candidate with zero reversal evidence."""
    high = _candidate("high_reversal", signature_reversal_score=0.9, cell_type_reversal_score=0.9)
    low = _candidate("low_reversal", signature_reversal_score=0.0, cell_type_reversal_score=0.0)
    fused = fuse_evidence([low, high], mode="kg+omics")
    assert fused[0].compound == "high_reversal"


def test_fusion_assigns_tier_and_explanation_after_attach() -> None:
    candidates = [_candidate("metformin", kg_rotate_score=0.91, qsvc_score=0.80)]
    fused = fuse_evidence(candidates, mode="kg+omics")
    attach_explanations(fused)
    assert fused[0].confidence_tier in {1, 2, 3, 4}
    assert "metformin" in fused[0].explanation.lower()


# ----- Report writer ------------------------------------------------------

def test_report_writer_emits_csv_and_md(tmp_path: Path) -> None:
    candidates = [
        _candidate("a", kg_rotate_score=0.8),
        _candidate("b", kg_rotate_score=0.6),
    ]
    fuse_evidence(candidates, mode="kg+omics")
    attach_explanations(candidates)
    csv_path = write_evidence_report(candidates, out_dir=str(tmp_path), top_n=2,
                                     disease_id="Disease::DOID:1")
    assert csv_path.exists()
    assert (tmp_path / "top_candidates.json").exists()
    assert (tmp_path / "final_repurposing_report.md").exists()


def test_report_writer_truncates_to_top_n(tmp_path: Path) -> None:
    candidates = [_candidate(f"d{i}", kg_rotate_score=0.5 + i * 0.01) for i in range(20)]
    fuse_evidence(candidates, mode="kg+omics")
    write_evidence_report(candidates, out_dir=str(tmp_path), top_n=5)
    data = json.loads((tmp_path / "top_candidates.json").read_text())
    assert len(data) == 5
