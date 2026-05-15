"""Smoke tests for benchmarking.components.

The Streamlit render_* functions are no-ops when streamlit is missing — these
tests verify they import cleanly and that the data_loader handles missing
artifacts gracefully.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarking.components.data_loader import (
    candidate_to_dict,
    filter_by_disease,
    list_diseases,
    load_cell_type_signatures,
    load_disease_signature,
    load_reversal_scores,
    load_run_summary,
    load_top_candidates,
    load_top_candidates_json,
)


# ----- Missing artifact paths return safe defaults ------------------------

def test_load_top_candidates_missing_returns_empty(tmp_path: Path) -> None:
    df = load_top_candidates(tmp_path / "nope.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_load_run_summary_missing_returns_empty_dict(tmp_path: Path) -> None:
    assert load_run_summary(tmp_path / "nope.json") == {}


def test_load_disease_signature_missing_returns_empty_dict(tmp_path: Path) -> None:
    assert load_disease_signature(tmp_path / "nope.json") == {}


def test_load_cell_type_signatures_missing_returns_empty_dict(tmp_path: Path) -> None:
    assert load_cell_type_signatures(tmp_path / "nope.json") == {}


def test_load_reversal_scores_missing_returns_empty_df(tmp_path: Path) -> None:
    df = load_reversal_scores(tmp_path / "nope.csv")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ----- Round-trip through real artifacts ----------------------------------

@pytest.fixture
def fake_candidates_dir(tmp_path: Path) -> Path:
    """Write a minimal candidates artifact set the loaders can read."""
    csv_path = tmp_path / "top_candidates.csv"
    json_path = tmp_path / "top_candidates.json"
    summary_path = tmp_path / "run_summary.json"

    df = pd.DataFrame([
        {"rank": 1, "compound": "metformin", "disease": "type 2 diabetes",
         "final_score": 0.72, "confidence_tier": 3,
         "kg_rotate_score": 0.91, "qsvc_score": 0.80,
         "classical_ensemble_score": 0.85,
         "signature_reversal_score": 0.7, "cell_type_reversal_score": 0.6,
         "pathway_reversal_score": 0.5, "clinical_evidence_score": 1.0},
        {"rank": 2, "compound": "aspirin", "disease": "diabetes",
         "final_score": 0.65, "confidence_tier": 4,
         "kg_rotate_score": 0.85, "qsvc_score": 0.72,
         "classical_ensemble_score": 0.79,
         "signature_reversal_score": 0.6, "cell_type_reversal_score": 0.5,
         "pathway_reversal_score": 0.4, "clinical_evidence_score": 0.5},
    ])
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records") + [{"explanation": "x"}]))
    summary_path.write_text(json.dumps({
        "mode": "kg+omics", "n_candidates": 2, "top_n": 2,
        "top_compound": "metformin", "top_score": 0.72,
        "tier_distribution": {"tier_1": 0, "tier_2": 0, "tier_3": 1, "tier_4": 1},
    }))
    return tmp_path


def test_load_top_candidates_round_trip(fake_candidates_dir: Path) -> None:
    df = load_top_candidates(fake_candidates_dir / "top_candidates.csv")
    assert len(df) == 2
    assert "compound" in df.columns


def test_load_run_summary_round_trip(fake_candidates_dir: Path) -> None:
    s = load_run_summary(fake_candidates_dir / "run_summary.json")
    assert s["top_compound"] == "metformin"
    assert s["tier_distribution"]["tier_3"] == 1


def test_list_diseases_unique_sorted(fake_candidates_dir: Path) -> None:
    df = load_top_candidates(fake_candidates_dir / "top_candidates.csv")
    diseases = list_diseases(df)
    assert diseases == sorted(diseases)
    assert len(set(diseases)) == len(diseases)


def test_filter_by_disease_substring_match(fake_candidates_dir: Path) -> None:
    df = load_top_candidates(fake_candidates_dir / "top_candidates.csv")
    sub = filter_by_disease(df, "diabetes")
    # Both rows contain "diabetes" in the disease column.
    assert len(sub) == 2


def test_candidate_to_dict_shape(fake_candidates_dir: Path) -> None:
    df = load_top_candidates(fake_candidates_dir / "top_candidates.csv")
    cdict = candidate_to_dict(df.iloc[0])
    assert cdict["compound"] == "metformin"
    assert cdict["final_score"] == pytest.approx(0.72)
    assert cdict["confidence_tier"] == 3


# ----- Streamlit render_* shouldn't raise when streamlit missing ----------

def test_render_functions_importable() -> None:
    # If streamlit is unavailable the components log a warning and return.
    from benchmarking.components.evidence_card import render_evidence_card
    from benchmarking.components.signature_view import render_signature_view
    from benchmarking.components.reversal_view import render_reversal_view
    from benchmarking.components.clinical_validation_view import (
        render_clinical_validation_view,
    )

    # Calling them in a non-Streamlit context must not raise.
    # (They internally try-import streamlit and bail with a warning.)
    candidate = {"compound": "x", "disease": "y", "final_score": 0.5,
                 "confidence_tier": 4, "kg_rotate_score": 0.5}
    try:
        render_evidence_card(candidate)
        render_signature_view({"up_genes": ["A"], "down_genes": ["B"]})
        render_reversal_view("drug", 0.5, {"T": 0.5}, {"path": 0.5})
        render_clinical_validation_view([candidate])
    except Exception as e:
        # If streamlit IS installed, calling these outside a Streamlit
        # session may raise — only fail if we hit something other than
        # a streamlit runtime issue.
        if "streamlit" not in str(e).lower() and "scriptrun" not in str(e).lower():
            raise
