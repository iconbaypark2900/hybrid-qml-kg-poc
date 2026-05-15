"""Integration tests for the perturbation_layer package.

Covers: reversal score math, CMap loader, feature vector builder, registry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from perturbation_layer.cmap_loader import cmap_to_up_down
from perturbation_layer.perturbation_registry import PerturbationRegistry
from perturbation_layer.reversal_features import build_reversal_feature_vector
from perturbation_layer.reversal_score import (
    compute_reversal_score,
    compute_reversal_scores_batch,
)


# ----- compute_reversal_score ---------------------------------------------

def test_reversal_perfect_opposition() -> None:
    score = compute_reversal_score(
        disease_up=["A", "B", "C"],
        disease_down=["D", "E"],
        drug_up=["D", "E"],
        drug_down=["A", "B", "C"],
    )
    assert score > 0.9


def test_reversal_no_overlap() -> None:
    score = compute_reversal_score(["A"], ["B"], ["C"], ["D"])
    assert score == 0.0


def test_reversal_aggravation_is_negative() -> None:
    # Drug mirrors disease direction → negative score.
    score = compute_reversal_score(
        disease_up=["A", "B"],
        disease_down=["C", "D"],
        drug_up=["A", "B"],
        drug_down=["C", "D"],
    )
    assert score < 0.0


def test_reversal_empty_inputs() -> None:
    assert compute_reversal_score([], [], [], []) == 0.0


def test_reversal_clipped_to_unit_interval() -> None:
    # Even with overwhelming evidence, magnitude must not exceed 1.
    big_up = [f"g{i}" for i in range(200)]
    big_down = [f"h{i}" for i in range(200)]
    score = compute_reversal_score(big_up, big_down, big_down, big_up)
    assert -1.0 <= score <= 1.0


# ----- batch scoring -------------------------------------------------------

def test_batch_scoring_returns_dict() -> None:
    disease = {"up_genes": ["A", "B"], "down_genes": ["C", "D"]}
    drugs = {
        "perfect": {"up_genes": ["C", "D"], "down_genes": ["A", "B"]},
        "neutral": {"up_genes": ["X"], "down_genes": ["Y"]},
    }
    scores = compute_reversal_scores_batch(disease, drugs)
    assert set(scores.keys()) == {"perfect", "neutral"}
    assert scores["perfect"] > scores["neutral"]


# ----- feature vector builder ----------------------------------------------

def test_feature_vector_shape() -> None:
    fv = build_reversal_feature_vector(
        overall_reversal=0.7,
        cell_type_scores={"T_cell": 0.6, "B_cell": 0.8},
        pathway_scores={"NF_kB": 0.4, "IL6": 0.7, "TNF": 0.3},
    )
    assert fv.shape == (6,)
    assert fv.dtype == np.float32
    # First element is overall_reversal.
    assert fv[0] == pytest.approx(0.7)


def test_feature_vector_missing_dicts_zero_filled() -> None:
    fv = build_reversal_feature_vector(overall_reversal=0.5)
    assert fv.shape == (6,)
    # All but the first element should be zero.
    assert (fv[1:] == 0.0).all()


def test_feature_vector_pathway_count() -> None:
    fv = build_reversal_feature_vector(
        overall_reversal=0.5,
        pathway_scores={"a": 0.1, "b": 0.5, "c": -0.2},
    )
    # n_pathways_reversed (last element) = count where score > 0 → 2.
    assert fv[-1] == pytest.approx(2.0)


# ----- CMap loader --------------------------------------------------------

def test_cmap_to_up_down_basic() -> None:
    df = pd.DataFrame({
        "compound": ["drug_A"] * 6 + ["drug_B"] * 6,
        "gene": [f"g{i}" for i in range(6)] * 2,
        "score": [3.0, 2.0, 1.0, -1.0, -2.0, -3.0] + [2.5, 1.5, 0.5, -0.5, -1.5, -2.5],
    })
    result = cmap_to_up_down(df, top_n=2)

    assert set(result.keys()) == {"drug_A", "drug_B"}
    assert result["drug_A"]["up_genes"][0] == "g0"  # highest score
    assert result["drug_A"]["down_genes"][-1] == "g5"  # lowest score


def test_cmap_to_up_down_missing_columns_raises() -> None:
    df = pd.DataFrame({"compound": ["x"], "wrong_col": ["y"]})
    with pytest.raises(ValueError):
        cmap_to_up_down(df)


# ----- registry -----------------------------------------------------------

def test_registry_starts_empty() -> None:
    reg = PerturbationRegistry()
    assert len(reg) == 0


def test_registry_register_and_get() -> None:
    reg = PerturbationRegistry()

    class FakeSig:
        up_genes = ["A"]
        down_genes = ["B"]

    reg.register_many({"drug_X": FakeSig()})
    assert "drug_X" in reg
    assert reg.get("drug_X").up_genes == ["A"]


def test_registry_missing_compound_raises() -> None:
    reg = PerturbationRegistry()
    with pytest.raises(KeyError):
        reg.get("not_registered")
