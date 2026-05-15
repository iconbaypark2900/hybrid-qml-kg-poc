"""Integration tests for single_cell_layer.cell_type_signature.

Most single_cell_layer functions require scanpy/anndata; these tests cover
the pure-Python signature aggregation paths so they run in any CI venv.
"""
from __future__ import annotations

import pandas as pd
import pytest

from single_cell_layer.cell_type_signature import (
    build_per_cell_type_signatures,
    consensus_signature,
    stratify_de_by_cell_type,
    summarize_signature_overlap,
)


# ----- Stratification -----------------------------------------------------

def test_stratify_returns_one_group_per_cell_type() -> None:
    df = pd.DataFrame({
        "names": [f"g{i}" for i in range(6)],
        "cell_type": ["T", "T", "B", "B", "NK", "NK"],
        "scores": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "logfoldchanges": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "pvals_adj": [0.001] * 6,
    })
    groups = stratify_de_by_cell_type(df)
    assert set(groups.keys()) == {"T", "B", "NK"}
    assert len(groups["T"]) == 2


def test_stratify_missing_column_returns_single_cohort() -> None:
    df = pd.DataFrame({"names": ["g1"], "scores": [1.0]})
    groups = stratify_de_by_cell_type(df)
    assert list(groups.keys()) == ["all_cells"]


# ----- Per-cell-type signatures -------------------------------------------

def _make_de_frame(n_per_cell_type: int = 40, n_cell_types: int = 2) -> pd.DataFrame:
    rows = []
    for ct_idx in range(n_cell_types):
        ct = f"ct_{ct_idx}"
        for i in range(n_per_cell_type):
            sign = 1 if i % 2 == 0 else -1
            rows.append({
                "names": f"GENE_{ct_idx:02d}_{i:03d}",
                "cell_type": ct,
                "scores": sign * (5.0 - 0.05 * i),
                "logfoldchanges": sign * (2.0 - 0.02 * i),
                "pvals_adj": 0.001,
            })
    return pd.DataFrame(rows)


def test_per_cell_type_signatures_built() -> None:
    df = _make_de_frame(n_per_cell_type=40, n_cell_types=2)
    sigs = build_per_cell_type_signatures(
        df, disease_id="Disease::DOID:9352",
        min_cells_per_type=10, top_n=10,
    )
    assert set(sigs.keys()) == {"ct_0", "ct_1"}
    for ct, sig in sigs.items():
        assert sig["disease"] == "Disease::DOID:9352"
        assert sig["cell_type"] == ct
        assert len(sig["up_genes"]) > 0


def test_per_cell_type_signatures_skip_small_cohort() -> None:
    df = _make_de_frame(n_per_cell_type=5, n_cell_types=2)
    sigs = build_per_cell_type_signatures(
        df, disease_id="Disease::DOID:9352",
        min_cells_per_type=50,  # > n_per_cell_type
    )
    assert sigs == {}


# ----- Consensus signature ------------------------------------------------

def test_consensus_signature_requires_threshold() -> None:
    sigs = {
        "T_cell": {"up_genes": ["A", "B", "C", "D"], "down_genes": ["E", "F"]},
        "B_cell": {"up_genes": ["A", "B", "X", "Y"], "down_genes": ["E", "Z"]},
        "NK_cell": {"up_genes": ["A", "P", "Q", "R"], "down_genes": ["E", "S"]},
    }
    up_2, down_2 = consensus_signature(sigs, min_cell_types=2)
    assert "A" in up_2
    assert "B" in up_2
    assert "E" in down_2

    up_3, down_3 = consensus_signature(sigs, min_cell_types=3)
    assert up_3 == ["A"]  # only A appears in all three
    assert down_3 == ["E"]


def test_consensus_signature_empty_input() -> None:
    up, down = consensus_signature({})
    assert up == []
    assert down == []


# ----- Overlap matrix ----------------------------------------------------

def test_signature_overlap_diagonal_is_one() -> None:
    sigs = {
        "T_cell": {"up_genes": ["A", "B", "C"], "down_genes": []},
        "B_cell": {"up_genes": ["B", "C", "D"], "down_genes": []},
    }
    mat = summarize_signature_overlap(sigs)
    assert mat.shape == (2, 2)
    assert mat.loc["T_cell", "T_cell"] == 1.0
    assert mat.loc["B_cell", "B_cell"] == 1.0
    # Jaccard of {A,B,C} ∩ {B,C,D} / {A,B,C,D} = 2/4 = 0.5
    assert mat.loc["T_cell", "B_cell"] == pytest.approx(0.5)
