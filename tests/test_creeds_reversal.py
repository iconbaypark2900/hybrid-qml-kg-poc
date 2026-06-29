"""Tests for CREEDS → Hetionet compound mapping and reversal scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from entity_resolution.compound_mapper import CompoundMapper
from perturbation_layer.creeds_reversal import (
    CreedsProfile,
    aggregate_profile_scores,
    best_reversal_for_candidate,
    build_creeds_profile_index,
    cosine01,
    enrich_candidates_with_creeds,
    load_creeds_reversal_context,
    normalize_compound_name,
    profiles_for_candidate,
    score_profile_cosine,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
CREEDS = REPO_ROOT / "artifacts/external/creeds/single_drug_perturbations-v1.0.json"
SIGNATURE = REPO_ROOT / "artifacts/signatures/tcga_brca_60/disease_signature.json"
GENE_MAP = REPO_ROOT / "artifacts/external/gdc_tcga_brca/converted/tcga_brca_gene_map.csv"


def test_normalize_compound_name_strips_salts_and_punctuation() -> None:
    assert normalize_compound_name("4-hydroxytamoxifen") == "4hydroxytamoxifen"
    assert normalize_compound_name("Metformin hydrochloride") == "metformin"
    assert normalize_compound_name("Doxorubicin (liposomal)") == "doxorubicin"


def test_aggregate_profile_scores_mean_top3() -> None:
    assert aggregate_profile_scores([0.2, 0.8, 0.5, 0.1]) == pytest.approx((0.8 + 0.5 + 0.2) / 3)
    assert aggregate_profile_scores([0.4]) == 0.4


def test_best_reversal_aggregates_multiple_profiles() -> None:
    profiles = [
        CreedsProfile(
            creeds_id="a",
            drug_name="DrugA",
            compound_hetionet_id="Compound::DB00001",
            up_genes=("G1",),
            down_genes=("G2",),
            up_gene_scores=(("G1", 1.0),),
            down_gene_scores=(("G2", -1.0),),
            cell_type="cell",
            geo_id="GSE1",
        ),
        CreedsProfile(
            creeds_id="b",
            drug_name="DrugA",
            compound_hetionet_id="Compound::DB00001",
            up_genes=("G3",),
            down_genes=("G4",),
            up_gene_scores=(("G3", 1.0),),
            down_gene_scores=(("G4", -1.0),),
            cell_type="cell",
            geo_id="GSE2",
        ),
    ]
    by_hetionet = {"Compound::DB00001": profiles}
    by_name: dict[str, list[CreedsProfile]] = {}
    mapper = CompoundMapper()

    normalized, _, _, profile, count = best_reversal_for_candidate(
        "DrugA",
        "Compound::DB00001",
        disease_up=["G2"],
        disease_down=["G1"],
        by_hetionet=by_hetionet,
        by_name=by_name,
        mapper=mapper,
    )
    assert count == 2
    assert profile is not None
    assert 0.0 <= normalized <= 1.0


def test_cosine_reversal_prefers_opposing_effect() -> None:
    profile = CreedsProfile(
        creeds_id="cosine-test",
        drug_name="TestDrug",
        compound_hetionet_id=None,
        up_genes=("G2",),
        down_genes=("G1",),
        up_gene_scores=(("G2", 1.0),),
        down_gene_scores=(("G1", -1.0),),
        cell_type="cell",
        geo_id="GSE-test",
    )
    signature_genes = ["G1", "G2"]
    lfc_by_symbol = {"G1": 2.0, "G2": -1.5}
    score = score_profile_cosine(profile, signature_genes, lfc_by_symbol)
    neutral = cosine01(np.zeros(2), -np.array([2.0, -1.5]))
    assert score > neutral


@pytest.mark.skipif(not CREEDS.exists(), reason="CREEDS artifact not downloaded")
def test_tamoxifen_profile_matches_by_substring() -> None:
    records = json.loads(CREEDS.read_text(encoding="utf-8"))
    mapper = CompoundMapper()
    by_hetionet, by_name = build_creeds_profile_index(records, organism="human", mapper=mapper)
    profiles = profiles_for_candidate(
        "tamoxifen",
        "Compound::DB00675",
        by_hetionet=by_hetionet,
        by_name=by_name,
        mapper=mapper,
    )
    assert profiles
    assert any("tamoxifen" in p.drug_name.lower() for p in profiles)


@pytest.mark.skipif(
    not all(p.exists() for p in (CREEDS, SIGNATURE, GENE_MAP)),
    reason="omics artifacts missing",
)
def test_enrich_candidates_assigns_creeds_reversal() -> None:
    context = load_creeds_reversal_context(
        creeds_path=CREEDS,
        disease_signature_path=SIGNATURE,
        gene_map_path=GENE_MAP,
    )
    candidates = [
        {
            "compound": "Anastrozole",
            "compound_hetionet_id": "Compound::DB00412",
            "disease": "breast cancer",
            "disease_hetionet_id": "Disease::DOID:1612",
            "kg_rotate_score": 0.8,
            "qsvc_score": 0.1,
            "classical_ensemble_score": 0.8,
        }
    ]
    enriched, stats = enrich_candidates_with_creeds(candidates, context)
    assert stats["n_creeds_matched"] == 1
    assert stats["profile_aggregation"] == "mean_top3"
    assert 0.0 <= enriched[0]["signature_reversal_score"] <= 1.0
    assert enriched[0].get("creeds_id")


@pytest.mark.skipif(
    not all(p.exists() for p in (CREEDS, SIGNATURE, GENE_MAP)),
    reason="omics artifacts missing",
)
def test_cosine_context_enriches_candidates() -> None:
    context = load_creeds_reversal_context(
        creeds_path=CREEDS,
        disease_signature_path=SIGNATURE,
        gene_map_path=GENE_MAP,
        reversal_method="cosine",
    )
    candidates = [
        {
            "compound": "Paclitaxel",
            "compound_hetionet_id": "Compound::DB01229",
            "disease": "breast cancer",
            "disease_hetionet_id": "Disease::DOID:1612",
            "kg_rotate_score": 0.8,
            "qsvc_score": 0.1,
            "classical_ensemble_score": 0.8,
        }
    ]
    enriched, stats = enrich_candidates_with_creeds(candidates, context)
    assert stats["reversal_method"] == "cosine"
    assert stats["n_creeds_matched"] == 1
    assert 0.0 <= enriched[0]["signature_reversal_score"] <= 1.0
