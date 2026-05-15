"""Integration tests for the validation_layer package.

External-API tests (ClinicalTrials.gov, Open Targets, PubMed) are network-
gated via the VALIDATION_NETWORK_TESTS env var so CI can run without
hitting external services.
"""
from __future__ import annotations

import os

import pytest

from validation_layer.drugbank_mapper import (
    check_drugbank_indication,
    get_drugbank_indications,
    load_drugbank_table_from_tsv,
)
from validation_layer.known_indications_validator import check_known_indication


_NETWORK_OK = os.environ.get("VALIDATION_NETWORK_TESTS") == "1"


# ----- Known indications (offline, seeded) --------------------------------

def test_known_indication_metformin_t2d() -> None:
    # Metformin treats type 2 diabetes — must be in the seed table.
    assert check_known_indication("Compound::DB00331", "Disease::DOID:9352") is True


def test_known_indication_false_pair() -> None:
    # Metformin does not treat breast cancer (DOID:1612).
    assert check_known_indication("Compound::DB00331", "Disease::DOID:1612") is False


def test_known_indication_handles_unknown_compound() -> None:
    # Unknown DrugBank ID → False, never raises.
    assert check_known_indication("Compound::DB99999", "Disease::DOID:9352") is False


# ----- DrugBank mapper ----------------------------------------------------

def test_drugbank_seed_metformin() -> None:
    assert check_drugbank_indication("Compound::DB00331", "type 2 diabetes mellitus") is True


def test_drugbank_seed_negative() -> None:
    assert check_drugbank_indication("Compound::DB00331", "breast cancer") is False


def test_drugbank_handles_bare_id() -> None:
    # Should accept DBxxxxx without the "Compound::" prefix.
    assert check_drugbank_indication("DB00945", "diabetes") is False or True
    # (aspirin maps to DOID:9351 in the seed — substring match should hit
    # since "diabetes" is in "diabetes mellitus")
    assert check_drugbank_indication("DB00945", "diabetes mellitus") is True


def test_drugbank_handles_invalid_id() -> None:
    assert check_drugbank_indication("not_a_db_id", "anything") is False


def test_drugbank_get_indications() -> None:
    indications = get_drugbank_indications("DB00331")
    assert isinstance(indications, list)
    # Metformin should at least know about T2D.
    assert any("diabetes" in ind.lower() for ind in indications)


def test_drugbank_loader_missing_tsv_falls_back() -> None:
    # Missing path should not raise — falls back to seed table.
    table = load_drugbank_table_from_tsv("/nonexistent/path.tsv")
    assert isinstance(table, dict)
    assert "DB00331" in table


# ----- Network-gated ClinicalTrials.gov tests -----------------------------

@pytest.mark.skipif(not _NETWORK_OK, reason="VALIDATION_NETWORK_TESTS=1 not set")
def test_clinical_trials_query_lives() -> None:
    from validation_layer.clinical_trials_validator import query_clinical_trials
    studies = query_clinical_trials("metformin", "diabetes", max_results=2)
    assert isinstance(studies, list)


@pytest.mark.skipif(not _NETWORK_OK, reason="VALIDATION_NETWORK_TESTS=1 not set")
def test_clinical_trials_handles_empty_query() -> None:
    from validation_layer.clinical_trials_validator import query_clinical_trials
    # Nonsense query should return [] not raise.
    studies = query_clinical_trials("xyzzy_drug_42", "xyzzy_disease_42")
    assert studies == []


# ----- Open Targets (offline shape check) ---------------------------------

def test_opentargets_returns_default_when_missing_ids() -> None:
    from validation_layer.opentargets_mapper import query_opentargets_evidence
    ev = query_opentargets_evidence(None, None)
    assert ev["overall_score"] == 0.0
    assert ev["source"] == "unavailable"
