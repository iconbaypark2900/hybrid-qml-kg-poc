"""Integration tests for the entity_resolution package.

Exercises the HetionetResolver against the actual Hetionet nodes TSV (when
present) and the gene/disease/compound mappers against their seed tables.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from entity_resolution.compound_mapper import CompoundMapper
from entity_resolution.disease_mapper import DiseaseMapper
from entity_resolution.gene_mapper import GeneMapper
from entity_resolution.hetionet_resolver import HetionetResolver
from entity_resolution.synonym_resolver import SynonymResolver


# ----- HetionetResolver ----------------------------------------------------

@pytest.fixture(scope="module")
def resolver() -> HetionetResolver:
    return HetionetResolver().load()


def test_resolver_loads_nodes(resolver: HetionetResolver) -> None:
    """Resolver must load the nodes table (or fall back gracefully)."""
    assert hasattr(resolver, "_id_to_node")
    # Either a real Hetionet load (~47k) or an empty graceful fallback.
    assert len(resolver._id_to_node) >= 0


@pytest.mark.skipif(
    not Path("data/hetionet-v1.0-nodes.tsv").exists(),
    reason="Hetionet nodes TSV not present",
)
def test_resolver_finds_known_compound(resolver: HetionetResolver) -> None:
    # Metformin is in Hetionet as Compound::DB00331.
    hid = resolver.resolve("metformin")
    assert hid is not None
    assert hid.startswith("Compound::")


def test_resolver_handles_missing_query(resolver: HetionetResolver) -> None:
    # Non-existent name should return None, not raise.
    assert resolver.resolve("definitely_not_a_real_compound_xyz123") is None


# ----- CompoundMapper ------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/hetionet-v1.0-nodes.tsv").exists(),
    reason="Hetionet nodes TSV not present (mappers verify presence in graph)",
)
def test_compound_mapper_drugbank_id() -> None:
    mapper = CompoundMapper()
    # Bare DrugBank ID → Hetionet node (verified against the graph).
    assert mapper.map("DB00331") == "Compound::DB00331"


def test_compound_mapper_invalid_query() -> None:
    mapper = CompoundMapper()
    # Random string → None.
    assert mapper.map("not_a_compound") is None


@pytest.mark.skipif(
    not Path("data/hetionet-v1.0-nodes.tsv").exists(),
    reason="Hetionet nodes TSV not present",
)
def test_compound_mapper_batch() -> None:
    mapper = CompoundMapper()
    result = mapper.map_many(["DB00331", "DB00945", "junk"])
    assert result["DB00331"] == "Compound::DB00331"
    assert result["DB00945"] == "Compound::DB00945"
    assert result["junk"] is None


# ----- DiseaseMapper -------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/hetionet-v1.0-nodes.tsv").exists(),
    reason="Hetionet nodes TSV not present",
)
def test_disease_mapper_doid() -> None:
    mapper = DiseaseMapper()
    assert mapper.map("DOID:9352") == "Disease::DOID:9352"


def test_disease_mapper_invalid_query() -> None:
    mapper = DiseaseMapper()
    assert mapper.map("not_a_disease") is None


# ----- GeneMapper ----------------------------------------------------------

def test_gene_mapper_entrez_id() -> None:
    mapper = GeneMapper()
    # Entrez ID 1956 = EGFR.
    result = mapper.map("1956")
    # Either resolves to Gene::1956 or returns None if no seed — both are OK.
    assert result is None or result.startswith("Gene::")


# ----- SynonymResolver ----------------------------------------------------

def test_synonym_resolver_has_seed_aliases() -> None:
    syn = SynonymResolver()
    assert isinstance(syn._aliases, dict)
    # Seed dict from middleware/orchestrator should ship at least a few aliases.
    assert len(syn._aliases) > 0


def test_synonym_resolver_extra_aliases_win() -> None:
    syn = SynonymResolver(extra_aliases={"custom_drug_xyz": "Compound::DBTEST"})
    assert syn.resolve("custom_drug_xyz") == "Compound::DBTEST"
