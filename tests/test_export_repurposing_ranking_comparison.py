from __future__ import annotations

from scripts.export_repurposing_ranking_comparison import (
    build_ranking_comparison_rows,
    candidate_id_for,
)


def _sample_candidates() -> list[dict]:
    return [
        {
            "compound": "DrugA",
            "compound_hetionet_id": "Compound::DB00001",
            "disease": "breast cancer",
            "disease_hetionet_id": "Disease::DOID:1612",
            "kg_rotate_score": 0.9,
            "kg_complex_score": 0.8,
            "graph_topology_score": 0.7,
            "qsvc_score": 0.55,
            "classical_ensemble_score": 0.85,
            "signature_reversal_score": 0.6,
            "cell_type_reversal_score": 0.6,
            "pathway_reversal_score": 0.0,
            "creeds_profile_count": 2,
            "creeds_id": "creeds-1",
            "creeds_match_status": "matched_human",
            "creeds_organism": "human",
        },
        {
            "compound": "DrugB",
            "compound_hetionet_id": "Compound::DB00002",
            "disease": "breast cancer",
            "disease_hetionet_id": "Disease::DOID:1612",
            "kg_rotate_score": 0.5,
            "kg_complex_score": 0.5,
            "graph_topology_score": 0.5,
            "qsvc_score": 0.5,
            "classical_ensemble_score": 0.5,
            "signature_reversal_score": 0.0,
            "cell_type_reversal_score": 0.0,
            "pathway_reversal_score": 0.0,
            "creeds_profile_count": 0,
            "creeds_match_status": "unmatched",
            "creeds_organism": "human",
        },
        {
            "compound": "DrugC",
            "compound_hetionet_id": "Compound::DB00003",
            "disease": "breast cancer",
            "disease_hetionet_id": "Disease::DOID:1612",
            "kg_rotate_score": 0.7,
            "kg_complex_score": 0.7,
            "graph_topology_score": 0.6,
            "qsvc_score": 0.45,
            "classical_ensemble_score": 0.7,
            "signature_reversal_score": 0.8,
            "cell_type_reversal_score": 0.8,
            "pathway_reversal_score": 0.0,
            "creeds_profile_count": 1,
            "creeds_id": "creeds-3",
            "creeds_match_status": "matched_human",
            "creeds_organism": "human",
        },
    ]


REQUIRED_COLS = {
    "candidate_id",
    "compound",
    "profile_gene_overlap",
    "signature_reversal_score",
    "kg_omics_final_score",
    "kg_omics_quantum_final_score",
    "quantum_delta_score",
}


def test_candidate_id_uses_hetionet_ids() -> None:
    row = _sample_candidates()[0]
    assert candidate_id_for(row) == "Compound::DB00001::Disease::DOID:1612"


def test_build_ranking_comparison_has_required_columns() -> None:
    rows = build_ranking_comparison_rows(_sample_candidates())
    assert len(rows) == 3
    for row in rows:
        assert REQUIRED_COLS.issubset(row.keys())


def test_omics_fusion_can_exceed_kg_only_when_reversal_present() -> None:
    rows = build_ranking_comparison_rows(_sample_candidates())
    by_compound = {row["compound"]: row for row in rows}
    assert by_compound["DrugC"]["kg_omics_final_score"] >= by_compound["DrugC"]["kg_only_final_score"]
    assert by_compound["DrugB"]["profile_gene_overlap"] == 0
