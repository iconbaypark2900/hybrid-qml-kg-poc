from __future__ import annotations

from pathlib import Path

from middleware.repurposing_workbench import (
    _bundle_path_for_disease,
    build_repurposing_candidates,
    list_repurposing_diseases,
)


def test_bundle_paths_registered() -> None:
    assert _bundle_path_for_disease("brca_external_validation") is not None
    assert _bundle_path_for_disease("brca_external_validation_organism_any") is not None


def test_build_repurposing_candidates_uses_artifact_bundle_when_present() -> None:
    path = _bundle_path_for_disease("brca_external_validation")
    if path is None or not path.exists():
        return
    resp = build_repurposing_candidates("brca_external_validation")
    assert resp["manifest"]["source"] == "repurposing_evidence_bundle"
    assert len(resp["candidates"]) > 0
    assert resp["candidates"][0]["compound_name"]
    labels = [item["label"] for item in resp["candidates"][0]["evidence_components"]]
    assert "CREEDS" in labels


def test_list_repurposing_diseases_includes_organism_any() -> None:
    resp = list_repurposing_diseases()
    ids = {item["id"] for item in resp["diseases"]}
    assert "brca_external_validation_organism_any" in ids
    assert "all_pairs_kg_omics" in ids


def test_organism_any_bundle_top_candidate_differs_from_human_when_both_exist() -> None:
    human_path = _bundle_path_for_disease("brca_external_validation")
    any_path = _bundle_path_for_disease("brca_external_validation_organism_any")
    if not human_path or not any_path or not human_path.exists() or not any_path.exists():
        return
    human = build_repurposing_candidates("brca_external_validation")
    any_org = build_repurposing_candidates("brca_external_validation_organism_any")
    human_top = human["candidates"][0]["compound_name"]
    any_top = any_org["candidates"][0]["compound_name"]
    assert human_top == "Vemurafenib"
    assert any_top == "Prednisolone"
