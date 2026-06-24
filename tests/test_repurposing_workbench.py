from fastapi.testclient import TestClient

from middleware.api import app


client = TestClient(app)


def test_repurposing_diseases_exposes_audited_counts_cohort() -> None:
    response = client.get("/repurposing/diseases")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ok"
    diseases = {item["id"]: item for item in payload["diseases"]}
    assert diseases["brca_external_validation"]["evidence_status"] == "review_ready"
    assert diseases["brca_external_validation"]["sample_count"] >= 60
    assert diseases["brca_external_validation"]["smallest_class_count"] >= 25


def test_repurposing_candidates_include_all_evidence_channels_and_guardrails() -> None:
    response = client.get("/repurposing/candidates?disease_id=brca_external_validation")
    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["manifest"]["paid_or_hosted_services_required"] is False
    assert payload["manifest"]["structure_mode"] == "artifact_first"
    assert payload["manifest"]["openfold_runner"] == "deferred"
    assert "quantum_benchmark_overlay" in payload["scoring_modes"]

    candidate = payload["candidates"][0]
    component_labels = {item["label"] for item in candidate["evidence_components"]}
    assert {"KG", "RNA-seq", "Structure", "Quantum"}.issubset(component_labels)
    assert candidate["structure"]["status"] in {"artifact_backed", "missing", "parse_error", "ready"}
    assert candidate["structure_targets"]["mapping_status"] == "mapped"
    assert candidate["structure_targets"]["target_count"] >= 1
    assert candidate["protein_structures"]
    assert candidate["structure_targets"]["parsed_structure_count"] >= 1
    assert any(item["viewer"]["supports_3d"] for item in candidate["protein_structures"])
    assert candidate["classical_ml"]["role"] == "primary comparator"
    assert candidate["quantum_benchmark"]["role"] == "secondary benchmark"
    assert candidate["audit"]["clinical_claim_allowed"] is False
    assert candidate["audit"]["quantum_advantage_claim_allowed"] is False
    assert "not clinical evidence" in candidate["summary"]


def test_repurposing_candidates_include_broader_structure_coverage() -> None:
    response = client.get("/repurposing/candidates?disease_id=brca_external_validation")
    assert response.status_code == 200
    payload = response.json()
    candidates = payload["candidates"]

    lapatinib = next(item for item in candidates if item["compound_name"] == "Lapatinib")
    assert lapatinib["structure_targets"]["target_count"] == 9
    assert lapatinib["structure_targets"]["parsed_structure_count"] >= 6
    assert sum(1 for item in lapatinib["protein_structures"] if item["viewer"]["supports_3d"]) >= 6

    vemurafenib = next(item for item in candidates if item["compound_name"] == "Vemurafenib")
    assert vemurafenib["structure_targets"]["target_count"] >= 50
    assert vemurafenib["structure_targets"]["parsed_structure_count"] >= 10
    assert sum(1 for item in vemurafenib["protein_structures"] if item["viewer"]["supports_3d"]) >= 10


def test_repurposing_fallback_disease_keeps_candidates_without_rnaseq_or_quantum_claims() -> None:
    response = client.get("/repurposing/candidates?disease_id=gout_fallback")
    assert response.status_code == 200
    payload = response.json()

    assert payload["disease"]["evidence_status"] == "fallback_only"
    assert payload["candidates"]
    candidate = payload["candidates"][0]
    assert candidate["audit"]["clinical_claim_allowed"] is False
    assert candidate["audit"]["quantum_advantage_claim_allowed"] is False
    assert any("fallback" in warning for warning in candidate["audit"]["warnings"])
    assert candidate["structure_targets"] == {}
    assert candidate["protein_structures"] == []


def test_structure_artifact_endpoint_serves_only_local_structure_files() -> None:
    candidates_response = client.get("/repurposing/candidates?disease_id=brca_external_validation")
    assert candidates_response.status_code == 200
    candidate = candidates_response.json()["candidates"][0]
    artifact_path = next(
        item["viewer"]["artifact_path"]
        for item in candidate["protein_structures"]
        if item["viewer"]["supports_3d"]
    )

    response = client.get("/repurposing/structure-artifact", params={"path": artifact_path})

    assert response.status_code == 200
    assert b"ATOM" in response.content

    blocked = client.get("/repurposing/structure-artifact", params={"path": "data/hetionet-v1.0-nodes.tsv"})
    assert blocked.status_code == 403


def test_repurposing_evidence_bundle_export_downloads_json_and_markdown() -> None:
    response = client.get("/repurposing/evidence-bundle", params={"disease_id": "brca_external_validation", "format": "json"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert "Do not claim cures" in payload["claim_policy"]
    assert payload["ranking"]["candidate_count"] >= 1

    markdown = client.get("/repurposing/evidence-bundle", params={"disease_id": "brca_external_validation", "format": "markdown"})
    assert markdown.status_code == 200
    assert "# Repurposing Evidence Bundle" in markdown.text
    assert "Claim policy" in markdown.text

    missing = client.get("/repurposing/evidence-bundle", params={"disease_id": "gout_fallback"})
    assert missing.status_code == 404

    bad_format = client.get("/repurposing/evidence-bundle", params={"format": "txt"})
    assert bad_format.status_code == 400
