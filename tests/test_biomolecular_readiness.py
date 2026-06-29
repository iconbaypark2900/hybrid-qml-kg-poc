from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/verify_openfold_readiness.py")
SPEC = importlib.util.spec_from_file_location("verify_openfold_readiness", SCRIPT_PATH)
openfold_module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["verify_openfold_readiness"] = openfold_module
SPEC.loader.exec_module(openfold_module)


TARGET_MAP_SCRIPT_PATH = Path("scripts/build_repurposing_candidate_target_map.py")
TARGET_MAP_SPEC = importlib.util.spec_from_file_location("build_repurposing_candidate_target_map", TARGET_MAP_SCRIPT_PATH)
target_map_module = importlib.util.module_from_spec(TARGET_MAP_SPEC)
assert TARGET_MAP_SPEC and TARGET_MAP_SPEC.loader
sys.modules["build_repurposing_candidate_target_map"] = target_map_module
TARGET_MAP_SPEC.loader.exec_module(target_map_module)


AUDIT_SCRIPT_PATH = Path("scripts/audit_repurposing_workbench.py")
AUDIT_SPEC = importlib.util.spec_from_file_location("audit_repurposing_workbench", AUDIT_SCRIPT_PATH)
audit_module = importlib.util.module_from_spec(AUDIT_SPEC)
assert AUDIT_SPEC and AUDIT_SPEC.loader
sys.modules["audit_repurposing_workbench"] = audit_module
AUDIT_SPEC.loader.exec_module(audit_module)


BUNDLE_SCRIPT_PATH = Path("scripts/build_repurposing_evidence_bundle.py")
BUNDLE_SPEC = importlib.util.spec_from_file_location("build_repurposing_evidence_bundle", BUNDLE_SCRIPT_PATH)
bundle_module = importlib.util.module_from_spec(BUNDLE_SPEC)
assert BUNDLE_SPEC and BUNDLE_SPEC.loader
sys.modules["build_repurposing_evidence_bundle"] = bundle_module
BUNDLE_SPEC.loader.exec_module(bundle_module)


def test_openfold_readiness_accepts_local_artifact_fixture() -> None:
    result = openfold_module.verify_openfold_readiness(
        registry="tests/fixtures/structure_artifacts/registry.json",
    )

    assert result["status"] == "ready"
    assert result["artifact_count"] == 2
    assert result["parse_success_count"] == 1
    checks = {item["check"]: item["status"] for item in result["checks"]}
    assert checks["source_tools_open_source_or_local"] == "pass"
    assert checks["minimum_parse_success"] == "pass"
    assert "not therapeutic efficacy" in result["claim_policy"]


def test_openfold_readiness_can_reject_fixture_tools() -> None:
    result = openfold_module.verify_openfold_readiness(
        registry="tests/fixtures/structure_artifacts/registry.json",
        allow_fixture_tools=False,
    )

    assert result["status"] == "not_ready"
    checks = {item["check"]: item for item in result["checks"]}
    assert checks["source_tools_open_source_or_local"]["status"] == "fail"
    assert "fixture" in checks["source_tools_open_source_or_local"]["evidence"]


def test_candidate_target_map_builds_from_local_hetionet_fixture(tmp_path: Path) -> None:
    ranking = tmp_path / "ranking.csv"
    ranking.write_text(
        "candidate_id,compound,kg_omics_final_score\n"
        "candidate-1,Drug A,0.9\n"
        "candidate-2,Missing Drug,0.1\n",
        encoding="utf-8",
    )
    nodes = tmp_path / "nodes.tsv"
    nodes.write_text(
        "id\tname\tkind\n"
        "Compound::DB1\tDrug A\tCompound\n"
        "Disease::D1\tbreast cancer\tDisease\n",
        encoding="utf-8",
    )
    edges = tmp_path / "edges.sif"
    edges.write_text(
        "source\tmetaedge\ttarget\n"
        "Compound::DB1\tCbG\tGene::1\n"
        "Compound::DB1\tCbG\tGene::2\n"
        "Disease::D1\tDdG\tGene::1\n",
        encoding="utf-8",
    )
    out = tmp_path / "candidate_target_map.csv"

    manifest = target_map_module.build_candidate_target_map(
        ranking_comparison=ranking,
        nodes=nodes,
        edges=edges,
        out=out,
        disease_name="breast cancer",
    )

    assert manifest["status"] == "ready"
    assert manifest["mapped_candidate_count"] == 1
    rows = out.read_text(encoding="utf-8")
    assert "candidate-1" in rows
    assert "Gene::1" in rows
    assert "compound_not_found" in rows


def test_repurposing_evidence_bundle_uses_real_rnaseq_and_ranking_artifacts(tmp_path: Path) -> None:
    target_map = tmp_path / "candidate_target_map.csv"
    target_map.write_text(
        "candidate_id,compound_name,compound_kg_id,disease_name,disease_kg_id,mapping_status,target_ids,target_count,compound_gene_count,disease_gene_count,target_source,notes\n"
        "Compound::DB08881::Disease::DOID:1612,Vemurafenib,Compound::DB08881,breast cancer,Disease::DOID:1612,mapped,Gene::1|Gene::2,2,10,1040,test_fixture,\n",
        encoding="utf-8",
    )

    result = bundle_module.build_repurposing_evidence_bundle(
        evidence_verification="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/evidence_bundle_verification.json",
        external_validation="artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/external_validation.json",
        benchmark_verdict="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/quantum_value_verdict.json",
        evidence_audit="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/evidence_audit.json",
        ranking_comparison="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/ranking_comparison.csv",
        structure_registry="tests/fixtures/structure_artifacts/registry.json",
        candidate_target_map=target_map,
        out_dir=tmp_path,
        top_k=5,
    )

    assert result["status"] == "ready"
    assert result["rnaseq_proof"]["verification_failed"] == 2
    assert result["rnaseq_proof"]["external_classical_roc_auc"] == 0.973
    assert result["rnaseq_proof"]["external_quantum_adds_value"] is False
    assert result["structure_proof"]["status"] == "ready"
    assert result["protein_structure_evidence"]["status"] == "ready"
    assert result["protein_structure_evidence"]["parsed_count"] == 1
    assert result["ranking"]["candidate_count"] == 5
    assert result["ranking"]["candidates"][0]["compound_name"] == "Carboplatin"
    assert result["candidate_target_mapping"]["mapped_output_candidate_count"] >= 1
    assert result["ranking"]["score_column"] == "kg_omics_structure_score"
    vemurafenib = next(c for c in result["ranking"]["candidates"] if c["compound_name"] == "Vemurafenib")
    assert vemurafenib["structure_targets"]["target_ids"] == ["Gene::1", "Gene::2"]
    checks = {item["check"]: item["status"] for item in result["audit"]["checks"]}
    assert checks["rnaseq_evidence_bundle_verified"] == "pass"
    assert checks["audit_review_ready"] == "warn"
    assert checks["candidate_target_map_available"] == "pass"
    assert checks["candidate_targets_resolved"] == "pass"
    assert checks["protein_structure_evidence_ready"] == "pass"
    assert checks["no_quantum_advantage_claim"] == "pass"
    assert (tmp_path / "repurposing_evidence_bundle.json").exists()
    written = json.loads((tmp_path / "repurposing_evidence_bundle.json").read_text())
    assert written["status"] == "ready"


def test_repurposing_workbench_acceptance_audit_tracks_required_gates(tmp_path: Path) -> None:
    result = audit_module.build_repurposing_workbench_audit(out_dir=tmp_path)

    assert result["status"] in ("ready", "ready_with_warnings")
    assert result["failed_required_count"] == 0
    checks = {item["check"]: item["status"] for item in result["checks"]}
    assert checks["rnaseq_evidence_verified"] == "pass"
    assert checks["protein_structure_evidence_ready"] == "pass"
    assert checks["mapped_candidates_have_structure_coverage"] == "pass"
    assert checks["quantum_is_benchmark_not_advantage_claim"] == "pass"
    assert checks["frontend_exports_evidence_bundle"] == "pass"
    assert (tmp_path / "repurposing_workbench_acceptance_audit.json").exists()
    assert (tmp_path / "repurposing_workbench_acceptance_audit.md").exists()
