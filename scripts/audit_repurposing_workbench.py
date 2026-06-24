#!/usr/bin/env python3
"""Audit the local disease-to-drug repurposing workbench readiness gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BUNDLE = "artifacts/repurposing/brca_external_validation/repurposing_evidence_bundle.json"
DEFAULT_READINESS = "artifacts/structures/alphafold/brca_anastrozole_targets/openfold_readiness.json"
DEFAULT_PAGE = "frontend/app/v2/repurposing/page.tsx"
DEFAULT_API = "frontend/lib/api.ts"
DEFAULT_BUILD_ID = "frontend/.next/BUILD_ID"
DEFAULT_FRONTEND_LIVE_SMOKE = "artifacts/repurposing/brca_external_validation/frontend_live_smoke.json"


def build_repurposing_workbench_audit(
    *,
    bundle_path: str | Path = DEFAULT_BUNDLE,
    structure_readiness_path: str | Path = DEFAULT_READINESS,
    frontend_page_path: str | Path = DEFAULT_PAGE,
    frontend_api_path: str | Path = DEFAULT_API,
    frontend_build_id_path: str | Path = DEFAULT_BUILD_ID,
    frontend_live_smoke_path: str | Path = DEFAULT_FRONTEND_LIVE_SMOKE,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    bundle = _read_json(bundle_path)
    readiness = _read_json(structure_readiness_path)
    page_text = Path(frontend_page_path).read_text(encoding="utf-8")
    api_text = Path(frontend_api_path).read_text(encoding="utf-8")
    build_id_path = Path(frontend_build_id_path)
    live_smoke_path = Path(frontend_live_smoke_path)
    current_build_id = build_id_path.read_text(encoding="utf-8").strip() if build_id_path.exists() else None
    live_smoke = _read_json(live_smoke_path) if live_smoke_path.exists() else {}

    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, evidence: Any, *, required: bool = True) -> None:
        checks.append({"check": name, "status": status, "required": required, "evidence": evidence})

    rnaseq = bundle.get("rnaseq_proof") or {}
    disease = bundle.get("disease") or {}
    sample_counts = disease.get("validation_sample_counts") or {}
    ranking = bundle.get("ranking") or {}
    mapping = bundle.get("candidate_target_mapping") or {}
    protein = bundle.get("protein_structure_evidence") or {}
    audit = bundle.get("audit") or {}

    add(
        "rnaseq_evidence_verified",
        "pass" if rnaseq.get("verification_status") == "pass" and rnaseq.get("verification_failed") == 0 else "fail",
        {"verification_status": rnaseq.get("verification_status"), "verification_failed": rnaseq.get("verification_failed")},
    )
    add(
        "independent_counts_cohort_threshold",
        "pass" if sample_counts.get("n_samples", 0) >= 60 and sample_counts.get("min_class_n", 0) >= 25 else "fail",
        sample_counts,
    )
    add(
        "structure_readiness_verified",
        "pass" if readiness.get("status") == "ready" and readiness.get("parse_success_count", 0) >= 15 else "fail",
        {"status": readiness.get("status"), "parse_success_count": readiness.get("parse_success_count"), "artifact_count": readiness.get("artifact_count")},
    )
    add(
        "protein_structure_evidence_ready",
        "pass" if protein.get("status") == "ready" and protein.get("parsed_count", 0) >= 15 else "fail",
        {"status": protein.get("status"), "parsed_count": protein.get("parsed_count"), "protein_count": protein.get("protein_count")},
    )
    add(
        "candidate_target_mapping_ready",
        "pass" if mapping.get("status") == "ready" and mapping.get("mapped_output_candidate_count", 0) >= 8 else "fail",
        mapping,
    )
    add(
        "mapped_candidates_have_structure_coverage",
        "pass" if mapping.get("structure_covered_candidate_count", 0) >= mapping.get("mapped_output_candidate_count", 1) else "fail",
        {
            "mapped_output_candidate_count": mapping.get("mapped_output_candidate_count"),
            "structure_covered_candidate_count": mapping.get("structure_covered_candidate_count"),
        },
    )
    top_candidate = (ranking.get("candidates") or [{}])[0]
    top_targets = top_candidate.get("structure_targets") or {}
    add(
        "ranking_uses_kg_omics_structure_score",
        "pass" if ranking.get("score_column") == "kg_omics_structure_score" and ranking.get("candidate_count", 0) >= 25 else "fail",
        {"score_column": ranking.get("score_column"), "candidate_count": ranking.get("candidate_count")},
    )
    add(
        "top_candidate_has_3d_structures",
        "pass" if top_targets.get("parsed_structure_count", 0) >= 1 else "fail",
        {"compound": top_candidate.get("compound_name"), "structure_targets": top_targets},
    )
    add(
        "quantum_is_benchmark_not_advantage_claim",
        "pass" if rnaseq.get("external_quantum_adds_value") is False and _audit_check_status(audit, "no_quantum_advantage_claim") == "pass" else "fail",
        {"external_quantum_adds_value": rnaseq.get("external_quantum_adds_value"), "audit_status": _audit_check_status(audit, "no_quantum_advantage_claim")},
    )
    claim_policy = str(bundle.get("claim_policy") or "")
    add(
        "claim_policy_blocks_cures_and_clinical_claims",
        "pass" if "Do not claim cures" in claim_policy and "clinical efficacy" in claim_policy else "fail",
        claim_policy,
    )
    add(
        "frontend_exports_evidence_bundle",
        "pass" if "getRepurposingEvidenceBundleUrl" in page_text and "Export evidence JSON" in page_text and "Export report" in page_text else "fail",
        {"page": str(frontend_page_path)},
    )
    add(
        "frontend_renders_local_protein_viewer",
        "pass" if "ProteinStructureViewer" in page_text and "getStructureArtifactUrl" in api_text else "fail",
        {"page": str(frontend_page_path), "api": str(frontend_api_path)},
    )
    add(
        "frontend_production_build_verified",
        "pass" if current_build_id else "fail",
        {"build_id_path": str(build_id_path), "exists": build_id_path.exists(), "build_id": current_build_id},
    )
    live_assertions = live_smoke.get("assertions") or {}
    live_smoke_pass = (
        live_smoke.get("status") == "pass"
        and live_smoke.get("frontend_build_id") == current_build_id
        and live_assertions.get("page_contains_ranked_hypotheses") is True
        and live_assertions.get("page_contains_top_candidate") is True
        and live_assertions.get("page_contains_research_hypothesis_policy") is True
        and live_assertions.get("page_contains_openfold_policy") is True
        and live_assertions.get("api_candidate_count_at_least_25") is True
        and live_assertions.get("api_quantum_advantage_claim_blocked") is True
    )
    add(
        "frontend_live_http_render_verified",
        "pass" if live_smoke_pass else "fail",
        {
            "smoke_path": str(live_smoke_path),
            "exists": live_smoke_path.exists(),
            "status": live_smoke.get("status"),
            "frontend_build_id": live_smoke.get("frontend_build_id"),
            "current_build_id": current_build_id,
            "assertions": live_assertions,
        },
    )

    failures = [item for item in checks if item["required"] and item["status"] != "pass"]
    warnings = [item for item in checks if item["status"] == "warn"]
    result = {
        "schema_version": "1.0",
        "status": "ready_with_warnings" if not failures and warnings else "ready" if not failures else "not_ready",
        "required_check_count": sum(1 for item in checks if item["required"]),
        "failed_required_count": len(failures),
        "warning_count": len(warnings),
        "checks": checks,
        "claim_policy": "Research hypotheses only; not cures, prescriptions, clinical validation, or quantum advantage claims.",
        "inputs": {
            "bundle_path": str(bundle_path),
            "structure_readiness_path": str(structure_readiness_path),
            "frontend_page_path": str(frontend_page_path),
            "frontend_api_path": str(frontend_api_path),
            "frontend_build_id_path": str(frontend_build_id_path),
            "frontend_live_smoke_path": str(frontend_live_smoke_path),
        },
    }
    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "repurposing_workbench_acceptance_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        _write_markdown(result, out_path / "repurposing_workbench_acceptance_audit.md")
    return result


def _audit_check_status(audit: dict[str, Any], name: str) -> str | None:
    for check in audit.get("checks") or []:
        if check.get("check") == name:
            return check.get("status")
    return None


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# Repurposing Workbench Acceptance Audit",
        "",
        f"Status: `{result['status']}`",
        f"Required checks failed: {result['failed_required_count']}/{result['required_check_count']}",
        f"Warnings: {result['warning_count']}",
        "",
        "## Checks",
    ]
    for check in result["checks"]:
        marker = "PASS" if check["status"] == "pass" else "WARN" if check["status"] == "warn" else "FAIL"
        lines.append(f"- {marker}: `{check['check']}`")
    lines.extend(["", "## Claim Policy", result["claim_policy"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--structure-readiness", default=DEFAULT_READINESS)
    parser.add_argument("--frontend-page", default=DEFAULT_PAGE)
    parser.add_argument("--frontend-api", default=DEFAULT_API)
    parser.add_argument("--frontend-build-id", default=DEFAULT_BUILD_ID)
    parser.add_argument("--frontend-live-smoke", default=DEFAULT_FRONTEND_LIVE_SMOKE)
    parser.add_argument("--out-dir", default="artifacts/repurposing/brca_external_validation")
    args = parser.parse_args()
    result = build_repurposing_workbench_audit(
        bundle_path=args.bundle,
        structure_readiness_path=args.structure_readiness,
        frontend_page_path=args.frontend_page,
        frontend_api_path=args.frontend_api,
        frontend_build_id_path=args.frontend_build_id,
        frontend_live_smoke_path=args.frontend_live_smoke,
        out_dir=args.out_dir,
    )
    print(json.dumps({"status": result["status"], "failed_required_count": result["failed_required_count"], "warning_count": result["warning_count"]}, indent=2))
    return 0 if result["failed_required_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
