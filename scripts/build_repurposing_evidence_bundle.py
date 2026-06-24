#!/usr/bin/env python3
"""Build an artifact-driven drug-repurposing evidence bundle.

The bundle is the backend handoff between proven RNA-seq evidence, KG/omics
ranking artifacts, candidate-target KG provenance, structure readiness, and
later UI/ranking surfaces.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_openfold_readiness import verify_openfold_readiness
from structure_layer import (
    build_protein_structure_evidence,
    resolve_candidate_protein_structure_evidence,
)

STRUCTURE_SCORE_WEIGHT = 0.02


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_repurposing_evidence_bundle(
    *,
    evidence_verification: str | Path,
    external_validation: str | Path,
    benchmark_verdict: str | Path,
    evidence_audit: str | Path,
    ranking_comparison: str | Path,
    structure_registry: str | Path,
    out_dir: str | Path,
    candidate_target_map: str | Path | None = None,
    disease_id: str = "brca_external_validation",
    disease_name: str = "Breast cancer",
    top_k: int = 25,
) -> dict[str, Any]:
    verification = read_json(evidence_verification)
    external = read_json(external_validation)
    verdict = read_json(benchmark_verdict)
    audit = read_json(evidence_audit)
    ranking = pd.read_csv(ranking_comparison)
    structure = verify_openfold_readiness(registry=structure_registry)
    protein_evidence = build_protein_structure_evidence(structure_registry)
    parsed_proteins = [row for row in protein_evidence if row.get("parse_success")]
    target_map, target_map_summary = read_candidate_target_map(candidate_target_map)

    required_ranking_cols = {
        "candidate_id",
        "compound",
        "profile_gene_overlap",
        "signature_reversal_score",
        "kg_omics_final_score",
        "kg_omics_quantum_final_score",
        "quantum_delta_score",
    }
    missing = sorted(required_ranking_cols - set(ranking.columns))
    if missing:
        raise ValueError(f"ranking_comparison missing required columns: {missing}")

    external_verdict = external.get("verdict") or {}
    candidates = []
    for _, row in ranking.iterrows():
        candidate_id = str(row["candidate_id"])
        target_record = target_map.get(candidate_id)
        target_ids = parse_target_ids((target_record or {}).get("target_ids"))
        candidate_proteins = (
            resolve_candidate_protein_structure_evidence(protein_evidence, target_ids)
            if target_ids
            else []
        )
        structure_score = structure_feature_score(candidate_proteins, target_ids)
        kg_omics_score = float(row["kg_omics_final_score"])
        kg_omics_structure_score = kg_omics_score + (STRUCTURE_SCORE_WEIGHT * structure_score)
        parsed_count = sum(1 for item in candidate_proteins if item.get("parse_success"))
        available_count = sum(1 for item in candidate_proteins if item.get("artifact_available"))
        missing_target_ids = [item.get("target_id") for item in candidate_proteins if not item.get("artifact_available")]
        candidates.append(
            {
                "rank": 0,
                "candidate_id": candidate_id,
                "compound_name": str(row["compound"]),
                "disease_id": disease_id,
                "disease_name": disease_name,
                "hypothesis_score": kg_omics_structure_score,
                "kg_omics_score": kg_omics_score,
                "kg_omics_structure_score": kg_omics_structure_score,
                "structure_feature_score": structure_score,
                "kg_omics_quantum_score": float(row["kg_omics_quantum_final_score"]),
                "quantum_delta_score": float(row["quantum_delta_score"]),
                "profile_gene_overlap": int(row["profile_gene_overlap"]),
                "signature_reversal_score": float(row["signature_reversal_score"]),
                "quantum_status": str(row.get("quantum_status", "unknown")),
                "cell_type": None if pd.isna(row.get("cell_type")) else str(row.get("cell_type")),
                "geo_id": None if pd.isna(row.get("geo_id")) else str(row.get("geo_id")),
                "summary": (
                    f"{row['compound']} is a ranked repurposing hypothesis from real KG+omics ranking artifacts; "
                    "this is not clinical evidence of efficacy."
                ),
                "structure_targets": {
                    "mapping_status": (target_record or {}).get("mapping_status", "target_map_missing" if target_map_summary["status"] != "not_provided" else "not_provided"),
                    "target_source": (target_record or {}).get("target_source", target_map_summary.get("target_source")),
                    "compound_kg_id": (target_record or {}).get("compound_kg_id"),
                    "disease_kg_id": (target_record or {}).get("disease_kg_id"),
                    "target_ids": target_ids,
                    "target_count": len(target_ids),
                    "structure_artifact_target_count": available_count,
                    "parsed_structure_count": parsed_count,
                    "missing_structure_target_ids": missing_target_ids,
                    "structure_feature_score": structure_score,
                    "notes": (target_record or {}).get("notes", ""),
                },
                "protein_structures": candidate_proteins,
            }
        )

    candidates.sort(key=lambda item: item["kg_omics_structure_score"], reverse=True)
    candidates = candidates[:top_k]
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank

    mapped_output_candidate_count = sum(1 for item in candidates if item["structure_targets"]["target_count"] > 0)
    structure_covered_candidate_count = sum(
        1 for item in candidates if item["structure_targets"]["structure_artifact_target_count"] > 0
    )
    total_mapped_targets = sum(item["structure_targets"]["target_count"] for item in candidates)
    total_structure_artifact_targets = sum(
        item["structure_targets"]["structure_artifact_target_count"] for item in candidates
    )

    checks = [
        {
            "check": "rnaseq_evidence_bundle_verified",
            "status": "pass" if verification.get("status") == "pass" and verification.get("n_failed") == 0 else "fail",
            "evidence": {"status": verification.get("status"), "n_checks": verification.get("n_checks"), "n_failed": verification.get("n_failed")},
        },
        {
            "check": "external_validation_independent",
            "status": "pass" if external.get("independent_cohorts") is True else "fail",
            "evidence": external.get("cohorts"),
        },
        {
            "check": "external_sample_threshold",
            "status": "pass" if (external.get("validation_sample_counts") or {}).get("n_samples", 0) >= 60 and (external.get("validation_sample_counts") or {}).get("min_class_n", 0) >= 25 else "fail",
            "evidence": external.get("validation_sample_counts"),
        },
        {
            "check": "audit_review_ready",
            "status": "pass" if audit.get("readiness") == "review_ready" and audit.get("worst_gate_status") == "pass" else "fail",
            "evidence": {"readiness": audit.get("readiness"), "worst_gate_status": audit.get("worst_gate_status")},
        },
        {
            "check": "ranking_uses_real_profiles",
            "status": "pass" if len(candidates) > 0 and max((int(item["profile_gene_overlap"]) for item in candidates), default=0) > 0 else "fail",
            "evidence": {"candidate_count": int(len(candidates)), "max_profile_gene_overlap": max((int(item["profile_gene_overlap"]) for item in candidates), default=0)},
        },
        {
            "check": "candidate_target_map_available",
            "status": "pass" if target_map_summary.get("status") == "ready" else "fail",
            "evidence": target_map_summary,
        },
        {
            "check": "candidate_targets_resolved",
            "status": "pass" if mapped_output_candidate_count > 0 else "fail",
            "evidence": {"mapped_output_candidate_count": mapped_output_candidate_count, "candidate_count": len(candidates)},
        },
        {
            "check": "structure_artifacts_ready",
            "status": "pass" if structure.get("status") == "ready" else "fail",
            "evidence": {"status": structure.get("status"), "parse_success_count": structure.get("parse_success_count")},
        },
        {
            "check": "protein_structure_evidence_ready",
            "status": "pass" if parsed_proteins else "fail",
            "evidence": {"protein_count": len(protein_evidence), "parsed_count": len(parsed_proteins)},
        },
        {
            "check": "candidate_structure_artifact_coverage",
            "status": "pass" if structure_covered_candidate_count > 0 else "warn",
            "evidence": {
                "structure_covered_candidate_count": structure_covered_candidate_count,
                "mapped_output_candidate_count": mapped_output_candidate_count,
                "total_mapped_targets": total_mapped_targets,
                "total_structure_artifact_targets": total_structure_artifact_targets,
            },
        },
        {
            "check": "no_quantum_advantage_claim",
            "status": "pass" if verdict.get("quantum_adds_value") is False and external_verdict.get("external_quantum_adds_value") is False else "fail",
            "evidence": {
                "development_quantum_adds_value": verdict.get("quantum_adds_value"),
                "external_quantum_adds_value": external_verdict.get("external_quantum_adds_value"),
            },
        },
    ]
    failed = [check for check in checks if check["status"] == "fail"]
    warnings = [check for check in checks if check["status"] == "warn"]
    bundle = {
        "schema_version": "1.1",
        "status": "ready" if not failed else "not_ready",
        "disease": {
            "id": disease_id,
            "name": disease_name,
            "cohorts": external.get("cohorts"),
            "validation_sample_counts": external.get("validation_sample_counts"),
        },
        "rnaseq_proof": {
            "verification_status": verification.get("status"),
            "verification_checks": verification.get("n_checks"),
            "verification_failed": verification.get("n_failed"),
            "external_classical_roc_auc": external_verdict.get("external_classical_roc_auc"),
            "external_quantum_roc_auc": external_verdict.get("external_quantum_roc_auc"),
            "external_delta_roc_auc": external_verdict.get("external_delta_roc_auc"),
            "external_quantum_adds_value": external_verdict.get("external_quantum_adds_value"),
        },
        "structure_proof": structure,
        "protein_structure_evidence": {
            "status": "ready" if parsed_proteins else "not_ready",
            "protein_count": len(protein_evidence),
            "parsed_count": len(parsed_proteins),
            "proteins": protein_evidence,
        },
        "candidate_target_mapping": {
            **target_map_summary,
            "mapped_output_candidate_count": mapped_output_candidate_count,
            "structure_covered_candidate_count": structure_covered_candidate_count,
            "total_mapped_targets": total_mapped_targets,
            "total_structure_artifact_targets": total_structure_artifact_targets,
        },
        "ranking": {
            "source_path": str(ranking_comparison),
            "candidate_count": int(len(candidates)),
            "score_column": "kg_omics_structure_score",
            "base_score_column": "kg_omics_final_score",
            "structure_score_weight": STRUCTURE_SCORE_WEIGHT,
            "quantum_overlay_column": "kg_omics_quantum_final_score",
            "candidates": candidates,
        },
        "audit": {
            "readiness": audit.get("readiness"),
            "worst_gate_status": audit.get("worst_gate_status"),
            "claim_guidance": audit.get("claim_guidance"),
            "checks": checks,
            "warning_count": len(warnings),
        },
        "claim_policy": "Ranked research hypotheses only. Do not claim cures, clinical efficacy, or quantum advantage from this bundle.",
        "inputs": {
            "evidence_verification": str(evidence_verification),
            "external_validation": str(external_validation),
            "benchmark_verdict": str(benchmark_verdict),
            "evidence_audit": str(evidence_audit),
            "ranking_comparison": str(ranking_comparison),
            "structure_registry": str(structure_registry),
            "candidate_target_map": str(candidate_target_map) if candidate_target_map else None,
        },
    }
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "repurposing_evidence_bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    _write_report(bundle, out_path / "repurposing_evidence_bundle.md")
    return bundle


def read_candidate_target_map(path: str | Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path:
        return {}, {"status": "not_provided", "path": None, "candidate_count": 0, "mapped_candidate_count": 0}
    map_path = Path(path)
    if not map_path.exists():
        return {}, {"status": "not_provided", "path": str(map_path), "candidate_count": 0, "mapped_candidate_count": 0}
    df = pd.read_csv(map_path).fillna("")
    required = {"candidate_id", "mapping_status", "target_ids", "target_count"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"candidate_target_map missing required columns: {missing}")
    records: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        record = row.to_dict()
        record["candidate_id"] = str(record["candidate_id"])
        record["target_ids"] = parse_target_ids(record.get("target_ids"))
        record["target_count"] = len(record["target_ids"])
        records[record["candidate_id"]] = record
    mapped = sum(1 for record in records.values() if record.get("target_count", 0) > 0)
    return records, {
        "status": "ready" if mapped else "not_ready",
        "path": str(map_path),
        "candidate_count": len(records),
        "mapped_candidate_count": mapped,
        "target_source": df["target_source"].iloc[0] if "target_source" in df.columns and len(df) else None,
    }


def parse_target_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item for item in str(value).split("|") if item]


def structure_feature_score(candidate_proteins: list[dict[str, Any]], target_ids: list[str]) -> float:
    if not target_ids:
        return 0.0
    parsed = [item for item in candidate_proteins if item.get("parse_success")]
    if not parsed:
        return 0.0
    confidences = []
    for item in parsed:
        mean_plddt = (item.get("confidence") or {}).get("mean_plddt")
        if isinstance(mean_plddt, (int, float)):
            confidences.append(max(0.0, min(1.0, float(mean_plddt) / 100.0)))
        else:
            confidences.append(0.5)
    confidence_score = sum(confidences) / len(confidences) if confidences else 0.0
    coverage = len(parsed) / len(target_ids)
    return coverage * confidence_score


def _write_report(bundle: dict[str, Any], path: Path) -> None:
    mapping = bundle.get("candidate_target_mapping", {})
    lines = [
        "# Repurposing Evidence Bundle",
        "",
        f"Status: `{bundle['status']}`",
        f"Disease: {bundle['disease']['name']}",
        "",
        "## RNA-seq proof",
        f"- Verification checks: {bundle['rnaseq_proof']['verification_checks']} checks, {bundle['rnaseq_proof']['verification_failed']} failed",
        f"- External classical ROC-AUC: {bundle['rnaseq_proof']['external_classical_roc_auc']}",
        f"- External quantum ROC-AUC: {bundle['rnaseq_proof']['external_quantum_roc_auc']}",
        f"- External delta ROC-AUC: {bundle['rnaseq_proof']['external_delta_roc_auc']}",
        "",
        "## Candidate target mapping",
        f"- Mapping status: {mapping.get('status')}",
        f"- Mapped output candidates: {mapping.get('mapped_output_candidate_count')}",
        f"- Structure-covered candidates: {mapping.get('structure_covered_candidate_count')}",
        "",
        "## Protein structure evidence",
        f"- Registry proteins: {bundle.get('protein_structure_evidence', {}).get('protein_count')}",
        f"- Parsed registry structures: {bundle.get('protein_structure_evidence', {}).get('parsed_count')}",
        "",
        "## Ranking",
        f"- Candidates: {bundle['ranking']['candidate_count']}",
        f"- Score column: `{bundle['ranking']['score_column']}`",
        f"- Source: `{bundle['ranking']['source_path']}`",
        "",
        "## Claim policy",
        bundle["claim_policy"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-verification", default="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/evidence_bundle_verification.json")
    parser.add_argument("--external-validation", default="artifacts/benchmarks/rnaseq_quantum_tcga_brca_gse225846_external/external_validation.json")
    parser.add_argument("--benchmark-verdict", default="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/quantum_value_verdict.json")
    parser.add_argument("--evidence-audit", default="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/evidence_audit.json")
    parser.add_argument("--ranking-comparison", default="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/ranking_comparison.csv")
    parser.add_argument("--structure-registry", default="artifacts/structures/alphafold/brca_anastrozole_targets/structure_registry.json")
    parser.add_argument("--candidate-target-map", default="artifacts/repurposing/brca_external_validation/candidate_target_map.csv")
    parser.add_argument("--out-dir", default="artifacts/repurposing/brca_external_validation")
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    bundle = build_repurposing_evidence_bundle(
        evidence_verification=args.evidence_verification,
        external_validation=args.external_validation,
        benchmark_verdict=args.benchmark_verdict,
        evidence_audit=args.evidence_audit,
        ranking_comparison=args.ranking_comparison,
        structure_registry=args.structure_registry,
        candidate_target_map=args.candidate_target_map,
        out_dir=args.out_dir,
        top_k=args.top_k,
    )
    print(json.dumps({"status": bundle["status"], "out_dir": args.out_dir, "candidate_count": bundle["ranking"]["candidate_count"], "mapped_output_candidate_count": bundle["candidate_target_mapping"]["mapped_output_candidate_count"]}, indent=2))
    return 0 if bundle["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
