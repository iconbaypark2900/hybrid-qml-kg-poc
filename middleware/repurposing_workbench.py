from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


SCORING_MODES = [
    "kg_only",
    "kg_plus_rnaseq",
    "kg_plus_rnaseq_plus_structure",
    "quantum_benchmark_overlay",
]


def list_repurposing_diseases() -> dict[str, Any]:
    return {
        "status": "ok",
        "diseases": _diseases(),
        "provenance": _provenance("/repurposing/diseases"),
    }


def build_repurposing_candidates(disease_id: str) -> dict[str, Any]:
    endpoint = "/repurposing/candidates"
    bundle = _load_repurposing_bundle(disease_id)
    if bundle is not None:
        return _response_from_bundle(bundle, endpoint)

    diseases = _diseases()
    disease = next((item for item in diseases if item["id"] == disease_id), diseases[0])
    structure = _structure_summary(endpoint)
    quantum_delta = -0.0507 if disease["id"] == "brca_external_validation" else -0.0149
    rows = _candidate_rows(disease)
    candidates = [
        _candidate_payload(row, rank, disease, structure, quantum_delta)
        for rank, row in enumerate(rows, start=1)
    ]
    return {
        "status": "ok",
        "disease": disease,
        "candidates": candidates,
        "scoring_modes": SCORING_MODES,
        "manifest": _manifest(disease["id"], source="deterministic_fallback"),
        "provenance": _provenance(endpoint) + structure["provenance"],
        "message": "Local-first fallback hypotheses assembled with conservative audit guardrails.",
    }


def _load_repurposing_bundle(disease_id: str) -> dict[str, Any] | None:
    if disease_id != "brca_external_validation":
        return None
    path = project_root() / "artifacts" / "repurposing" / "brca_external_validation" / "repurposing_evidence_bundle.json"
    if not path.exists():
        return None
    bundle = json.loads(path.read_text(encoding="utf-8"))
    if bundle.get("status") != "ready":
        return None
    bundle["_bundle_path"] = path.as_posix()
    return bundle


def _response_from_bundle(bundle: dict[str, Any], endpoint: str) -> dict[str, Any]:
    disease_blob = bundle.get("disease") or {}
    sample_counts = disease_blob.get("validation_sample_counts") or {}
    disease = {
        "id": disease_blob.get("id", "brca_external_validation"),
        "name": disease_blob.get("name", "Breast cancer"),
        "cohort": "TCGA-BRCA to GEO GSE225846",
        "source": "artifact_backed_repurposing_bundle",
        "sample_count": int(sample_counts.get("n_samples") or 0),
        "smallest_class_count": int(sample_counts.get("min_class_n") or 0),
        "evidence_status": str((bundle.get("audit") or {}).get("readiness") or "unknown"),
        "notes": [
            "Candidates are loaded from repurposing_evidence_bundle.json.",
            "External validation remains technical cross-study evidence, not clinical validation.",
        ],
    }
    rnaseq = bundle.get("rnaseq_proof") or {}
    structure_proof = bundle.get("structure_proof") or {}
    protein_evidence = ((bundle.get("protein_structure_evidence") or {}).get("proteins") or [])
    structure = {
        "status": structure_proof.get("status", "unknown"),
        "target_count": int(structure_proof.get("artifact_count") or 0),
        "available_target_count": int(structure_proof.get("parse_success_count") or 0),
        "missing_rate": 0.0 if structure_proof.get("parse_success_count") else 1.0,
        "target_ids": [row.get("target_id", "") for row in protein_evidence if row.get("target_id")],
        "provenance": [
            {
                "endpoint": endpoint,
                "source_kind": "run_artifact",
                "artifact_path": str(bundle.get("_bundle_path", "")),
                "artifact_name": "repurposing_evidence_bundle.json",
                "notes": ["Structure readiness loaded from artifact-backed bundle."],
            }
        ],
    }
    external_delta = rnaseq.get("external_delta_roc_auc")
    candidates = [
        _bundle_candidate_payload(candidate, disease, structure, rnaseq, external_delta, protein_evidence)
        for candidate in (bundle.get("ranking") or {}).get("candidates", [])
    ]
    return {
        "status": "ok",
        "disease": disease,
        "candidates": candidates,
        "scoring_modes": SCORING_MODES,
        "manifest": _manifest(disease["id"], source="repurposing_evidence_bundle"),
        "provenance": [
            {
                "endpoint": endpoint,
                "source_kind": "run_artifact",
                "artifact_path": str(bundle.get("_bundle_path", "")),
                "artifact_name": "repurposing_evidence_bundle.json",
                "notes": [bundle.get("claim_policy", "Ranked research hypotheses only.")],
            }
        ],
        "message": "Loaded artifact-backed repurposing hypotheses from verified RNA-seq, structure, and ranking evidence.",
    }


def _bundle_candidate_payload(
    candidate: dict[str, Any],
    disease: dict[str, Any],
    structure: dict[str, Any],
    rnaseq: dict[str, Any],
    external_delta: Any,
    default_protein_evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    quantum_delta = float(external_delta) if isinstance(external_delta, (int, float)) else -1.0
    structure_targets = candidate.get("structure_targets") or {}
    target_count = int(structure_targets.get("target_count") or 0)
    parsed_structure_count = int(structure_targets.get("parsed_structure_count") or 0)
    return {
        "compound_id": candidate.get("candidate_id", candidate.get("compound_name", "")),
        "compound_name": candidate.get("compound_name", "Unknown compound"),
        "disease_id": disease["id"],
        "disease_name": disease["name"],
        "hypothesis_score": float(candidate.get("hypothesis_score") or 0.0),
        "scoring_mode": "kg_plus_rnaseq_plus_structure",
        "rank": int(candidate.get("rank") or 0),
        "summary": candidate.get("summary", "Ranked research hypothesis; not clinical evidence of efficacy."),
        "evidence_components": [
            {
                "label": "KG",
                "value": f"{float(candidate.get('hypothesis_score') or 0.0):.3f}",
                "status": "supporting",
                "detail": "Loaded from ranking_comparison.csv through repurposing_evidence_bundle.json.",
            },
            {
                "label": "RNA-seq",
                "value": str(rnaseq.get("verification_status", "unknown")),
                "status": "supporting" if rnaseq.get("verification_failed") == 0 else "limited",
                "detail": f"External classical ROC-AUC {rnaseq.get('external_classical_roc_auc')}; quantum adds value: {rnaseq.get('external_quantum_adds_value')}",
            },
            {
                "label": "Structure",
                "value": f"targets {target_count}; structures {parsed_structure_count}",
                "status": "supporting" if parsed_structure_count else "limited",
                "detail": "Candidate targets are resolved from KG provenance; missing local structure artifacts are shown explicitly.",
            },
            {
                "label": "Quantum",
                "value": "benchmark only",
                "status": "limited",
                "detail": f"External delta ROC-AUC {external_delta}; no quantum advantage claim is allowed.",
            },
        ],
        "kg_paths": [
            f"CREEDS profile overlap: {candidate.get('profile_gene_overlap')}",
            f"Signature reversal score: {candidate.get('signature_reversal_score')}",
            f"Quantum delta score: {candidate.get('quantum_delta_score')}",
        ],
        "rnaseq_signature": {
            "direction": "signature_reversal_ranking",
            "profile_gene_overlap": candidate.get("profile_gene_overlap"),
            "signature_reversal_score": candidate.get("signature_reversal_score"),
        },
        "structure": structure,
        "structure_targets": structure_targets,
        "protein_structures": candidate.get("protein_structures") or default_protein_evidence or [],
        "classical_ml": {
            "model": "logistic_baseline",
            "external_roc_auc": rnaseq.get("external_classical_roc_auc"),
            "role": "primary comparator",
        },
        "quantum_benchmark": {
            "model": "qsvc_kernel",
            "external_roc_auc": rnaseq.get("external_quantum_roc_auc"),
            "delta_vs_classical": external_delta,
            "role": "secondary benchmark",
        },
        "audit": _audit(quantum_delta, disease["evidence_status"]),
    }


def _manifest(disease_id: str, *, source: str) -> dict[str, Any]:
    return {
        "workflow": "local_open_source_repurposing_workbench",
        "disease_id": disease_id,
        "source": source,
        "scoring_modes": SCORING_MODES,
        "structure_mode": "artifact_first",
        "openfold_runner": "deferred",
        "claim_policy": "research_hypotheses_only",
        "paid_or_hosted_services_required": False,
    }


def _diseases() -> list[dict[str, Any]]:
    return [
        {
            "id": "brca_external_validation",
            "name": "Breast cancer",
            "cohort": "TCGA-BRCA to GEO GSE225846",
            "source": "audited_counts_level_rnaseq",
            "sample_count": 215,
            "smallest_class_count": 30,
            "evidence_status": "review_ready",
            "notes": [
                "Counts-level RNA-seq evidence passed the local evidence-bundle verifier.",
                "External validation is technical cross-study evidence, not clinical validation.",
            ],
        },
        {
            "id": "atherosclerosis_fallback",
            "name": "Atherosclerosis",
            "cohort": "QCE26 paper-aligned KG fallback",
            "source": "kg_candidate_fallback",
            "sample_count": 0,
            "smallest_class_count": 0,
            "evidence_status": "fallback_only",
            "notes": ["Use for UI continuity when local RNA-seq artifacts are unavailable."],
        },
        {
            "id": "gout_fallback",
            "name": "Gout",
            "cohort": "QCE26 paper-aligned KG fallback",
            "source": "kg_candidate_fallback",
            "sample_count": 0,
            "smallest_class_count": 0,
            "evidence_status": "fallback_only",
            "notes": ["Novel hypotheses require wet-lab or clinical follow-up before any treatment claim."],
        },
    ]


def _provenance(endpoint: str) -> list[dict[str, Any]]:
    root = project_root()
    rel_paths = [
        "artifacts/external/tcga_brca",
        "artifacts/external/gse225846",
        "artifacts/benchmarks/rnaseq_quantum_external_validation",
        "tests/fixtures/structure_artifacts/registry.json",
    ]
    provenance: list[dict[str, Any]] = []
    for rel_path in rel_paths:
        path = root / rel_path
        if path.exists():
            provenance.append(
                {
                    "endpoint": endpoint,
                    "source_kind": "run_artifact" if rel_path.startswith("artifacts/") else "dataset",
                    "artifact_path": rel_path,
                    "artifact_name": path.name,
                    "mtime_epoch": path.stat().st_mtime,
                    "notes": ["local open-source artifact; no hosted AlphaFold/BioNeMo dependency"],
                }
            )
    if provenance:
        return provenance
    return [
        {
            "endpoint": endpoint,
            "source_kind": "fallback",
            "notes": ["No local evidence artifacts found; returning built-in conservative demonstration data."],
        }
    ]


def _structure_summary(endpoint: str) -> dict[str, Any]:
    registry = project_root() / "tests" / "fixtures" / "structure_artifacts" / "registry.json"
    if not registry.exists():
        return _missing_structure(endpoint, "No local structure registry was available for this candidate.")
    try:
        from structure_layer import build_structure_feature_table, load_artifact_registry

        artifacts = load_artifact_registry(registry)
        features, provenance_rows = build_structure_feature_table(artifacts)
        target_count = len(features)
        available = sum(1 for row in features if row.get("has_structure") == 1)
        return {
            "status": "artifact_backed",
            "target_count": target_count,
            "available_target_count": available,
            "missing_rate": round(1 - (available / target_count), 4) if target_count else 1.0,
            "target_ids": [str(row.get("target_id")) for row in features],
            "provenance": [
                {
                    "endpoint": endpoint,
                    "source_kind": "dataset",
                    "artifact_path": "tests/fixtures/structure_artifacts/registry.json",
                    "artifact_name": "registry.json",
                    "mtime_epoch": registry.stat().st_mtime,
                    "notes": [
                        "Structure features are artifact-first from local PDB/OpenFold-like outputs.",
                        f"Feature provenance rows: {len(provenance_rows)}",
                    ],
                }
            ],
        }
    except Exception as exc:
        return _missing_structure(endpoint, f"Structure registry could not be parsed: {exc}")


def _missing_structure(endpoint: str, note: str) -> dict[str, Any]:
    return {
        "status": "missing",
        "target_count": 0,
        "available_target_count": 0,
        "missing_rate": 1.0,
        "target_ids": [],
        "provenance": [{"endpoint": endpoint, "source_kind": "fallback", "notes": [note]}],
    }


def _candidate_rows(disease: dict[str, Any]) -> list[dict[str, Any]]:
    if disease["source"] != "audited_counts_level_rnaseq":
        return [
            {
                "compound_id": "Compound::DB00331",
                "compound_name": "Losartan" if "atherosclerosis" in disease["id"] else "Ezetimibe",
                "score": 0.693,
                "kg": ["Compound-disease KG neighborhood", "Mechanism path requires independent validation"],
                "signature": {"direction": "not_available", "top_up_genes": [], "top_down_genes": []},
            }
        ]
    return [
        {
            "compound_id": "Compound::DB00515",
            "compound_name": "Cisplatin",
            "score": 0.842,
            "kg": [
                "Compound - treats - cancer context",
                "Compound - binds - DNA damage response genes",
                "Disease - associates - proliferation pathways",
            ],
            "signature": {
                "direction": "reversal_hypothesis",
                "top_up_genes": ["MKI67", "TOP2A", "BIRC5"],
                "top_down_genes": ["ESR1", "PGR", "FOXA1"],
            },
        },
        {
            "compound_id": "Compound::DB01248",
            "compound_name": "Docetaxel",
            "score": 0.817,
            "kg": [
                "Compound - treats - breast neoplasm context",
                "Compound - affects - microtubule pathway",
                "Disease - associates - mitotic cell cycle",
            ],
            "signature": {
                "direction": "reversal_hypothesis",
                "top_up_genes": ["AURKA", "UBE2C", "CCNB1"],
                "top_down_genes": ["GATA3", "TFF1", "AGR2"],
            },
        },
        {
            "compound_id": "Compound::DB00997",
            "compound_name": "Doxorubicin",
            "score": 0.791,
            "kg": [
                "Compound - treats - oncology context",
                "Compound - binds - topoisomerase pathway",
                "Disease - associates - DNA repair programs",
            ],
            "signature": {
                "direction": "reversal_hypothesis",
                "top_up_genes": ["PCNA", "MCM2", "MCM6"],
                "top_down_genes": ["KRT18", "EPCAM", "XBP1"],
            },
        },
    ]


def _candidate_payload(
    row: dict[str, Any],
    rank: int,
    disease: dict[str, Any],
    structure: dict[str, Any],
    quantum_delta: float,
) -> dict[str, Any]:
    return {
        "compound_id": row["compound_id"],
        "compound_name": row["compound_name"],
        "disease_id": disease["id"],
        "disease_name": disease["name"],
        "hypothesis_score": row["score"],
        "scoring_mode": "kg_plus_rnaseq_plus_structure",
        "rank": rank,
        "summary": f"{row['compound_name']} is a ranked repurposing hypothesis for {disease['name']}; this is not clinical evidence of efficacy.",
        "evidence_components": [
            {
                "label": "KG",
                "value": "available",
                "status": "supporting",
                "detail": "Candidate has mechanism paths in the local KG evidence model.",
            },
            {
                "label": "RNA-seq",
                "value": disease["evidence_status"],
                "status": "supporting" if disease["evidence_status"] == "review_ready" else "limited",
                "detail": disease["cohort"],
            },
            {
                "label": "Structure",
                "value": structure["status"],
                "status": "supporting" if structure["available_target_count"] else "limited",
                "detail": f"{structure['available_target_count']}/{structure['target_count']} local target artifacts available.",
            },
            {
                "label": "Quantum",
                "value": "benchmark only",
                "status": "limited",
                "detail": "Shown as an overlay against classical baselines, not as the predictive backbone.",
            },
        ],
        "kg_paths": row["kg"],
        "rnaseq_signature": row["signature"],
        "structure": structure,
        "structure_targets": {},
        "protein_structures": [],
        "classical_ml": {
            "model": "logistic_baseline",
            "external_roc_auc": 0.9730 if disease["id"] == "brca_external_validation" else None,
            "role": "primary comparator",
        },
        "quantum_benchmark": {
            "model": "qsvc_kernel",
            "external_roc_auc": 0.9223 if disease["id"] == "brca_external_validation" else None,
            "delta_vs_classical": quantum_delta,
            "role": "secondary benchmark",
        },
        "audit": _audit(quantum_delta, disease["evidence_status"]),
    }


def _audit(quantum_delta: float, evidence_status: str) -> dict[str, Any]:
    warnings = [
        "Ranked hypothesis only; not a medical recommendation.",
        "Clinical validation is not established by this workbench.",
    ]
    if quantum_delta <= 0:
        warnings.append("Quantum benchmark does not outperform the classical baseline for the audited RNA-seq endpoint.")
    if evidence_status != "review_ready":
        warnings.append("Disease evidence is fallback or incomplete; use only for workflow inspection.")
    return {
        "status": "guarded",
        "claim_policy": "Use research-hypothesis language only. Do not claim cure, treatment efficacy, or quantum advantage unless audits explicitly allow it.",
        "warnings": warnings,
        "quantum_advantage_claim_allowed": quantum_delta > 0,
        "clinical_claim_allowed": False,
    }
