#!/usr/bin/env python3
"""Audit RNA-seq quantum benchmark evidence for review readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _gate(gate_id: str, status: str, message: str, evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "id": gate_id,
        "status": status,
        "message": message,
        "evidence": evidence or {},
    }


def _status_rank(status: str) -> int:
    return {"pass": 0, "warn": 1, "fail": 2}.get(status, 2)


def _worst_status(gates: List[Dict[str, Any]]) -> str:
    if not gates:
        return "fail"
    return max((str(g["status"]) for g in gates), key=_status_rank)


def audit_verdict(
    verdict: Dict[str, Any],
    *,
    min_publishable_samples: int,
    min_publishable_class_n: int,
    min_screening_samples: int,
    min_screening_class_n: int,
    min_ranking_candidates: int,
    min_full_permutations: int,
    alpha: float,
    external_validation: Optional[Dict[str, Any]] = None,
    min_external_samples: int = 60,
    min_external_class_n: int = 25,
    min_external_bootstrap: int = 1000,
    min_external_permutations: int = 1000,
) -> Dict[str, Any]:
    gates: List[Dict[str, Any]] = []

    n_samples = int(verdict.get("classifier_n_samples") or 0)
    min_class_n = int(verdict.get("classifier_min_class_n") or 0)
    if n_samples >= min_publishable_samples and min_class_n >= min_publishable_class_n:
        sample_status = "pass"
        sample_msg = "Classifier sample size meets the configured review threshold."
    elif n_samples >= min_screening_samples and min_class_n >= min_screening_class_n:
        sample_status = "warn"
        sample_msg = "Classifier sample size is screening-grade and below the configured review threshold."
    else:
        sample_status = "fail"
        sample_msg = "Classifier sample size is pilot-only and underpowered."
    gates.append(
        _gate(
            "classifier_sample_size",
            sample_status,
            sample_msg,
            {
                "n_samples": n_samples,
                "min_class_n": min_class_n,
                "review_threshold": {
                    "n_samples": min_publishable_samples,
                    "min_class_n": min_publishable_class_n,
                },
                "screening_threshold": {
                    "n_samples": min_screening_samples,
                    "min_class_n": min_screening_class_n,
                },
            },
        )
    )

    full_perm = verdict.get("full_retraining_permutation") or {}
    full_available = bool(full_perm.get("available"))
    full_n = int(full_perm.get("n_valid") or 0)
    if full_available and full_n >= min_full_permutations:
        status = "pass"
        msg = "Full retraining permutation null is available with enough valid permutations."
    elif full_available:
        status = "warn"
        msg = "Full retraining permutation null is available but has fewer permutations than requested."
    else:
        status = "fail"
        msg = "Full retraining permutation null is missing."
    gates.append(
        _gate(
            "full_retraining_permutation",
            status,
            msg,
            {
                "available": full_available,
                "n_valid": full_n,
                "required_valid": min_full_permutations,
            },
        )
    )

    quantum_adds_value = bool(verdict.get("quantum_adds_value"))
    delta_auc = verdict.get("delta_roc_auc")
    delta_ci_low = (verdict.get("bootstrap_delta_roc_auc") or {}).get("delta_ci95_low")
    delta_p = full_perm.get("p_value_delta_auc_ge_observed")
    if quantum_adds_value:
        supported = (
            delta_auc is not None
            and float(delta_auc) > 0
            and delta_ci_low is not None
            and float(delta_ci_low) > 0
            and delta_p is not None
            and float(delta_p) <= alpha
        )
        gates.append(
            _gate(
                "quantum_classifier_value_claim",
                "pass" if supported else "fail",
                (
                    "Quantum classifier value claim is supported by delta, bootstrap CI, and full permutation p-value."
                    if supported
                    else "Quantum classifier value claim is not supported by the configured statistical gates."
                ),
                {"delta_roc_auc": delta_auc, "bootstrap_delta_ci95_low": delta_ci_low, "full_permutation_delta_p": delta_p},
            )
        )
    else:
        gates.append(
            _gate(
                "quantum_classifier_value_claim",
                "pass",
                "No positive quantum classifier value claim is made.",
                {"classifier_verdict": verdict.get("classifier_verdict"), "delta_roc_auc": delta_auc},
            )
        )

    ranking_real = bool(verdict.get("ranking_is_real_evidence"))
    ranking_candidates = int(float(verdict.get("ranking_candidate_count") or 0))
    if ranking_real and ranking_candidates >= min_ranking_candidates:
        status = "pass"
        msg = "Ranking uses real perturbation evidence with enough candidates."
    elif ranking_real:
        status = "warn"
        msg = "Ranking uses real perturbation evidence but candidate count is low."
    else:
        status = "fail"
        msg = "Ranking evidence is not real perturbation evidence."
    gates.append(
        _gate(
            "ranking_evidence",
            status,
            msg,
            {
                "ranking_is_real_evidence": ranking_real,
                "ranking_evidence_level": verdict.get("ranking_evidence_level"),
                "ranking_candidate_count": ranking_candidates,
                "required_candidates": min_ranking_candidates,
            },
        )
    )

    materiality = str(verdict.get("ranking_quantum_materiality") or "not_evaluable")
    if materiality == "material":
        status = "pass"
        msg = "Quantum scores materially change the KG+omics ranking."
    elif materiality in {"minor", "negligible"}:
        status = "pass" if not quantum_adds_value else "warn"
        msg = (
            "Quantum scores do not materially alter ranking, consistent with the absence of a positive quantum claim."
            if not quantum_adds_value
            else "Quantum scores do not materially improve or alter the ranking despite a positive quantum claim."
        )
    else:
        status = "fail"
        msg = "Quantum ranking materiality is not evaluable."
    gates.append(
        _gate(
            "ranking_quantum_materiality",
            status,
            msg,
            {
                "ranking_quantum_materiality": materiality,
                "ranking_spearman": verdict.get("ranking_spearman_kg_omics_vs_quantum"),
                "top_10_overlap": verdict.get("ranking_top_10_overlap_fraction"),
            },
        )
    )

    external = external_validation or {}
    external_counts = external.get("validation_sample_counts") or {}
    external_leakage = external.get("leakage_controls") or {}
    external_verdict = external.get("verdict") or {}
    external_n = int(external_counts.get("n_samples") or 0)
    external_min_class = int(external_counts.get("min_class_n") or 0)
    independent = bool(external.get("independent_cohorts"))
    leakage_ok = bool(external_leakage) and all(
        [
            not bool(external_leakage.get("validation_labels_used_for_feature_selection")),
            not bool(external_leakage.get("validation_labels_used_for_hyperparameter_selection")),
            not bool(external_leakage.get("validation_samples_used_for_scaler_or_pca_fit")),
            not bool(external_leakage.get("validation_samples_used_for_model_fit")),
            bool(external_leakage.get("development_configuration_locked_before_validation")),
            not bool(external_leakage.get("validation_de_signature_generated")),
            bool(external_leakage.get("shared_gene_universe_normalization_verified")),
        ]
    )
    external_size_ok = external_n >= min_external_samples and external_min_class >= min_external_class_n
    external_design_ok = independent and leakage_ok and external_size_ok
    gates.append(
        _gate(
            "independent_external_validation",
            "pass" if external_design_ok else "fail",
            (
                "Independent external validation meets cohort-size and leakage-control requirements."
                if external_design_ok
                else "Independent external validation is missing or fails cohort-size/leakage-control requirements."
            ),
            {
                "available": bool(external),
                "independent_cohorts": independent,
                "n_samples": external_n,
                "min_class_n": external_min_class,
                "required_n_samples": min_external_samples,
                "required_min_class_n": min_external_class_n,
                "leakage_controls_pass": leakage_ok,
            },
        )
    )

    external_bootstrap = external_verdict.get("patient_cluster_bootstrap") or {}
    external_permutation = external_verdict.get("pair_aware_permutation") or {}
    bootstrap_n = int(external_bootstrap.get("n_valid") or 0)
    permutation_n = int(external_permutation.get("n_permutations") or 0)
    external_stats_ok = (
        bool(external_bootstrap.get("available"))
        and bootstrap_n >= min_external_bootstrap
        and bool(external_permutation.get("available"))
        and permutation_n >= min_external_permutations
    )
    gates.append(
        _gate(
            "external_statistical_evidence",
            "pass" if external_stats_ok else "fail",
            (
                "External patient-cluster bootstrap and pair-aware permutation evidence are sufficient."
                if external_stats_ok
                else "External bootstrap or pair-aware permutation evidence is missing or undersized."
            ),
            {
                "bootstrap_valid": bootstrap_n,
                "required_bootstrap": min_external_bootstrap,
                "permutations": permutation_n,
                "required_permutations": min_external_permutations,
            },
        )
    )

    external_classical_auc = external_verdict.get("external_classical_roc_auc")
    external_classical_p = external_permutation.get("p_value_classical_auc_ge_observed")
    external_classical_ok = (
        external_classical_auc is not None
        and float(external_classical_auc) >= 0.7
        and external_classical_p is not None
        and float(external_classical_p) <= alpha
    )
    gates.append(
        _gate(
            "external_classical_signal",
            "pass" if external_classical_ok else "fail",
            (
                "Locked classical model generalizes with significant external discrimination."
                if external_classical_ok
                else "Locked classical model lacks sufficient significant external discrimination."
            ),
            {
                "external_classical_roc_auc": external_classical_auc,
                "pair_aware_permutation_p": external_classical_p,
                "minimum_roc_auc": 0.7,
                "alpha": alpha,
            },
        )
    )

    external_quantum_claim = bool(external_verdict.get("external_quantum_adds_value"))
    external_delta = external_verdict.get("external_delta_roc_auc")
    external_delta_ci_low = (
        external_bootstrap.get("metrics", {}).get("delta_roc_auc", {}).get("ci95_low")
    )
    external_delta_p = external_permutation.get("p_value_delta_auc_ge_observed")
    external_quantum_supported = (
        external_quantum_claim
        and external_delta is not None
        and float(external_delta) > 0
        and external_delta_ci_low is not None
        and float(external_delta_ci_low) > 0
        and external_delta_p is not None
        and float(external_delta_p) <= alpha
    )
    external_quantum_gate_ok = not external_quantum_claim or external_quantum_supported
    gates.append(
        _gate(
            "external_quantum_value_claim",
            "pass" if external_quantum_gate_ok else "fail",
            (
                "No external quantum-advantage claim is made."
                if not external_quantum_claim
                else (
                    "External quantum-value claim passes delta, confidence-interval, and permutation gates."
                    if external_quantum_supported
                    else "External quantum-value claim is not supported by the predeclared gates."
                )
            ),
            {
                "external_quantum_adds_value": external_quantum_claim,
                "external_delta_roc_auc": external_delta,
                "external_delta_ci95_low": external_delta_ci_low,
                "external_delta_permutation_p": external_delta_p,
            },
        )
    )

    classical_p = full_perm.get("p_value_best_classical_auc_ge_observed")
    if classical_p is not None and float(classical_p) <= alpha:
        status = "pass"
        msg = "Best classical classifier signal is significant under full retraining permutation."
    elif classical_p is not None:
        status = "warn"
        msg = "Best classical classifier signal is not significant under full retraining permutation."
    else:
        status = "fail"
        msg = "Best classical classifier permutation p-value is unavailable."
    gates.append(
        _gate(
            "classical_classifier_signal",
            status,
            msg,
            {
                "best_classical_roc_auc": verdict.get("best_classical_roc_auc"),
                "full_permutation_classical_p": classical_p,
                "alpha": alpha,
            },
        )
    )

    fail_count = sum(1 for gate in gates if gate["status"] == "fail")
    warn_count = sum(1 for gate in gates if gate["status"] == "warn")
    if fail_count == 0 and warn_count == 0:
        readiness = "review_ready"
    elif fail_count == 0:
        readiness = "screening_grade"
    elif sample_status == "fail":
        readiness = "pilot_only"
    else:
        readiness = "not_review_ready"

    if readiness == "review_ready":
        claim_guidance = (
            "The technical evidence package passes the configured external-review gates. "
            "Limit conclusions to the tested cross-study endpoint and the observed quantum result."
        )
    elif readiness == "screening_grade":
        claim_guidance = "Use as screening evidence; avoid strong biological or quantum-advantage claims."
    elif readiness == "pilot_only":
        claim_guidance = "Use only as a pilot sanity check; collect a larger independent cohort before claims."
    else:
        claim_guidance = "Do not use for external scientific claims until failed gates are resolved."

    return {
        "readiness": readiness,
        "worst_gate_status": _worst_status(gates),
        "claim_guidance": claim_guidance,
        "quantum_adds_value": quantum_adds_value,
        "classifier_verdict": verdict.get("classifier_verdict"),
        "external_classifier_verdict": external_verdict.get("external_classifier_verdict"),
        "gates": gates,
        "thresholds": {
            "min_review_samples": min_publishable_samples,
            "min_review_class_n": min_publishable_class_n,
            "min_screening_samples": min_screening_samples,
            "min_screening_class_n": min_screening_class_n,
            "min_ranking_candidates": min_ranking_candidates,
            "min_full_permutations": min_full_permutations,
            "alpha": alpha,
            "min_external_samples": min_external_samples,
            "min_external_class_n": min_external_class_n,
            "min_external_bootstrap": min_external_bootstrap,
            "min_external_permutations": min_external_permutations,
        },
    }


def _write_markdown(audit: Dict[str, Any], path: Path) -> None:
    rows = []
    for gate in audit["gates"]:
        rows.append(
            f"| `{gate['id']}` | `{gate['status']}` | {gate['message']} |"
        )
    lines = [
        "# RNA-seq Quantum Evidence Audit",
        "",
        f"- Readiness: `{audit['readiness']}`",
        f"- Worst gate status: `{audit['worst_gate_status']}`",
        f"- Classifier verdict: `{audit.get('classifier_verdict')}`",
        f"- External classifier verdict: `{audit.get('external_classifier_verdict')}`",
        f"- Quantum adds value: `{audit.get('quantum_adds_value')}`",
        f"- Claim guidance: {audit['claim_guidance']}",
        "",
        "## Gates",
        "",
        "| Gate | Status | Message |",
        "|:--|:--|:--|",
        *rows,
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", default="artifacts/benchmarks/rnaseq_quantum_creeds_airway")
    parser.add_argument("--verdict", default=None, help="Optional explicit quantum_value_verdict.json path.")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--min-publishable-samples", type=int, default=60)
    parser.add_argument("--min-publishable-class-n", type=int, default=25)
    parser.add_argument("--min-screening-samples", type=int, default=20)
    parser.add_argument("--min-screening-class-n", type=int, default=10)
    parser.add_argument("--min-ranking-candidates", type=int, default=50)
    parser.add_argument("--min-full-permutations", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--external-validation",
        default=None,
        help="Independent external_validation.json generated by run_rnaseq_external_validation.py.",
    )
    parser.add_argument("--min-external-samples", type=int, default=60)
    parser.add_argument("--min-external-class-n", type=int, default=25)
    parser.add_argument("--min-external-bootstrap", type=int, default=1000)
    parser.add_argument("--min-external-permutations", type=int, default=1000)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    verdict_path = Path(args.verdict) if args.verdict else benchmark_dir / "quantum_value_verdict.json"
    external_validation = _read_json(Path(args.external_validation)) if args.external_validation else None
    audit = audit_verdict(
        _read_json(verdict_path),
        min_publishable_samples=args.min_publishable_samples,
        min_publishable_class_n=args.min_publishable_class_n,
        min_screening_samples=args.min_screening_samples,
        min_screening_class_n=args.min_screening_class_n,
        min_ranking_candidates=args.min_ranking_candidates,
        min_full_permutations=args.min_full_permutations,
        alpha=args.alpha,
        external_validation=external_validation,
        min_external_samples=args.min_external_samples,
        min_external_class_n=args.min_external_class_n,
        min_external_bootstrap=args.min_external_bootstrap,
        min_external_permutations=args.min_external_permutations,
    )

    out_json = Path(args.out_json) if args.out_json else benchmark_dir / "evidence_audit.json"
    out_md = Path(args.out_md) if args.out_md else benchmark_dir / "evidence_audit.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    _write_markdown(audit, out_md)

    print(json.dumps({"audit": audit, "outputs": {"json": str(out_json), "markdown": str(out_md)}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
