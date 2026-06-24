from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/audit_rnaseq_quantum_evidence.py")
SPEC = importlib.util.spec_from_file_location("audit_rnaseq_quantum_evidence", SCRIPT_PATH)
audit_module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["audit_rnaseq_quantum_evidence"] = audit_module
SPEC.loader.exec_module(audit_module)


def _external_validation(*, quantum_adds_value: bool) -> dict:
    delta = 0.05 if quantum_adds_value else -0.06
    return {
        "independent_cohorts": True,
        "validation_sample_counts": {"n_samples": 155, "min_class_n": 75},
        "leakage_controls": {
            "validation_labels_used_for_feature_selection": False,
            "validation_labels_used_for_hyperparameter_selection": False,
            "validation_samples_used_for_scaler_or_pca_fit": False,
            "validation_samples_used_for_model_fit": False,
            "development_configuration_locked_before_validation": True,
            "validation_de_signature_generated": False,
            "shared_gene_universe_normalization_verified": True,
        },
        "verdict": {
            "external_classifier_verdict": (
                "quantum_adds_value_external"
                if quantum_adds_value
                else "quantum_underperforms_classical_external"
            ),
            "external_quantum_adds_value": quantum_adds_value,
            "external_classical_roc_auc": 0.95,
            "external_delta_roc_auc": delta,
            "patient_cluster_bootstrap": {
                "available": True,
                "n_valid": 2000,
                "metrics": {"delta_roc_auc": {"ci95_low": 0.01 if quantum_adds_value else -0.1}},
            },
            "pair_aware_permutation": {
                "available": True,
                "n_permutations": 10000,
                "p_value_classical_auc_ge_observed": 0.001,
                "p_value_delta_auc_ge_observed": 0.01 if quantum_adds_value else 0.98,
            },
        },
    }


def _audit(verdict: dict, external_validation: dict | None = None) -> dict:
    return audit_module.audit_verdict(
        verdict,
        min_publishable_samples=60,
        min_publishable_class_n=25,
        min_screening_samples=20,
        min_screening_class_n=10,
        min_ranking_candidates=50,
        min_full_permutations=100,
        alpha=0.05,
        external_validation=external_validation,
    )


def test_audit_flags_current_scale_as_pilot_only() -> None:
    verdict = {
        "classifier_verdict": "quantum_underperforms_classical",
        "quantum_adds_value": False,
        "classifier_n_samples": 8,
        "classifier_min_class_n": 4,
        "delta_roc_auc": -0.3125,
        "best_classical_roc_auc": 1.0,
        "full_retraining_permutation": {
            "available": True,
            "n_valid": 100,
            "p_value_best_classical_auc_ge_observed": 0.158,
            "p_value_delta_auc_ge_observed": 0.663,
        },
        "ranking_is_real_evidence": True,
        "ranking_evidence_level": "creeds_signatures",
        "ranking_candidate_count": 67,
        "ranking_quantum_materiality": "negligible",
        "ranking_spearman_kg_omics_vs_quantum": 0.997,
        "ranking_top_10_overlap_fraction": 1.0,
    }

    audit = _audit(verdict)

    assert audit["readiness"] == "pilot_only"
    assert audit["worst_gate_status"] == "fail"
    gates = {gate["id"]: gate for gate in audit["gates"]}
    assert gates["classifier_sample_size"]["status"] == "fail"
    assert gates["full_retraining_permutation"]["status"] == "pass"
    assert gates["ranking_evidence"]["status"] == "pass"
    assert gates["ranking_quantum_materiality"]["status"] == "pass"
    assert gates["independent_external_validation"]["status"] == "fail"


def test_audit_allows_review_ready_when_all_gates_pass() -> None:
    verdict = {
        "classifier_verdict": "quantum_adds_value",
        "quantum_adds_value": True,
        "classifier_n_samples": 80,
        "classifier_min_class_n": 35,
        "delta_roc_auc": 0.08,
        "bootstrap_delta_roc_auc": {"delta_ci95_low": 0.02},
        "best_classical_roc_auc": 0.82,
        "full_retraining_permutation": {
            "available": True,
            "n_valid": 500,
            "p_value_best_classical_auc_ge_observed": 0.01,
            "p_value_delta_auc_ge_observed": 0.02,
        },
        "ranking_is_real_evidence": True,
        "ranking_evidence_level": "creeds_signatures",
        "ranking_candidate_count": 250,
        "ranking_quantum_materiality": "material",
        "ranking_spearman_kg_omics_vs_quantum": 0.90,
        "ranking_top_10_overlap_fraction": 0.7,
    }

    audit = _audit(verdict, _external_validation(quantum_adds_value=True))

    assert audit["readiness"] == "review_ready"
    assert audit["worst_gate_status"] == "pass"
    assert all(gate["status"] == "pass" for gate in audit["gates"])


def test_audit_allows_review_ready_negative_quantum_result_with_external_validation() -> None:
    verdict = {
        "classifier_verdict": "quantum_matches_classical_within_delta",
        "quantum_adds_value": False,
        "classifier_n_samples": 60,
        "classifier_min_class_n": 30,
        "delta_roc_auc": 0.0,
        "best_classical_roc_auc": 1.0,
        "full_retraining_permutation": {
            "available": True,
            "n_valid": 100,
            "p_value_best_classical_auc_ge_observed": 0.01,
            "p_value_delta_auc_ge_observed": 0.5,
        },
        "ranking_is_real_evidence": True,
        "ranking_evidence_level": "creeds_signatures",
        "ranking_candidate_count": 63,
        "ranking_quantum_materiality": "negligible",
        "ranking_spearman_kg_omics_vs_quantum": 0.999,
        "ranking_top_10_overlap_fraction": 1.0,
    }

    audit = _audit(verdict, _external_validation(quantum_adds_value=False))

    assert audit["readiness"] == "review_ready"
    gates = {gate["id"]: gate for gate in audit["gates"]}
    assert gates["ranking_quantum_materiality"]["status"] == "pass"
    assert gates["independent_external_validation"]["status"] == "pass"
    assert gates["external_quantum_value_claim"]["status"] == "pass"
