from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import importlib.util
import pandas as pd


SCRIPT_PATH = Path("scripts/run_rnaseq_quantum_benchmark.py")
SPEC = importlib.util.spec_from_file_location("run_rnaseq_quantum_benchmark", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["run_rnaseq_quantum_benchmark"] = benchmark
SPEC.loader.exec_module(benchmark)


FIXTURE_DIR = Path("tests/fixtures/rnaseq_counts")


def test_load_expression_inputs_can_expose_full_feature_universe(tmp_path: Path) -> None:
    normalized = tmp_path / "normalized.csv"
    normalized.write_text(
        "sample_id,GENE_A,GENE_B,GENE_C\n"
        "case,2,3,4\ncontrol,1,2,3\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("sample_id,condition\ncase,disease\ncontrol,control\n", encoding="utf-8")

    full, _, _, genes = benchmark.load_expression_inputs(
        normalized,
        metadata,
        sample_id_col="sample_id",
        condition_col="condition",
        case_label="disease",
        control_label="control",
        signature_genes=None,
    )
    restricted, _, _, _ = benchmark.load_expression_inputs(
        normalized,
        metadata,
        sample_id_col="sample_id",
        condition_col="condition",
        case_label="disease",
        control_label="control",
        signature_genes=["GENE_A"],
    )

    assert genes == ["GENE_A", "GENE_B", "GENE_C"]
    assert list(full.columns) == ["GENE_A", "GENE_B", "GENE_C"]
    assert list(restricted.columns) == ["GENE_A"]


def test_build_value_verdict_flags_quantum_lift() -> None:
    metrics = pd.DataFrame(
        [
            {"model": "logistic_regression", "status": "ok", "roc_auc": 0.70, "balanced_accuracy": 0.65},
            {"model": "rbf_svm", "status": "ok", "roc_auc": 0.75, "balanced_accuracy": 0.70},
            {
                "model": "qsvc_quantum",
                "status": "ok",
                "message": "best_config=qsvc_quantum_dim2_reps1",
                "qml_dim": 2,
                "qsvc_reps": 1,
                "roc_auc": 0.82,
                "balanced_accuracy": 0.80,
            },
        ]
    )
    predictions = pd.DataFrame(
        [
            {"sample_index": 0, "model": "rbf_svm", "y_true": 0, "y_score": 0.2, "y_pred": 0},
            {"sample_index": 1, "model": "rbf_svm", "y_true": 0, "y_score": 0.4, "y_pred": 0},
            {"sample_index": 2, "model": "rbf_svm", "y_true": 1, "y_score": 0.6, "y_pred": 1},
            {"sample_index": 3, "model": "rbf_svm", "y_true": 1, "y_score": 0.7, "y_pred": 1},
            {"sample_index": 0, "model": "qsvc_quantum", "y_true": 0, "y_score": 0.1, "y_pred": 0},
            {"sample_index": 1, "model": "qsvc_quantum", "y_true": 0, "y_score": 0.2, "y_pred": 0},
            {"sample_index": 2, "model": "qsvc_quantum", "y_true": 1, "y_score": 0.8, "y_pred": 1},
            {"sample_index": 3, "model": "qsvc_quantum", "y_true": 1, "y_score": 0.9, "y_pred": 1},
        ]
    )

    verdict = benchmark.build_value_verdict(
        metrics,
        min_delta=0.02,
        predictions_df=predictions,
        n_bootstrap=25,
        random_state=1,
    )

    assert verdict["classifier_verdict"] == "quantum_adds_value"
    assert verdict["quantum_adds_value"] is True
    assert verdict["delta_roc_auc"] == 0.06999999999999995
    assert verdict["best_quantum_qml_dim"] == 2
    assert verdict["best_classical_model"] == "rbf_svm"
    assert verdict["bootstrap_delta_roc_auc"]["available"] is True
    assert verdict["classifier_evidence_grade"] == "pilot_underpowered"
    assert verdict["classifier_sample_size_warning"] is True
    assert verdict["permutation_auc_best_quantum"]["available"] is True


def test_build_value_verdict_flags_quantum_underperformance() -> None:
    metrics = pd.DataFrame(
        [
            {"model": "logistic_regression", "status": "ok", "roc_auc": 1.0, "balanced_accuracy": 1.0},
            {"model": "rbf_svm", "status": "ok", "roc_auc": 0.9, "balanced_accuracy": 0.9},
            {
                "model": "qsvc_quantum",
                "status": "ok",
                "message": "best_config=qsvc_quantum_dim3_reps1",
                "qml_dim": 3,
                "qsvc_reps": 1,
                "roc_auc": 0.4,
                "balanced_accuracy": 0.5,
            },
        ]
    )

    verdict = benchmark.build_value_verdict(metrics, min_delta=0.02)

    assert verdict["classifier_verdict"] == "quantum_underperforms_classical"
    assert verdict["quantum_adds_value"] is False
    assert "bootstrap_delta_roc_auc" in verdict
    assert verdict["classifier_evidence_grade"] == "not_evaluable"


def test_build_ranking_materiality_flags_negligible_real_ranking() -> None:
    ranking_metrics = pd.DataFrame(
        [
            {"ranking_evidence_level": "creeds_signatures", "quantum_status": "ok", "metric": "candidate_count", "value": 67.0},
            {
                "ranking_evidence_level": "creeds_signatures",
                "quantum_status": "ok",
                "metric": "kg_omics_vs_quantum_spearman",
                "value": 0.997,
            },
            {"ranking_evidence_level": "creeds_signatures", "quantum_status": "ok", "metric": "rank_shift_mean", "value": 0.5},
            {"ranking_evidence_level": "creeds_signatures", "quantum_status": "ok", "metric": "rank_shift_max", "value": 4.0},
            {
                "ranking_evidence_level": "creeds_signatures",
                "quantum_status": "ok",
                "metric": "quantum_delta_score_std",
                "value": 0.001,
            },
            {
                "ranking_evidence_level": "creeds_signatures",
                "quantum_status": "ok",
                "metric": "top_10_overlap_fraction",
                "value": 1.0,
            },
        ]
    )

    materiality = benchmark.build_ranking_materiality(ranking_metrics, ranking_is_real_evidence=True)

    assert materiality["ranking_quantum_materiality"] == "negligible"
    assert materiality["ranking_quantum_changes_top_k"] is False
    assert materiality["ranking_spearman_kg_omics_vs_quantum"] == 0.997


def test_load_cmap_candidate_profiles_requires_overlap(tmp_path: Path) -> None:
    cmap = tmp_path / "cmap.csv"
    cmap.write_text(
        "compound,gene,score\n"
        "drug_a,GENE_UP,-2.0\n"
        "drug_a,GENE_DOWN,2.0\n"
        "drug_b,GENE_UP,1.5\n"
        "drug_b,GENE_DOWN,-1.5\n"
        "drug_sparse,GENE_UP,0.1\n",
        encoding="utf-8",
    )

    profiles = benchmark.load_cmap_candidate_profiles(
        cmap,
        ["GENE_UP", "GENE_DOWN"],
        min_gene_overlap=2,
    )

    assert set(profiles["compound"]) == {"drug_a", "drug_b"}
    assert {"GENE_UP", "GENE_DOWN", "profile_gene_overlap"}.issubset(profiles.columns)


def test_load_creeds_candidate_profiles_uses_gene_map(tmp_path: Path) -> None:
    creeds = tmp_path / "creeds.json"
    creeds.write_text(
        json.dumps(
            [
                {
                    "id": "drug:1",
                    "drug_name": "drug_a",
                    "organism": "human",
                    "drugbank_id": "DBTEST",
                    "geo_id": "GSETEST",
                    "cell_type": "cell",
                    "up_genes": [["GENEUP", 2.0]],
                    "down_genes": [["GENEDOWN", -2.0]],
                },
                {
                    "id": "drug:2",
                    "drug_name": "mouse_drug",
                    "organism": "mouse",
                    "up_genes": [["GENEUP", 2.0]],
                    "down_genes": [["GENEDOWN", -2.0]],
                },
            ]
        ),
        encoding="utf-8",
    )
    gene_map = tmp_path / "gene_map.csv"
    gene_map.write_text(
        "gene_id,gene_symbol\nENSG_UP,GENEUP\nENSG_DOWN,GENEDOWN\n",
        encoding="utf-8",
    )

    profiles = benchmark.load_creeds_candidate_profiles(
        creeds,
        ["ENSG_UP", "ENSG_DOWN"],
        gene_map_path=str(gene_map),
        organism="human",
        min_gene_overlap=2,
        max_profiles=10,
    )

    assert profiles["compound"].tolist() == ["drug_a"]
    assert profiles.iloc[0]["ENSG_UP"] == 2.0
    assert profiles.iloc[0]["ENSG_DOWN"] == -2.0
    assert profiles.iloc[0]["profile_gene_overlap"] == 2


def test_load_kg_scores_merged_creeds_profiles(tmp_path: Path) -> None:
    creeds = tmp_path / "creeds.json"
    creeds.write_text(
        json.dumps(
            [
                {
                    "id": "drug:1",
                    "drug_name": "Vemurafenib",
                    "organism": "human",
                    "up_genes": [["GENEUP", 2.0]],
                    "down_genes": [["GENEDOWN", -2.0]],
                },
                {
                    "id": "drug:2",
                    "drug_name": "UnmatchedDrug",
                    "organism": "human",
                    "up_genes": [["GENEUP", 1.0]],
                    "down_genes": [["GENEDOWN", -1.0]],
                },
            ]
        ),
        encoding="utf-8",
    )
    gene_map = tmp_path / "gene_map.csv"
    gene_map.write_text(
        "gene_id,gene_symbol\nENSG_UP,GENEUP\nENSG_DOWN,GENEDOWN\n",
        encoding="utf-8",
    )
    kg_scores = tmp_path / "kg_scores.json"
    kg_scores.write_text(
        json.dumps(
            [
                {
                    "compound": "Vemurafenib",
                    "compound_hetionet_id": "Compound::DB123",
                    "disease": "Breast cancer",
                    "disease_hetionet_id": "Disease::DOID:1612",
                    "kg_rotate_score": 0.9,
                    "kg_complex_score": 0.8,
                    "graph_topology_score": 0.7,
                    "qsvc_score": 0.6,
                    "classical_ensemble_score": 0.85,
                },
                {
                    "compound": "OtherDrug",
                    "compound_hetionet_id": "Compound::DB999",
                    "disease": "Breast cancer",
                    "disease_hetionet_id": "Disease::DOID:1612",
                    "kg_rotate_score": 0.99,
                    "kg_complex_score": 0.5,
                    "graph_topology_score": 0.5,
                    "qsvc_score": 0.5,
                    "classical_ensemble_score": 0.5,
                },
            ]
        ),
        encoding="utf-8",
    )

    merged = benchmark.load_kg_scores_merged_creeds_profiles(
        kg_scores,
        creeds,
        ["ENSG_UP", "ENSG_DOWN"],
        gene_map_path=str(gene_map),
        organism="human",
        min_gene_overlap=2,
        max_profiles=10,
        disease_hetionet_id="Disease::DOID:1612",
    )

    assert set(merged["compound"]) == {"Vemurafenib"}
    assert merged.iloc[0]["kg_rotate_score"] == 0.9
    assert merged.iloc[0]["candidate_id"] == "Compound::DB123::Disease::DOID:1612"


def test_rnaseq_quantum_benchmark_cli_smoke(tmp_path: Path) -> None:
    single_cell_dir = tmp_path / "single_cell"
    signatures_dir = tmp_path / "signatures"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_rnaseq_counts_pipeline.py",
            "--input",
            str(FIXTURE_DIR / "counts.csv"),
            "--format",
            "count-matrix",
            "--metadata",
            str(FIXTURE_DIR / "metadata.csv"),
            "--out-dir",
            str(single_cell_dir),
            "--signatures-dir",
            str(signatures_dir),
            "--disease-id",
            "Disease::TEST",
            "--top-n",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    out_dir = tmp_path / "benchmark"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_rnaseq_quantum_benchmark.py",
            "--normalized-counts",
            str(single_cell_dir / "normalized_counts.csv"),
            "--metadata",
            str(FIXTURE_DIR / "metadata.csv"),
            "--signature",
            str(signatures_dir / "disease_signature.json"),
            "--out-dir",
            str(out_dir),
            "--case-label",
            "disease",
            "--control-label",
            "control",
            "--top-genes",
            "4",
            "--qml-dim",
            "2",
            "--qml-dims",
            "2,3",
            "--qsvc-reps",
            "1",
            "--full-permutations",
            "2",
            "--full-permutation-qml-dims",
            "2",
            "--full-permutation-qsvc-reps-list",
            "1",
            "--demo-ranking",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["outputs"]["quantum_value_verdict"].endswith("quantum_value_verdict.json")
    assert payload["verdict"]["evidence_scope"] == "real_classifier_only_ranking_not_evaluable"
    assert payload["verdict"]["ranking_is_real_evidence"] is False
    assert payload["verdict"]["classifier_evidence_grade"] == "pilot_underpowered"
    assert payload["verdict"]["ranking_quantum_materiality"] == "not_real_evidence"
    assert payload["verdict"]["full_retraining_permutation"]["available"] is True
    assert payload["verdict"]["full_retraining_permutation"]["n_permutations"] == 2

    expected = [
        out_dir / "classifier_metrics.csv",
        out_dir / "classifier_predictions.csv",
        out_dir / "classifier_full_permutation.csv",
        out_dir / "ranking_comparison.csv",
        out_dir / "ranking_metrics.csv",
        out_dir / "quantum_value_verdict.json",
        out_dir / "benchmark_manifest.json",
        out_dir / "benchmark_report.md",
    ]
    for path in expected:
        assert path.exists(), path

    metrics = pd.read_csv(out_dir / "classifier_metrics.csv")
    assert {"logistic_regression", "rbf_svm", "qsvc_quantum"}.issubset(set(metrics["model"]))
    assert {"qsvc_quantum_dim2_reps1", "qsvc_quantum_dim3_reps1"}.issubset(set(metrics["model"]))
    assert metrics.loc[metrics["model"] == "qsvc_quantum", "status"].iloc[0] in {"ok", "failed"}

    manifest = json.loads((out_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ranking"]["ranking_evidence_level"] == "demo_signature_challenge"
    assert manifest["quantum_sweep"]["qml_dims"] == [2, 3]
    assert manifest["quantum_sweep"]["full_permutation_sweep"]["qml_dims"] == [2]
    assert manifest["statistical_guardrails"]["score_label_permutations"] == 1000
    assert manifest["statistical_guardrails"]["full_retraining_permutations"] == 2
    assert manifest["classifier_feature_universe"] == {
        "n_measured_genes": 6,
        "source": "all_measured_genes",
        "selection_timing": "inside_each_training_fold",
        "cohort_wide_signature_used_for_classifier": False,
    }
