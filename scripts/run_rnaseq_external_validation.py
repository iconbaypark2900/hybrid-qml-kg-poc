#!/usr/bin/env python3
"""Evaluate locked RNA-seq classical and quantum models on an independent cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import confusion_matrix, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import run_rnaseq_quantum_benchmark as benchmark


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extended_metrics(y: np.ndarray, scores: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
    metrics = benchmark._probability_metrics(y, scores, predictions)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics.update(
        {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
            "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
            "n_correct": int(tp + tn),
            "n_errors": int(fp + fn),
        }
    )
    return metrics


def _cluster_resample_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique = np.unique(groups)
    sampled = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([np.flatnonzero(groups == group) for group in sampled])


def cluster_bootstrap(
    y: np.ndarray,
    groups: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    model_predictions: Dict[str, np.ndarray],
    *,
    quantum_model: str,
    classical_model: str,
    n_bootstrap: int,
    random_state: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []
    skipped = 0
    for iteration in range(int(n_bootstrap)):
        idx = _cluster_resample_indices(groups, rng)
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        row: Dict[str, Any] = {"bootstrap": iteration}
        for model in [classical_model, quantum_model]:
            row[f"{model}_roc_auc"] = float(roc_auc_score(y[idx], model_scores[model][idx]))
            row[f"{model}_balanced_accuracy"] = float(
                benchmark.balanced_accuracy_score(y[idx], model_predictions[model][idx])
            )
        row["delta_roc_auc"] = row[f"{quantum_model}_roc_auc"] - row[f"{classical_model}_roc_auc"]
        row["delta_balanced_accuracy"] = (
            row[f"{quantum_model}_balanced_accuracy"] - row[f"{classical_model}_balanced_accuracy"]
        )
        rows.append(row)

    details = pd.DataFrame(rows)
    if details.empty:
        return {"available": False, "reason": "no valid patient-cluster bootstrap samples"}, details

    summary: Dict[str, Any] = {
        "available": True,
        "method": "patient_cluster_bootstrap",
        "n_requested": int(n_bootstrap),
        "n_valid": int(len(details)),
        "n_skipped": int(skipped),
        "n_clusters": int(len(np.unique(groups))),
        "metrics": {},
    }
    for column in details.columns:
        if column == "bootstrap":
            continue
        values = details[column].to_numpy(dtype=float)
        summary["metrics"][column] = {
            "mean": float(values.mean()),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
    return summary, details


def _pair_aware_permutation(y: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    permuted = y.copy()
    singleton_indices: List[int] = []
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        labels = y[idx]
        if len(idx) == 2 and set(labels.tolist()) == {0, 1}:
            if rng.integers(0, 2):
                permuted[idx] = labels[::-1]
        else:
            singleton_indices.extend(idx.tolist())
    if singleton_indices:
        singleton = np.asarray(singleton_indices, dtype=int)
        permuted[singleton] = rng.permutation(y[singleton])
    return permuted


def pair_aware_permutation_test(
    y: np.ndarray,
    groups: np.ndarray,
    model_scores: Dict[str, np.ndarray],
    *,
    quantum_model: str,
    classical_model: str,
    n_permutations: int,
    random_state: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    observed_q = float(roc_auc_score(y, model_scores[quantum_model]))
    observed_c = float(roc_auc_score(y, model_scores[classical_model]))
    observed_delta = observed_q - observed_c
    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []
    for iteration in range(int(n_permutations)):
        labels = _pair_aware_permutation(y, groups, rng)
        q_auc = float(roc_auc_score(labels, model_scores[quantum_model]))
        c_auc = float(roc_auc_score(labels, model_scores[classical_model]))
        rows.append(
            {
                "permutation": iteration,
                "quantum_roc_auc": q_auc,
                "classical_roc_auc": c_auc,
                "delta_roc_auc": q_auc - c_auc,
            }
        )
    details = pd.DataFrame(rows)
    if details.empty:
        return {"available": False, "reason": "permutation count is zero"}, details
    return {
        "available": True,
        "method": "within_pair_swap_plus_unpaired_label_permutation",
        "n_permutations": int(len(details)),
        "observed_quantum_roc_auc": observed_q,
        "observed_classical_roc_auc": observed_c,
        "observed_delta_roc_auc": observed_delta,
        "p_value_quantum_auc_ge_observed": float(
            (1 + np.sum(details["quantum_roc_auc"] >= observed_q)) / (len(details) + 1)
        ),
        "p_value_classical_auc_ge_observed": float(
            (1 + np.sum(details["classical_roc_auc"] >= observed_c)) / (len(details) + 1)
        ),
        "p_value_delta_auc_ge_observed": float(
            (1 + np.sum(details["delta_roc_auc"] >= observed_delta)) / (len(details) + 1)
        ),
        "null_quantum_auc_mean": float(details["quantum_roc_auc"].mean()),
        "null_classical_auc_mean": float(details["classical_roc_auc"].mean()),
        "null_delta_auc_mean": float(details["delta_roc_auc"].mean()),
    }, details


def _mcnemar(quantum_correct: np.ndarray, classical_correct: np.ndarray) -> Dict[str, Any]:
    q_only = int(np.sum(quantum_correct & ~classical_correct))
    c_only = int(np.sum(classical_correct & ~quantum_correct))
    discordant = q_only + c_only
    p_value = float(binomtest(min(q_only, c_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "method": "exact_mcnemar_binomial",
        "quantum_only_correct": q_only,
        "classical_only_correct": c_only,
        "discordant": discordant,
        "p_value_two_sided": p_value,
    }


def _write_report(result: Dict[str, Any], path: Path) -> None:
    verdict = result["verdict"]
    lines = [
        "# Independent RNA-seq External Validation",
        "",
        f"- Development cohort: `{result['cohorts']['development']}`",
        f"- Validation cohort: `{result['cohorts']['validation']}`",
        f"- Validation samples: `{result['validation_sample_counts']['n_samples']}`",
        f"- Validation patients: `{result['validation_sample_counts']['n_patients']}`",
        f"- Endpoint: `{result['endpoint']}`",
        f"- Selected genes: `{result['feature_lock']['n_selected_genes']}` from the development cohort only",
        f"- Classical ROC-AUC: `{verdict['external_classical_roc_auc']:.4f}`",
        f"- Quantum ROC-AUC: `{verdict['external_quantum_roc_auc']:.4f}`",
        f"- Quantum minus classical ROC-AUC: `{verdict['external_delta_roc_auc']:.4f}`",
        f"- External verdict: `{verdict['external_classifier_verdict']}`",
        f"- Quantum adds value externally: `{verdict['external_quantum_adds_value']}`",
        "",
        "## Leakage Controls",
        "",
        "- Validation labels were not used for feature selection, scaling, PCA, hyperparameter selection, or fitting.",
        "- Model family and quantum circuit configuration were locked from development cross-validation.",
        "- Patient-cluster bootstrap and pair-aware permutations account for matched tumor/normal samples.",
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_external_validation(args: argparse.Namespace) -> Dict[str, Any]:
    development_verdict = _read_json(args.development_verdict)
    development_manifest = _read_json(args.development_manifest)
    harmonization_manifest = _read_json(args.harmonization_manifest)
    if args.development_cohort == args.validation_cohort:
        raise ValueError("Development and validation cohort identifiers must differ.")
    if harmonization_manifest.get("development_cohort") != args.development_cohort:
        raise ValueError("Harmonization manifest development cohort does not match --development-cohort.")
    if harmonization_manifest.get("validation_cohort") != args.validation_cohort:
        raise ValueError("Harmonization manifest validation cohort does not match --validation-cohort.")
    if bool(harmonization_manifest.get("labels_used")):
        raise ValueError("Harmonization manifest indicates that labels were used.")
    harmonized_outputs = harmonization_manifest.get("outputs") or {}
    expected_development = Path(str(harmonized_outputs.get("development_normalized_counts", ""))).resolve()
    expected_validation = Path(str(harmonized_outputs.get("validation_normalized_counts", ""))).resolve()
    if Path(args.development_counts).resolve() != expected_development:
        raise ValueError("Development counts path does not match the harmonization manifest output.")
    if Path(args.validation_counts).resolve() != expected_validation:
        raise ValueError("Validation counts path does not match the harmonization manifest output.")

    dev_X, dev_meta, dev_y, _ = benchmark.load_expression_inputs(
        args.development_counts,
        args.development_metadata,
        sample_id_col=args.sample_id_col,
        condition_col=args.condition_col,
        case_label=args.case_label,
        control_label=args.control_label,
        signature_genes=None,
    )
    val_X, val_meta, val_y, _ = benchmark.load_expression_inputs(
        args.validation_counts,
        args.validation_metadata,
        sample_id_col=args.sample_id_col,
        condition_col=args.condition_col,
        case_label=args.case_label,
        control_label=args.control_label,
        signature_genes=None,
    )
    common_genes = [gene for gene in dev_X.columns if gene in val_X.columns]
    development_genes = list(map(str, dev_X.columns))
    dev_all = dev_X.to_numpy(dtype=float)
    selected_idx = benchmark._select_fold_features(dev_all, dev_y, development_genes, args.top_genes)
    selected_genes = [development_genes[idx] for idx in selected_idx]
    missing_locked_genes = [gene for gene in selected_genes if gene not in val_X.columns]
    if missing_locked_genes:
        raise ValueError(
            "The validation cohort is missing genes selected and locked from development: "
            f"{missing_locked_genes[:10]}"
        )
    dev_selected = dev_all[:, selected_idx]
    val_selected = val_X[selected_genes].to_numpy(dtype=float)

    case_mean = dev_selected[dev_y == 1].mean(axis=0)
    control_mean = dev_selected[dev_y == 0].mean(axis=0)
    feature_lock = pd.DataFrame(
        {
            "rank": np.arange(1, len(selected_genes) + 1),
            "gene": selected_genes,
            "development_abs_log2fc": np.abs(np.log2((case_mean + 1.0) / (control_mean + 1.0))),
        }
    )

    classical_name = str(development_verdict.get("best_classical_model") or "")
    if classical_name not in {"logistic_regression", "rbf_svm"}:
        raise ValueError(f"Unsupported locked classical model: {classical_name!r}")
    qml_dim = int(development_verdict.get("best_quantum_qml_dim") or 0)
    qsvc_reps = int(development_verdict.get("best_quantum_qsvc_reps") or 0)
    if qml_dim < 1 or qsvc_reps < 1:
        raise ValueError("Development verdict does not contain a valid locked quantum configuration.")

    classical = benchmark._fit_full_classical_models(
        dev_selected,
        dev_y,
        random_state=args.random_state,
    )[classical_name]
    classical_scores = classical.predict_proba(val_selected)[:, 1]
    classical_predictions = classical.predict(val_selected).astype(int)

    dev_quantum, val_quantum = benchmark._prepare_quantum_features(
        dev_selected,
        val_selected,
        qml_dim=qml_dim,
    )
    quantum = benchmark._fit_local_qsvc(dev_quantum, dev_y, qml_dim=qml_dim, reps=qsvc_reps)
    quantum_scores = benchmark._decision_scores(quantum, val_quantum)
    quantum_predictions = benchmark._score_to_pred(quantum_scores)

    model_scores = {classical_name: classical_scores, "qsvc_quantum": quantum_scores}
    model_predictions = {classical_name: classical_predictions, "qsvc_quantum": quantum_predictions}
    metrics = {
        model: _extended_metrics(val_y, model_scores[model], model_predictions[model])
        for model in [classical_name, "qsvc_quantum"]
    }

    if args.patient_id_col and args.patient_id_col in val_meta.columns:
        groups = val_meta[args.patient_id_col].astype(str).to_numpy()
        cluster_source = args.patient_id_col
    else:
        groups = val_meta[args.sample_id_col].astype(str).to_numpy()
        cluster_source = args.sample_id_col
    bootstrap_summary, bootstrap_details = cluster_bootstrap(
        val_y,
        groups,
        model_scores,
        model_predictions,
        quantum_model="qsvc_quantum",
        classical_model=classical_name,
        n_bootstrap=args.bootstrap,
        random_state=args.random_state,
    )
    permutation_summary, permutation_details = pair_aware_permutation_test(
        val_y,
        groups,
        model_scores,
        quantum_model="qsvc_quantum",
        classical_model=classical_name,
        n_permutations=args.permutations,
        random_state=args.random_state + 1,
    )
    mcnemar = _mcnemar(quantum_predictions == val_y, classical_predictions == val_y)

    quantum_auc = metrics["qsvc_quantum"]["roc_auc"]
    classical_auc = metrics[classical_name]["roc_auc"]
    delta_auc = quantum_auc - classical_auc
    delta_ci = bootstrap_summary.get("metrics", {}).get("delta_roc_auc", {})
    delta_p = permutation_summary.get("p_value_delta_auc_ge_observed")
    quantum_adds_value = bool(
        delta_auc > args.min_delta
        and float(delta_ci.get("ci95_low", -math.inf)) > 0
        and delta_p is not None
        and float(delta_p) <= args.alpha
    )
    if quantum_adds_value:
        external_verdict = "quantum_adds_value_external"
    elif delta_auc < -args.min_delta:
        external_verdict = "quantum_underperforms_classical_external"
    else:
        external_verdict = "quantum_matches_classical_within_delta_external"

    verdict = {
        "external_classifier_verdict": external_verdict,
        "external_quantum_adds_value": quantum_adds_value,
        "min_delta": float(args.min_delta),
        "alpha": float(args.alpha),
        "locked_classical_model": classical_name,
        "locked_quantum_model": f"qsvc_quantum_dim{qml_dim}_reps{qsvc_reps}",
        "external_classical_roc_auc": float(classical_auc),
        "external_quantum_roc_auc": float(quantum_auc),
        "external_delta_roc_auc": float(delta_auc),
        "external_classical_balanced_accuracy": metrics[classical_name]["balanced_accuracy"],
        "external_quantum_balanced_accuracy": metrics["qsvc_quantum"]["balanced_accuracy"],
        "external_delta_balanced_accuracy": float(
            metrics["qsvc_quantum"]["balanced_accuracy"] - metrics[classical_name]["balanced_accuracy"]
        ),
        "patient_cluster_bootstrap": bootstrap_summary,
        "pair_aware_permutation": permutation_summary,
        "mcnemar": mcnemar,
    }

    sample_ids = val_meta[args.sample_id_col].astype(str).to_numpy()
    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "patient_id": groups,
                    "model": model,
                    "y_true": val_y,
                    "y_score": model_scores[model],
                    "y_pred": model_predictions[model],
                }
            )
            for model in [classical_name, "qsvc_quantum"]
        ],
        ignore_index=True,
    )
    metric_rows = [{"model": model, **values} for model, values in metrics.items()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out_dir / "external_metrics.csv",
        "predictions": out_dir / "external_predictions.csv",
        "selected_features": out_dir / "locked_features.csv",
        "bootstrap": out_dir / "external_cluster_bootstrap.csv",
        "permutations": out_dir / "external_pair_aware_permutations.csv",
        "verdict": out_dir / "external_validation.json",
        "manifest": out_dir / "external_validation_manifest.json",
        "report": out_dir / "external_validation_report.md",
    }
    pd.DataFrame(metric_rows).to_csv(paths["metrics"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    feature_lock.to_csv(paths["selected_features"], index=False)
    bootstrap_details.to_csv(paths["bootstrap"], index=False)
    permutation_details.to_csv(paths["permutations"], index=False)

    patient_condition_counts = val_meta.groupby(groups)[args.condition_col].nunique()
    result = {
        "schema_version": "1.0",
        "cohorts": {"development": args.development_cohort, "validation": args.validation_cohort},
        "independent_cohorts": args.development_cohort != args.validation_cohort,
        "endpoint": f"{args.case_label}_vs_{args.control_label}",
        "validation_sample_counts": {
            "n_samples": int(len(val_y)),
            "n_case": int(val_y.sum()),
            "n_control": int((1 - val_y).sum()),
            "min_class_n": int(min(val_y.sum(), (1 - val_y).sum())),
            "n_patients": int(len(np.unique(groups))),
            "n_paired_patients": int((patient_condition_counts > 1).sum()),
            "cluster_id_source": cluster_source,
        },
        "feature_lock": {
            "n_development_measured_genes": int(dev_X.shape[1]),
            "n_validation_measured_genes": int(val_X.shape[1]),
            "n_common_genes": int(len(common_genes)),
            "n_selected_genes": int(len(selected_genes)),
            "selection_method": "development_cohort_abs_log2fc",
            "selection_feature_universe": "all_development_genes_before_validation_availability_check",
            "missing_locked_genes_in_validation": missing_locked_genes,
            "selection_used_validation_labels": False,
        },
        "leakage_controls": {
            "validation_labels_used_for_feature_selection": False,
            "validation_labels_used_for_hyperparameter_selection": False,
            "validation_samples_used_for_scaler_or_pca_fit": False,
            "validation_samples_used_for_model_fit": False,
            "development_configuration_locked_before_validation": True,
            "validation_de_signature_generated": False,
            "shared_gene_universe_normalization_verified": True,
        },
        "harmonization": {
            "manifest_path": str(args.harmonization_manifest),
            "method": harmonization_manifest.get("method"),
            "labels_used": bool(harmonization_manifest.get("labels_used")),
            "n_shared_genes": (harmonization_manifest.get("diagnostics") or {}).get("n_shared_genes"),
        },
        "development_lock": {
            "verdict_path": str(args.development_verdict),
            "manifest_path": str(args.development_manifest),
            "best_quantum_model_from_development": development_verdict.get("best_quantum_model"),
            "classifier_feature_universe": development_manifest.get("classifier_feature_universe"),
        },
        "verdict": verdict,
        "interpretation": (
            "This is a cross-study technical validation of tumor-versus-normal discrimination. "
            "It does not establish clinical utility, causal biomarkers, or quantum advantage unless the predeclared "
            "external delta, confidence interval, and permutation gates all pass."
        ),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["verdict"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "inputs": {
            "development_counts": {"path": args.development_counts, "sha256": _sha256(args.development_counts)},
            "development_metadata": {"path": args.development_metadata, "sha256": _sha256(args.development_metadata)},
            "development_verdict": {"path": args.development_verdict, "sha256": _sha256(args.development_verdict)},
            "development_manifest": {"path": args.development_manifest, "sha256": _sha256(args.development_manifest)},
            "harmonization_manifest": {
                "path": args.harmonization_manifest,
                "sha256": _sha256(args.harmonization_manifest),
            },
            "validation_counts": {"path": args.validation_counts, "sha256": _sha256(args.validation_counts)},
            "validation_metadata": {"path": args.validation_metadata, "sha256": _sha256(args.validation_metadata)},
        },
        "parameters": {
            "top_genes": int(args.top_genes),
            "bootstrap": int(args.bootstrap),
            "permutations": int(args.permutations),
            "random_state": int(args.random_state),
            "min_delta": float(args.min_delta),
            "alpha": float(args.alpha),
        },
        "outputs": result["outputs"],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_report(result, paths["report"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-counts", required=True)
    parser.add_argument("--development-metadata", required=True)
    parser.add_argument("--development-verdict", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--harmonization-manifest", required=True)
    parser.add_argument("--validation-counts", required=True)
    parser.add_argument("--validation-metadata", required=True)
    parser.add_argument("--development-cohort", required=True)
    parser.add_argument("--validation-cohort", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--patient-id-col", default="patient_id")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--case-label", default="disease")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--top-genes", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--min-delta", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    result = run_external_validation(args)
    print(json.dumps({"verdict": result["verdict"], "outputs": result["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
