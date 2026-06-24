#!/usr/bin/env python3
"""Benchmark whether quantum models add value for RNA-seq evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.feature_fusion import fuse_evidence


@dataclass
class ClassifierResult:
    model: str
    y_true: np.ndarray
    y_score: np.ndarray
    y_pred: np.ndarray
    status: str = "ok"
    message: str = ""
    qml_dim: Optional[int] = None
    qsvc_reps: Optional[int] = None


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_int_list(value: str | int | Iterable[int]) -> List[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    else:
        parsed = [int(v) for v in value]
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    if any(v <= 0 for v in parsed):
        raise ValueError(f"Values must be positive integers: {parsed}")
    return parsed


def load_signature_genes(signature_path: str | Path, top_n: int) -> List[str]:
    """Return ordered signature genes, preferring ranked_genes then up/down lists."""

    sig = _read_json(signature_path)
    genes: List[str] = []
    for item in sig.get("ranked_genes", []):
        gene = str(item.get("gene", "")).strip()
        if gene and gene not in genes:
            genes.append(gene)
    for key in ("up_genes", "down_genes"):
        for gene in sig.get(key, []):
            gene = str(gene).strip()
            if gene and gene not in genes:
                genes.append(gene)
    return genes[:top_n]


def load_expression_inputs(
    normalized_counts_path: str | Path,
    metadata_path: str | Path,
    *,
    sample_id_col: str,
    condition_col: str,
    case_label: str,
    control_label: str,
    signature_genes: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, List[str]]:
    """Load normalized expression and metadata.

    When ``signature_genes`` is omitted, return the complete measured-gene
    matrix. Classifier workflows should use this mode so feature selection is
    performed inside each training fold rather than leaking a cohort-wide DE
    signature into cross-validation.
    """

    expr = pd.read_csv(normalized_counts_path)
    if sample_id_col not in expr.columns:
        first_col = expr.columns[0]
        expr = expr.rename(columns={first_col: sample_id_col})
    expr[sample_id_col] = expr[sample_id_col].astype(str)
    expr = expr.set_index(sample_id_col, drop=False)

    meta = pd.read_csv(metadata_path)
    if sample_id_col not in meta.columns:
        raise ValueError(f"Metadata must include '{sample_id_col}'.")
    if condition_col not in meta.columns:
        raise ValueError(f"Metadata must include '{condition_col}'.")
    meta[sample_id_col] = meta[sample_id_col].astype(str)
    meta = meta.set_index(sample_id_col, drop=False)

    missing_meta = sorted(set(expr.index) - set(meta.index))
    extra_meta = sorted(set(meta.index) - set(expr.index))
    if missing_meta or extra_meta:
        raise ValueError(
            "Normalized counts samples must exactly match metadata IDs. "
            f"Missing metadata: {missing_meta[:5]}; metadata without counts: {extra_meta[:5]}"
        )
    meta = meta.loc[expr.index].copy()

    labels = set(meta[condition_col].astype(str))
    missing = {case_label, control_label} - labels
    if missing:
        raise ValueError(f"Missing required labels in '{condition_col}': {sorted(missing)}")

    gene_columns = [col for col in expr.columns if col != sample_id_col]
    if not gene_columns:
        raise ValueError("Normalized counts must include at least one gene column.")
    expr_all = expr[gene_columns].apply(pd.to_numeric, errors="raise")
    if signature_genes is None:
        available = list(map(str, expr_all.columns))
    else:
        available = [g for g in signature_genes if g in expr_all.columns]
        if not available:
            raise ValueError("None of the signature genes are present in normalized counts.")

    y = (meta[condition_col].astype(str).to_numpy() == case_label).astype(int)
    return expr_all[available].astype(float), meta, y, available


def _cv_splitter(y: np.ndarray, requested_folds: int):
    class_counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(class_counts.min())
    if len(y) < 10 or min_class < 3:
        return LeaveOneOut(), "leave_one_out"
    folds = max(2, min(int(requested_folds), min_class))
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=42), f"stratified_{folds}_fold"


def _probability_metrics(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["pr_auc"] = float("nan")
    return metrics


def _score_to_pred(scores: np.ndarray) -> np.ndarray:
    return (np.asarray(scores, dtype=float) >= 0.5).astype(int)


def _select_fold_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    top_n: int,
) -> np.ndarray:
    """Select top genes using training samples only to avoid leakage."""

    case = X_train[y_train == 1]
    control = X_train[y_train == 0]
    if case.size == 0 or control.size == 0:
        raise ValueError("Each training fold must contain both classes for feature selection.")
    case_mean = case.mean(axis=0)
    control_mean = control.mean(axis=0)
    scores = np.abs(np.log2((case_mean + 1.0) / (control_mean + 1.0)))
    order = np.argsort(scores)[::-1]
    k = max(1, min(int(top_n), len(feature_names)))
    return order[:k]


def _classical_cv_results(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    *,
    feature_names: List[str],
    top_n: int,
    random_state: int,
) -> List[ClassifierResult]:
    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
    }
    results: List[ClassifierResult] = []
    for name, model in models.items():
        scores = np.zeros(len(y), dtype=float)
        preds = np.zeros(len(y), dtype=int)
        for train_idx, test_idx in cv.split(X, y):
            selected = _select_fold_features(X[train_idx], y[train_idx], feature_names, top_n)
            model.fit(X[train_idx][:, selected], y[train_idx])
            scores[test_idx] = model.predict_proba(X[test_idx][:, selected])[:, 1]
            preds[test_idx] = model.predict(X[test_idx][:, selected])
        results.append(ClassifierResult(name, y.copy(), scores, preds))
    return results


def _prepare_quantum_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    *,
    qml_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale/PCA expression features into a fixed quantum feature-map dimension."""

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    max_components = min(qml_dim, X_train_s.shape[0] - 1, X_train_s.shape[1])
    if max_components >= 1:
        pca = PCA(n_components=max_components, random_state=42)
        X_train_q = pca.fit_transform(X_train_s)
        X_test_q = pca.transform(X_test_s)
    else:
        X_train_q = X_train_s[:, :1]
        X_test_q = X_test_s[:, :1]

    if X_train_q.shape[1] < qml_dim:
        pad_train = np.zeros((X_train_q.shape[0], qml_dim - X_train_q.shape[1]))
        pad_test = np.zeros((X_test_q.shape[0], qml_dim - X_test_q.shape[1]))
        X_train_q = np.hstack([X_train_q, pad_train])
        X_test_q = np.hstack([X_test_q, pad_test])

    mm = MinMaxScaler(feature_range=(0.0, math.pi))
    return mm.fit_transform(X_train_q), mm.transform(X_test_q)


def _fit_local_qsvc(X_train: np.ndarray, y_train: np.ndarray, *, qml_dim: int, reps: int):
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel

    feature_map = ZZFeatureMap(feature_dimension=qml_dim, reps=reps, entanglement="linear")
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    model = QSVC(quantum_kernel=kernel)
    model.fit(X_train, y_train)
    return model


def _decision_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-raw))
    pred = np.asarray(model.predict(X), dtype=float)
    return pred


def _quantum_cv_result(
    X: np.ndarray,
    y: np.ndarray,
    cv,
    *,
    qml_dim: int,
    qsvc_reps: int,
    quantum_mode: str,
    allow_ibm_submit: bool,
    feature_names: List[str],
    top_n: int,
) -> ClassifierResult:
    if quantum_mode == "ibm" and not allow_ibm_submit:
        return ClassifierResult(
            f"qsvc_quantum_dim{qml_dim}_reps{qsvc_reps}",
            y.copy(),
            np.full(len(y), np.nan),
            np.zeros(len(y), dtype=int),
            status="skipped",
            message="IBM mode requires --allow-ibm-submit; local mode remains the default runnable benchmark.",
            qml_dim=qml_dim,
            qsvc_reps=qsvc_reps,
        )
    if quantum_mode == "ibm":
        return ClassifierResult(
            f"qsvc_quantum_dim{qml_dim}_reps{qsvc_reps}",
            y.copy(),
            np.full(len(y), np.nan),
            np.zeros(len(y), dtype=int),
            status="skipped",
            message="IBM submission is gated for this benchmark; run local first, then wire selected full-train jobs.",
            qml_dim=qml_dim,
            qsvc_reps=qsvc_reps,
        )

    scores = np.zeros(len(y), dtype=float)
    preds = np.zeros(len(y), dtype=int)
    try:
        for train_idx, test_idx in cv.split(X, y):
            selected = _select_fold_features(X[train_idx], y[train_idx], feature_names, top_n)
            X_train_q, X_test_q = _prepare_quantum_features(
                X[train_idx][:, selected],
                X[test_idx][:, selected],
                qml_dim=qml_dim,
            )
            model = _fit_local_qsvc(X_train_q, y[train_idx], qml_dim=qml_dim, reps=qsvc_reps)
            fold_scores = _decision_scores(model, X_test_q)
            scores[test_idx] = fold_scores
            preds[test_idx] = _score_to_pred(fold_scores)
    except Exception as exc:
        return ClassifierResult(
            f"qsvc_quantum_dim{qml_dim}_reps{qsvc_reps}",
            y.copy(),
            np.full(len(y), np.nan),
            np.zeros(len(y), dtype=int),
            status="failed",
            message=str(exc),
            qml_dim=qml_dim,
            qsvc_reps=qsvc_reps,
        )
    return ClassifierResult(
        f"qsvc_quantum_dim{qml_dim}_reps{qsvc_reps}",
        y.copy(),
        scores,
        preds,
        qml_dim=qml_dim,
        qsvc_reps=qsvc_reps,
    )


def run_classifier_benchmark(
    X_df: pd.DataFrame,
    y: np.ndarray,
    *,
    cv_folds: int,
    top_n: int,
    qml_dims: Iterable[int],
    qsvc_reps_list: Iterable[int],
    quantum_mode: str,
    allow_ibm_submit: bool,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    X = X_df.to_numpy(dtype=float)
    feature_names = list(map(str, X_df.columns))
    cv, cv_name = _cv_splitter(y, cv_folds)
    qml_dims = list(qml_dims)
    qsvc_reps_list = list(qsvc_reps_list)
    results = _classical_cv_results(
        X,
        y,
        cv,
        feature_names=feature_names,
        top_n=top_n,
        random_state=random_state,
    )
    for qml_dim in qml_dims:
        for qsvc_reps in qsvc_reps_list:
            results.append(
                _quantum_cv_result(
                    X,
                    y,
                    cv,
                    qml_dim=qml_dim,
                    qsvc_reps=qsvc_reps,
                    quantum_mode=quantum_mode,
                    allow_ibm_submit=allow_ibm_submit,
                    feature_names=feature_names,
                    top_n=top_n,
                )
            )

    best_qsvc = None
    ok_qsvc = []
    for result in results:
        if result.model.startswith("qsvc_quantum") and result.status == "ok":
            metrics = _probability_metrics(result.y_true, result.y_score, result.y_pred)
            key = metrics["roc_auc"] if not math.isnan(metrics["roc_auc"]) else metrics["balanced_accuracy"]
            ok_qsvc.append((key, result.model))
    if ok_qsvc:
        best_qsvc = max(ok_qsvc, key=lambda item: item[0])[1]

    results.append(
        _select_best_quantum_result(
            [r for r in results if r.model.startswith("qsvc_quantum")],
            best_qsvc,
        )
    )

    metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    for result in results:
        if result.status == "ok":
            metrics = _probability_metrics(result.y_true, result.y_score, result.y_pred)
        else:
            metrics = {"accuracy": np.nan, "balanced_accuracy": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
        metric_rows.append(
            {
                "model": result.model,
                "status": result.status,
                "message": result.message,
                "qml_dim": result.qml_dim,
                "qsvc_reps": result.qsvc_reps,
                **metrics,
            }
        )
        for idx, (yt, ys, yp) in enumerate(zip(result.y_true, result.y_score, result.y_pred)):
            pred_rows.append(
                {
                    "sample_index": idx,
                    "model": result.model,
                    "y_true": int(yt),
                    "y_score": None if np.isnan(ys) else float(ys),
                    "y_pred": int(yp),
                }
            )

    context = {
        "cv": cv_name,
        "n_samples": int(len(y)),
        "n_case": int(y.sum()),
        "n_control": int((1 - y).sum()),
        "n_features": int(X.shape[1]),
        "classifier_top_genes_per_fold": int(min(top_n, X.shape[1])),
        "feature_selection": "train_fold_abs_log2fc",
        "qml_dims": [int(v) for v in qml_dims],
        "qsvc_reps_list": [int(v) for v in qsvc_reps_list],
        "best_quantum_model": best_qsvc,
        "quantum_mode": quantum_mode,
    }
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_rows), context


def _select_best_quantum_result(results: List[ClassifierResult], best_model: Optional[str]) -> ClassifierResult:
    if best_model is None:
        message = "No successful QSVC configuration."
        if results:
            message = "; ".join(sorted({r.message for r in results if r.message})) or message
        return ClassifierResult(
            "qsvc_quantum",
            results[0].y_true.copy() if results else np.array([], dtype=int),
            np.full(len(results[0].y_true), np.nan) if results else np.array([], dtype=float),
            np.zeros(len(results[0].y_true), dtype=int) if results else np.array([], dtype=int),
            status="failed" if any(r.status == "failed" for r in results) else "skipped",
            message=message,
        )
    selected = next(r for r in results if r.model == best_model)
    return ClassifierResult(
        "qsvc_quantum",
        selected.y_true.copy(),
        selected.y_score.copy(),
        selected.y_pred.copy(),
        status=selected.status,
        message=f"best_config={selected.model}",
        qml_dim=selected.qml_dim,
        qsvc_reps=selected.qsvc_reps,
    )


def _best_classical_metric(metrics_df: pd.DataFrame, metric: str) -> float:
    classical = metrics_df[metrics_df["model"].isin(["logistic_regression", "rbf_svm"])]
    vals = pd.to_numeric(classical[metric], errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else float("nan")


def _best_classical_model(metrics_df: pd.DataFrame, metric: str = "roc_auc") -> Optional[str]:
    classical = metrics_df[metrics_df["model"].isin(["logistic_regression", "rbf_svm"])].copy()
    if classical.empty:
        return None
    classical["_primary"] = pd.to_numeric(classical[metric], errors="coerce")
    if classical["_primary"].isna().all():
        classical["_primary"] = pd.to_numeric(classical["balanced_accuracy"], errors="coerce")
    classical = classical.dropna(subset=["_primary"])
    if classical.empty:
        return None
    return str(classical.sort_values("_primary", ascending=False).iloc[0]["model"])


def _bootstrap_metric_delta(
    predictions_df: Optional[pd.DataFrame],
    *,
    quantum_model: str,
    classical_model: Optional[str],
    metric: str,
    n_bootstrap: int,
    random_state: int,
) -> Dict[str, Any]:
    if predictions_df is None or classical_model is None or predictions_df.empty:
        return {"available": False, "reason": "predictions unavailable"}

    q = predictions_df[predictions_df["model"] == quantum_model].sort_values("sample_index")
    c = predictions_df[predictions_df["model"] == classical_model].sort_values("sample_index")
    if q.empty or c.empty or len(q) != len(c):
        return {"available": False, "reason": "matching prediction rows unavailable"}
    y = q["y_true"].to_numpy(dtype=int)
    q_score = pd.to_numeric(q["y_score"], errors="coerce").to_numpy(dtype=float)
    c_score = pd.to_numeric(c["y_score"], errors="coerce").to_numpy(dtype=float)
    q_pred = q["y_pred"].to_numpy(dtype=int)
    c_pred = c["y_pred"].to_numpy(dtype=int)
    if np.isnan(q_score).any() or np.isnan(c_score).any():
        return {"available": False, "reason": "prediction scores contain NaN"}

    rng = np.random.default_rng(random_state)
    deltas: List[float] = []
    skipped = 0
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            skipped += 1
            continue
        if metric == "roc_auc":
            q_metric = roc_auc_score(y[idx], q_score[idx])
            c_metric = roc_auc_score(y[idx], c_score[idx])
        elif metric == "balanced_accuracy":
            q_metric = balanced_accuracy_score(y[idx], q_pred[idx])
            c_metric = balanced_accuracy_score(y[idx], c_pred[idx])
        else:
            raise ValueError(f"Unsupported bootstrap metric: {metric}")
        deltas.append(float(q_metric - c_metric))

    if not deltas:
        return {"available": False, "reason": "no valid bootstrap resamples"}
    arr = np.asarray(deltas, dtype=float)
    return {
        "available": True,
        "metric": metric,
        "n_bootstrap": int(n_bootstrap),
        "n_valid": int(len(arr)),
        "n_skipped": int(skipped),
        "delta_mean": float(arr.mean()),
        "delta_ci95_low": float(np.quantile(arr, 0.025)),
        "delta_ci95_high": float(np.quantile(arr, 0.975)),
        "prob_delta_gt_0": float(np.mean(arr > 0.0)),
    }


def _classifier_evidence_context(predictions_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if predictions_df is None or predictions_df.empty:
        return {
            "classifier_evidence_grade": "not_evaluable",
            "classifier_sample_size_warning": True,
            "classifier_sample_size_reason": "prediction rows unavailable",
        }

    rows = predictions_df[predictions_df["model"] == "qsvc_quantum"].sort_values("sample_index")
    if rows.empty:
        rows = predictions_df.sort_values(["model", "sample_index"]).drop_duplicates("sample_index")
    y = rows["y_true"].to_numpy(dtype=int)
    class_counts = np.bincount(y, minlength=2)
    n_samples = int(len(y))
    n_case = int(class_counts[1])
    n_control = int(class_counts[0])
    min_class = min(n_case, n_control)

    if n_samples < 20 or min_class < 10:
        grade = "pilot_underpowered"
        warning = True
        reason = "fewer than 20 samples or fewer than 10 samples in the smallest class"
    elif n_samples < 60 or min_class < 25:
        grade = "screening"
        warning = True
        reason = "sample size is suitable for screening, not strong biological claims"
    else:
        grade = "analysis_ready"
        warning = False
        reason = "sample size passes the default screening thresholds"

    return {
        "classifier_evidence_grade": grade,
        "classifier_sample_size_warning": warning,
        "classifier_sample_size_reason": reason,
        "classifier_n_samples": n_samples,
        "classifier_n_case": n_case,
        "classifier_n_control": n_control,
        "classifier_min_class_n": int(min_class),
    }


def _score_label_permutation_auc(
    predictions_df: Optional[pd.DataFrame],
    *,
    model: Optional[str],
    n_permutations: int,
    random_state: int,
) -> Dict[str, Any]:
    """Permute held-out labels against fixed scores to contextualize ROC-AUC."""

    if predictions_df is None or predictions_df.empty or not model:
        return {"available": False, "reason": "predictions unavailable"}
    rows = predictions_df[predictions_df["model"] == model].sort_values("sample_index")
    if rows.empty:
        return {"available": False, "reason": f"prediction rows unavailable for {model}"}
    y = rows["y_true"].to_numpy(dtype=int)
    scores = pd.to_numeric(rows["y_score"], errors="coerce").to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        return {"available": False, "reason": "observed labels contain fewer than two classes"}
    if np.isnan(scores).any():
        return {"available": False, "reason": "prediction scores contain NaN"}
    if n_permutations <= 0:
        return {"available": False, "reason": "permutation count is zero"}

    observed = float(roc_auc_score(y, scores))
    rng = np.random.default_rng(random_state)
    null = np.zeros(int(n_permutations), dtype=float)
    for idx in range(int(n_permutations)):
        null[idx] = roc_auc_score(rng.permutation(y), scores)
    return {
        "available": True,
        "method": "score_label_permutation",
        "model": model,
        "n_permutations": int(n_permutations),
        "observed_roc_auc": observed,
        "null_mean": float(null.mean()),
        "null_ci95_low": float(np.quantile(null, 0.025)),
        "null_ci95_high": float(np.quantile(null, 0.975)),
        "p_value_auc_ge_observed": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
        "note": "Labels are permuted against fixed held-out scores; this is not a full retraining permutation test.",
    }


def run_full_retraining_permutation(
    X_df: pd.DataFrame,
    y: np.ndarray,
    observed_metrics: pd.DataFrame,
    *,
    cv_folds: int,
    top_n: int,
    qml_dims: Iterable[int],
    qsvc_reps_list: Iterable[int],
    quantum_mode: str,
    allow_ibm_submit: bool,
    n_permutations: int,
    random_state: int,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Rerun the full CV workflow on shuffled labels as a leakage-aware null."""

    if n_permutations <= 0:
        return {"available": False, "reason": "full retraining permutation count is zero"}, pd.DataFrame()
    if quantum_mode != "local":
        return {"available": False, "reason": "full retraining permutations are only supported in local mode"}, pd.DataFrame()

    q_row = observed_metrics[observed_metrics["model"] == "qsvc_quantum"]
    if q_row.empty or str(q_row.iloc[0]["status"]) != "ok":
        return {"available": False, "reason": "observed qsvc_quantum result is not evaluable"}, pd.DataFrame()
    observed_quantum_auc = float(q_row.iloc[0]["roc_auc"])
    observed_classical_auc = _best_classical_metric(observed_metrics, "roc_auc")
    if math.isnan(observed_quantum_auc) or math.isnan(observed_classical_auc):
        return {"available": False, "reason": "observed ROC-AUC is not evaluable"}, pd.DataFrame()

    qml_dims = [int(v) for v in qml_dims]
    qsvc_reps_list = [int(v) for v in qsvc_reps_list]
    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, Any]] = []
    for permutation_idx in range(int(n_permutations)):
        y_perm = rng.permutation(y)
        metrics_perm, _, context = run_classifier_benchmark(
            X_df,
            y_perm,
            cv_folds=cv_folds,
            top_n=top_n,
            qml_dims=qml_dims,
            qsvc_reps_list=qsvc_reps_list,
            quantum_mode=quantum_mode,
            allow_ibm_submit=allow_ibm_submit,
            random_state=random_state + permutation_idx + 1,
        )
        q_perm = metrics_perm[metrics_perm["model"] == "qsvc_quantum"]
        q_auc = float(q_perm.iloc[0]["roc_auc"]) if not q_perm.empty and str(q_perm.iloc[0]["status"]) == "ok" else float("nan")
        c_auc = _best_classical_metric(metrics_perm, "roc_auc")
        rows.append(
            {
                "permutation": permutation_idx,
                "best_classical_roc_auc": c_auc,
                "quantum_roc_auc": q_auc,
                "delta_roc_auc": q_auc - c_auc if not math.isnan(q_auc) and not math.isnan(c_auc) else float("nan"),
                "best_quantum_model": context.get("best_quantum_model"),
            }
        )

    details = pd.DataFrame(rows)
    valid = details.dropna(subset=["best_classical_roc_auc", "quantum_roc_auc"]).copy()
    if valid.empty:
        return {"available": False, "reason": "no valid full retraining permutation runs"}, details

    observed_delta = observed_quantum_auc - observed_classical_auc
    summary = {
        "available": True,
        "method": "full_retraining_label_permutation",
        "n_permutations": int(n_permutations),
        "n_valid": int(len(valid)),
        "observed_quantum_roc_auc": observed_quantum_auc,
        "observed_best_classical_roc_auc": observed_classical_auc,
        "observed_delta_roc_auc": float(observed_delta),
        "null_quantum_auc_mean": float(valid["quantum_roc_auc"].mean()),
        "null_quantum_auc_ci95_low": float(valid["quantum_roc_auc"].quantile(0.025)),
        "null_quantum_auc_ci95_high": float(valid["quantum_roc_auc"].quantile(0.975)),
        "null_best_classical_auc_mean": float(valid["best_classical_roc_auc"].mean()),
        "null_best_classical_auc_ci95_low": float(valid["best_classical_roc_auc"].quantile(0.025)),
        "null_best_classical_auc_ci95_high": float(valid["best_classical_roc_auc"].quantile(0.975)),
        "null_delta_auc_mean": float(valid["delta_roc_auc"].mean()),
        "null_delta_auc_ci95_low": float(valid["delta_roc_auc"].quantile(0.025)),
        "null_delta_auc_ci95_high": float(valid["delta_roc_auc"].quantile(0.975)),
        "p_value_quantum_auc_ge_observed": float(
            (1 + np.sum(valid["quantum_roc_auc"].to_numpy(dtype=float) >= observed_quantum_auc)) / (len(valid) + 1)
        ),
        "p_value_best_classical_auc_ge_observed": float(
            (1 + np.sum(valid["best_classical_roc_auc"].to_numpy(dtype=float) >= observed_classical_auc)) / (len(valid) + 1)
        ),
        "p_value_delta_auc_ge_observed": float(
            (1 + np.sum(valid["delta_roc_auc"].to_numpy(dtype=float) >= observed_delta)) / (len(valid) + 1)
        ),
        "note": "Labels are permuted before feature selection and cross-validation, so this reruns the full classifier workflow.",
    }
    return summary, details


def build_value_verdict(
    metrics_df: pd.DataFrame,
    *,
    min_delta: float,
    predictions_df: Optional[pd.DataFrame] = None,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    q_rows = metrics_df[metrics_df["model"] == "qsvc_quantum"]
    if q_rows.empty or str(q_rows.iloc[0]["status"]) != "ok":
        payload = {
            "classifier_verdict": "quantum_not_evaluable",
            "quantum_adds_value": False,
            "reason": str(q_rows.iloc[0]["message"]) if not q_rows.empty else "qsvc_quantum row missing",
        }
        payload.update(_classifier_evidence_context(predictions_df))
        return payload

    quantum_auc = float(q_rows.iloc[0]["roc_auc"])
    classical_auc = _best_classical_metric(metrics_df, "roc_auc")
    quantum_bal_acc = float(q_rows.iloc[0]["balanced_accuracy"])
    classical_bal_acc = _best_classical_metric(metrics_df, "balanced_accuracy")
    best_classical = _best_classical_model(metrics_df)

    primary_delta = quantum_auc - classical_auc if not math.isnan(quantum_auc) and not math.isnan(classical_auc) else quantum_bal_acc - classical_bal_acc
    if primary_delta > min_delta:
        verdict = "quantum_adds_value"
        adds_value = True
    elif primary_delta < -min_delta:
        verdict = "quantum_underperforms_classical"
        adds_value = False
    else:
        verdict = "quantum_matches_classical_within_delta"
        adds_value = False

    verdict_payload = {
        "classifier_verdict": verdict,
        "quantum_adds_value": adds_value,
        "min_delta": float(min_delta),
        "best_quantum_model": str(q_rows.iloc[0].get("message", "")).replace("best_config=", ""),
        "best_classical_model": best_classical,
        "best_quantum_qml_dim": None if pd.isna(q_rows.iloc[0].get("qml_dim")) else int(q_rows.iloc[0].get("qml_dim")),
        "best_quantum_qsvc_reps": None if pd.isna(q_rows.iloc[0].get("qsvc_reps")) else int(q_rows.iloc[0].get("qsvc_reps")),
        "quantum_roc_auc": quantum_auc,
        "best_classical_roc_auc": classical_auc,
        "delta_roc_auc": None if math.isnan(quantum_auc) or math.isnan(classical_auc) else float(quantum_auc - classical_auc),
        "quantum_balanced_accuracy": quantum_bal_acc,
        "best_classical_balanced_accuracy": classical_bal_acc,
        "delta_balanced_accuracy": float(quantum_bal_acc - classical_bal_acc),
    }
    verdict_payload["bootstrap_delta_roc_auc"] = _bootstrap_metric_delta(
        predictions_df,
        quantum_model="qsvc_quantum",
        classical_model=best_classical,
        metric="roc_auc",
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    verdict_payload["bootstrap_delta_balanced_accuracy"] = _bootstrap_metric_delta(
        predictions_df,
        quantum_model="qsvc_quantum",
        classical_model=best_classical,
        metric="balanced_accuracy",
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )
    verdict_payload.update(_classifier_evidence_context(predictions_df))
    verdict_payload["permutation_auc_best_quantum"] = _score_label_permutation_auc(
        predictions_df,
        model="qsvc_quantum",
        n_permutations=n_permutations,
        random_state=random_state,
    )
    verdict_payload["permutation_auc_best_classical"] = _score_label_permutation_auc(
        predictions_df,
        model=best_classical,
        n_permutations=n_permutations,
        random_state=random_state + 1,
    )
    return verdict_payload


def _signature_lfc_vector(signature_path: str | Path, genes: List[str]) -> np.ndarray:
    sig = _read_json(signature_path)
    by_gene = {str(r.get("gene")): float(r.get("logfc", 0.0)) for r in sig.get("ranked_genes", [])}
    vec = np.array([by_gene.get(g, 0.0) for g in genes], dtype=float)
    if np.allclose(vec, 0.0):
        vec = np.ones(len(genes), dtype=float)
    return vec


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.5
    return float((np.dot(a, b) / denom + 1.0) / 2.0)


def build_demo_candidate_profiles(signature_path: str | Path, genes: List[str]) -> pd.DataFrame:
    """Create a deterministic signature-derived challenge panel, not real drug evidence."""

    lfc = _signature_lfc_vector(signature_path, genes)
    rng = np.random.default_rng(123)
    profiles = [
        ("demo_anti_signature_strong", "anti-signature strong", -1.0 * lfc, 1),
        ("demo_anti_signature_partial", "anti-signature partial", -0.45 * lfc, 1),
        ("demo_same_signature", "same-signature", 1.0 * lfc, 0),
        ("demo_noisy_same_signature", "noisy same-signature", 0.65 * lfc + rng.normal(0, 0.05, len(lfc)), 0),
        ("demo_neutral", "neutral", np.zeros(len(lfc)), 0),
        ("demo_noisy_reversal", "noisy reversal", -0.7 * lfc + rng.normal(0, 0.05, len(lfc)), 1),
    ]
    rows: List[Dict[str, Any]] = []
    for idx, (candidate_id, compound, effect, expected) in enumerate(profiles):
        row: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "compound": compound,
            "disease": "signature_challenge",
            "compound_hetionet_id": None,
            "disease_hetionet_id": None,
            "kg_rotate_score": 0.45 + idx * 0.05,
            "kg_complex_score": 0.48 + idx * 0.03,
            "graph_topology_score": 0.40 + idx * 0.04,
            "expected_reverser": expected,
        }
        row.update({gene: float(value) for gene, value in zip(genes, effect)})
        rows.append(row)
    return pd.DataFrame(rows)


def load_cmap_candidate_profiles(
    path: str | Path,
    genes: List[str],
    *,
    min_gene_overlap: int,
) -> pd.DataFrame:
    """Load real tidy compound perturbation scores into the ranking profile format."""

    raw = pd.read_csv(path)
    if {"compound", "gene", "score"}.issubset(raw.columns):
        tidy = raw[["compound", "gene", "score"]].copy()
        tidy["score"] = pd.to_numeric(tidy["score"], errors="coerce")
        tidy = tidy.dropna(subset=["compound", "gene", "score"])
    else:
        from perturbation_layer.cmap_loader import load_cmap_csv

        tidy = load_cmap_csv(str(path))[["compound", "gene", "score"]]

    tidy["compound"] = tidy["compound"].astype(str)
    tidy["gene"] = tidy["gene"].astype(str)
    overlap = tidy[tidy["gene"].isin(genes)].copy()
    if overlap.empty:
        raise ValueError("CMap signatures have no overlap with selected RNA-seq signature genes.")

    rows: List[Dict[str, Any]] = []
    for compound, sub in overlap.groupby("compound"):
        scores = {gene: 0.0 for gene in genes}
        compound_scores = sub.groupby("gene")["score"].mean()
        n_overlap = int(compound_scores.index.isin(genes).sum())
        if n_overlap < min_gene_overlap:
            continue
        for gene, score in compound_scores.items():
            if gene in scores:
                scores[gene] = float(score)
        row: Dict[str, Any] = {
            "candidate_id": compound,
            "compound": compound,
            "disease": "rnaseq_signature",
            "compound_hetionet_id": None,
            "disease_hetionet_id": None,
            "kg_rotate_score": 0.5,
            "kg_complex_score": 0.5,
            "graph_topology_score": 0.5,
            "profile_gene_overlap": n_overlap,
        }
        row.update(scores)
        rows.append(row)

    if not rows:
        raise ValueError(
            "No CMap compounds met the profile gene overlap threshold "
            f"({min_gene_overlap}). Lower --min-profile-gene-overlap or provide richer profiles."
        )
    return pd.DataFrame(rows)


def _load_gene_map(path: Optional[str], genes: List[str]) -> Dict[str, str]:
    if not path:
        return {gene: gene for gene in genes}
    df = pd.read_csv(path)
    id_col = next((c for c in ["gene_id", "ensembl_id", "feature_id", "gene"] if c in df.columns), None)
    symbol_col = next((c for c in ["gene_symbol", "symbol", "gene_name", "profile_gene"] if c in df.columns), None)
    if id_col is None or symbol_col is None:
        raise ValueError(f"Gene map must include ID and symbol columns. Found: {list(df.columns)}")
    mapping = {
        str(row[id_col]): str(row[symbol_col])
        for _, row in df.dropna(subset=[id_col, symbol_col]).iterrows()
    }
    return {gene: mapping.get(gene, gene) for gene in genes}


def _creeds_gene_scores(items: list) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for item in items:
        if isinstance(item, list) and item:
            gene = str(item[0]).upper()
            score = float(item[1]) if len(item) > 1 else 1.0
        else:
            gene = str(item).upper()
            score = 1.0
        scores[gene] = score
    return scores


def load_creeds_candidate_profiles(
    path: str | Path,
    genes: List[str],
    *,
    gene_map_path: Optional[str],
    organism: str,
    min_gene_overlap: int,
    max_profiles: int,
) -> pd.DataFrame:
    """Load public CREEDS drug perturbation JSON into ranking profile format."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    feature_to_symbol = _load_gene_map(gene_map_path, genes)
    feature_to_profile_gene = {feature: symbol.upper() for feature, symbol in feature_to_symbol.items()}
    rows: List[Dict[str, Any]] = []
    organism_lc = organism.lower()

    for idx, record in enumerate(data):
        if organism_lc and str(record.get("organism", "")).lower() != organism_lc:
            continue
        score_by_symbol: Dict[str, float] = {}
        score_by_symbol.update(_creeds_gene_scores(record.get("down_genes", [])))
        score_by_symbol.update(_creeds_gene_scores(record.get("up_genes", [])))

        values: Dict[str, float] = {}
        overlap = 0
        for feature in genes:
            symbol = feature_to_profile_gene.get(feature, feature.upper())
            value = float(score_by_symbol.get(symbol, 0.0))
            if symbol in score_by_symbol:
                overlap += 1
            values[feature] = value
        if overlap < min_gene_overlap:
            continue

        drug_name = str(record.get("drug_name") or record.get("id") or f"creeds_{idx}")
        candidate_id = "|".join(
            [
                str(record.get("id", f"creeds:{idx}")),
                drug_name,
                str(record.get("geo_id", "")),
                str(record.get("cell_type", ""))[:80],
            ]
        )
        row: Dict[str, Any] = {
            "candidate_id": candidate_id,
            "compound": drug_name,
            "disease": "rnaseq_signature",
            "compound_hetionet_id": record.get("drugbank_id") or None,
            "disease_hetionet_id": None,
            "kg_rotate_score": 0.5,
            "kg_complex_score": 0.5,
            "graph_topology_score": 0.5,
            "profile_gene_overlap": overlap,
            "creeds_id": record.get("id"),
            "geo_id": record.get("geo_id"),
            "cell_type": record.get("cell_type"),
            "organism": record.get("organism"),
        }
        row.update(values)
        rows.append(row)

    if not rows:
        raise ValueError(
            "No CREEDS signatures met the profile gene overlap threshold "
            f"({min_gene_overlap}) for organism '{organism}'."
        )
    rows.sort(key=lambda row: int(row.get("profile_gene_overlap", 0)), reverse=True)
    if max_profiles > 0:
        rows = rows[:max_profiles]
    return pd.DataFrame(rows)


def _fit_full_classical_models(X: np.ndarray, y: np.ndarray, *, random_state: int):
    models = {}
    for name, model in {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)),
            ]
        ),
    }.items():
        models[name] = model.fit(X, y)
    return models


def _fit_full_quantum_model(X: np.ndarray, y: np.ndarray, *, qml_dim: int, qsvc_reps: int):
    X_q, _ = _prepare_quantum_features(X, X[:1], qml_dim=qml_dim)
    model = _fit_local_qsvc(X_q, y, qml_dim=qml_dim, reps=qsvc_reps)
    return model


def run_ranking_benchmark(
    X_df: pd.DataFrame,
    y: np.ndarray,
    signature_path: str | Path,
    genes: List[str],
    *,
    candidate_profiles_path: Optional[str],
    cmap_signatures_path: Optional[str],
    creeds_signatures_path: Optional[str],
    gene_map_path: Optional[str],
    creeds_organism: str,
    demo_ranking: bool,
    min_profile_gene_overlap: int,
    max_ranking_profiles: int,
    qml_dim: int,
    qsvc_reps: int,
    quantum_mode: str,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if candidate_profiles_path:
        candidates = pd.read_csv(candidate_profiles_path)
        source = "candidate_profiles"
    elif cmap_signatures_path:
        candidates = load_cmap_candidate_profiles(
            cmap_signatures_path,
            genes,
            min_gene_overlap=min_profile_gene_overlap,
        )
        source = "cmap_signatures"
    elif creeds_signatures_path:
        candidates = load_creeds_candidate_profiles(
            creeds_signatures_path,
            genes,
            gene_map_path=gene_map_path,
            organism=creeds_organism,
            min_gene_overlap=min_profile_gene_overlap,
            max_profiles=max_ranking_profiles,
        )
        source = "creeds_signatures"
    elif demo_ranking:
        candidates = build_demo_candidate_profiles(signature_path, genes)
        source = "demo_signature_challenge"
    else:
        empty_metrics = pd.DataFrame(
            [{"ranking_verdict": "not_evaluable", "reason": "Provide --candidate-profiles or pass --demo-ranking."}]
        )
        return pd.DataFrame(), empty_metrics, {"ranking_evidence_level": "missing_candidate_profiles"}

    missing_genes = [g for g in genes if g not in candidates.columns]
    if missing_genes:
        raise ValueError(f"Candidate profiles missing selected genes: {missing_genes[:5]}")

    X = X_df[genes].to_numpy(dtype=float)
    case_centroid = X[y == 1].mean(axis=0)
    effects = candidates[genes].to_numpy(dtype=float)
    candidate_states = case_centroid[None, :] + effects
    lfc_vec = _signature_lfc_vector(signature_path, genes)

    classical_models = _fit_full_classical_models(X, y, random_state=random_state)
    classical_scores = []
    for model in classical_models.values():
        p_disease = model.predict_proba(candidate_states)[:, 1]
        classical_scores.append(1.0 - p_disease)
    classical_ensemble = np.mean(np.vstack(classical_scores), axis=0)

    if quantum_mode == "local":
        X_train_q, X_candidates_q = _prepare_quantum_features(X, candidate_states, qml_dim=qml_dim)
        q_model = _fit_local_qsvc(X_train_q, y, qml_dim=qml_dim, reps=qsvc_reps)
        quantum_score = 1.0 - _decision_scores(q_model, X_candidates_q)
        quantum_status = "ok"
    else:
        quantum_score = np.zeros(len(candidates), dtype=float)
        quantum_status = "skipped_ibm_gated"

    rows: List[Dict[str, Any]] = []
    fused_inputs = []
    for idx, row in candidates.iterrows():
        effect = effects[idx]
        reversal = _cosine01(effect, -lfc_vec)
        ef = EvidenceFeatures(
            compound=str(row.get("compound", row.get("candidate_id", idx))),
            compound_hetionet_id=row.get("compound_hetionet_id"),
            disease=str(row.get("disease", "rnaseq_signature")),
            disease_hetionet_id=row.get("disease_hetionet_id"),
            kg_rotate_score=float(row.get("kg_rotate_score", 0.5)),
            kg_complex_score=float(row.get("kg_complex_score", 0.5)),
            graph_topology_score=float(row.get("graph_topology_score", 0.5)),
            qsvc_score=float(quantum_score[idx]),
            classical_ensemble_score=float(classical_ensemble[idx]),
            signature_reversal_score=float(reversal),
            cell_type_reversal_score=float(reversal),
            pathway_reversal_score=0.0,
        )
        fused_inputs.append(ef)
        rows.append(
            {
                "candidate_id": row.get("candidate_id", idx),
                "compound": ef.compound,
                "expected_reverser": row.get("expected_reverser", None),
                "profile_gene_overlap": row.get("profile_gene_overlap", None),
                "creeds_id": row.get("creeds_id", None),
                "geo_id": row.get("geo_id", None),
                "cell_type": row.get("cell_type", None),
                "organism": row.get("organism", None),
                "kg_rotate_score": ef.kg_rotate_score,
                "kg_complex_score": ef.kg_complex_score,
                "graph_topology_score": ef.graph_topology_score,
                "signature_reversal_score": ef.signature_reversal_score,
                "classical_ensemble_score": ef.classical_ensemble_score,
                "qsvc_score": ef.qsvc_score,
                "quantum_status": quantum_status,
            }
        )

    kg_only = [EvidenceFeatures(**{**ef.to_dict(), "qsvc_score": 0.0, "classical_ensemble_score": 0.0}) for ef in fused_inputs]
    kg_omics = [EvidenceFeatures(**{**ef.to_dict(), "qsvc_score": 0.0}) for ef in fused_inputs]
    kg_omics_quantum = [EvidenceFeatures(**ef.to_dict()) for ef in fused_inputs]
    kg_only = fuse_evidence(kg_only, mode="kg-only")
    kg_omics = fuse_evidence(kg_omics, mode="kg+omics")
    kg_omics_quantum = fuse_evidence(kg_omics_quantum, mode="kg+omics")
    score_by_compound = {
        "kg_only": {x.compound: x.final_score for x in kg_only},
        "kg_omics": {x.compound: x.final_score for x in kg_omics},
        "kg_omics_quantum": {x.compound: x.final_score for x in kg_omics_quantum},
    }
    for row in rows:
        compound = row["compound"]
        row["kg_only_final_score"] = score_by_compound["kg_only"][compound]
        row["kg_omics_final_score"] = score_by_compound["kg_omics"][compound]
        row["kg_omics_quantum_final_score"] = score_by_compound["kg_omics_quantum"][compound]
        row["quantum_delta_score"] = row["kg_omics_quantum_final_score"] - row["kg_omics_final_score"]

    ranking_df = pd.DataFrame(rows).sort_values("kg_omics_quantum_final_score", ascending=False)
    metric_rows: List[Dict[str, Any]] = [
        {
            "ranking_evidence_level": source,
            "quantum_status": quantum_status,
            "metric": "candidate_count",
            "value": float(len(ranking_df)),
        }
    ]
    if "expected_reverser" in ranking_df.columns and ranking_df["expected_reverser"].notna().any():
        labels = pd.to_numeric(ranking_df["expected_reverser"], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(labels)
        for score_col in ["kg_only_final_score", "kg_omics_final_score", "kg_omics_quantum_final_score", "qsvc_score"]:
            try:
                auc = roc_auc_score(labels[valid], ranking_df.loc[valid, score_col].to_numpy(dtype=float))
            except Exception:
                auc = float("nan")
            metric_rows.append(
                {
                    "ranking_evidence_level": source,
                    "quantum_status": quantum_status,
                    "metric": f"{score_col}_roc_auc",
                    "value": auc,
                }
            )

    if len(ranking_df) > 1:
        from scipy.stats import spearmanr

        try:
            rho = float(
                spearmanr(
                    ranking_df["kg_omics_final_score"],
                    ranking_df["kg_omics_quantum_final_score"],
                ).statistic
            )
        except Exception:
            rho = float("nan")
        metric_rows.append(
            {
                "ranking_evidence_level": source,
                "quantum_status": quantum_status,
                "metric": "kg_omics_vs_quantum_spearman",
                "value": rho,
            }
        )
        rank_omics = ranking_df["kg_omics_final_score"].rank(ascending=False, method="min")
        rank_quantum = ranking_df["kg_omics_quantum_final_score"].rank(ascending=False, method="min")
        abs_shift = (rank_omics - rank_quantum).abs()
        metric_rows.extend(
            [
                {
                    "ranking_evidence_level": source,
                    "quantum_status": quantum_status,
                    "metric": "rank_shift_mean",
                    "value": float(abs_shift.mean()),
                },
                {
                    "ranking_evidence_level": source,
                    "quantum_status": quantum_status,
                    "metric": "rank_shift_max",
                    "value": float(abs_shift.max()),
                },
                {
                    "ranking_evidence_level": source,
                    "quantum_status": quantum_status,
                    "metric": "quantum_delta_score_std",
                    "value": float(ranking_df["quantum_delta_score"].std()),
                },
            ]
        )
        for k in [5, 10, 20]:
            if len(ranking_df) >= k:
                omics_top = set(
                    ranking_df.sort_values("kg_omics_final_score", ascending=False)
                    .head(k)["candidate_id"]
                )
                quantum_top = set(
                    ranking_df.sort_values("kg_omics_quantum_final_score", ascending=False)
                    .head(k)["candidate_id"]
                )
                metric_rows.append(
                    {
                        "ranking_evidence_level": source,
                        "quantum_status": quantum_status,
                        "metric": f"top_{k}_overlap_fraction",
                        "value": float(len(omics_top & quantum_top) / k),
                    }
                )

    return ranking_df, pd.DataFrame(metric_rows), {"ranking_evidence_level": source, "quantum_status": quantum_status}


def _ranking_metric_value(ranking_metrics: pd.DataFrame, metric: str) -> Optional[float]:
    if ranking_metrics.empty or "metric" not in ranking_metrics.columns:
        return None
    rows = ranking_metrics[ranking_metrics["metric"] == metric]
    if rows.empty:
        return None
    value = pd.to_numeric(rows.iloc[0].get("value"), errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def build_ranking_materiality(ranking_metrics: pd.DataFrame, *, ranking_is_real_evidence: bool) -> Dict[str, Any]:
    if ranking_metrics.empty:
        return {
            "ranking_quantum_materiality": "not_evaluable",
            "ranking_quantum_changes_top_k": False,
            "ranking_materiality_reason": "ranking metrics unavailable",
        }

    evidence_level = str(ranking_metrics["ranking_evidence_level"].dropna().iloc[0])
    quantum_status = str(ranking_metrics["quantum_status"].dropna().iloc[0])
    candidate_count = _ranking_metric_value(ranking_metrics, "candidate_count")
    spearman = _ranking_metric_value(ranking_metrics, "kg_omics_vs_quantum_spearman")
    shift_mean = _ranking_metric_value(ranking_metrics, "rank_shift_mean")
    shift_max = _ranking_metric_value(ranking_metrics, "rank_shift_max")
    delta_std = _ranking_metric_value(ranking_metrics, "quantum_delta_score_std")
    top_5 = _ranking_metric_value(ranking_metrics, "top_5_overlap_fraction")
    top_10 = _ranking_metric_value(ranking_metrics, "top_10_overlap_fraction")
    top_20 = _ranking_metric_value(ranking_metrics, "top_20_overlap_fraction")
    top_overlap = top_10 if top_10 is not None else top_5

    changes_top_k = bool(top_overlap is not None and top_overlap < 1.0)
    payload = {
        "ranking_quantum_materiality": "not_evaluable",
        "ranking_quantum_changes_top_k": changes_top_k,
        "ranking_materiality_reason": "",
        "ranking_candidate_count": candidate_count,
        "ranking_spearman_kg_omics_vs_quantum": spearman,
        "ranking_rank_shift_mean": shift_mean,
        "ranking_rank_shift_max": shift_max,
        "ranking_quantum_delta_score_std": delta_std,
        "ranking_top_5_overlap_fraction": top_5,
        "ranking_top_10_overlap_fraction": top_10,
        "ranking_top_20_overlap_fraction": top_20,
        "ranking_materiality_thresholds": {
            "negligible_spearman_min": 0.98,
            "negligible_top_k_overlap_min": 1.0,
            "negligible_rank_shift_mean_max": 1.0,
            "negligible_delta_score_std_max": 0.005,
            "material_spearman_max": 0.95,
            "material_rank_shift_mean_min": 2.0,
        },
    }

    if not ranking_is_real_evidence:
        payload["ranking_quantum_materiality"] = "not_real_evidence"
        payload["ranking_materiality_reason"] = f"{evidence_level} is not a real perturbation-ranking evidence source"
    elif quantum_status != "ok":
        payload["ranking_quantum_materiality"] = "not_evaluable"
        payload["ranking_materiality_reason"] = f"quantum ranking status is {quantum_status}"
    elif candidate_count is not None and candidate_count < 20:
        payload["ranking_quantum_materiality"] = "too_few_candidates"
        payload["ranking_materiality_reason"] = "fewer than 20 candidates met the ranking input thresholds"
    elif (
        spearman is not None
        and shift_mean is not None
        and delta_std is not None
        and top_overlap is not None
        and spearman >= 0.98
        and top_overlap >= 1.0
        and shift_mean < 1.0
        and delta_std < 0.005
    ):
        payload["ranking_quantum_materiality"] = "negligible"
        payload["ranking_materiality_reason"] = "KG+omics and KG+omics+quantum rankings are effectively unchanged"
    elif (
        changes_top_k
        or (spearman is not None and spearman < 0.95)
        or (shift_mean is not None and shift_mean >= 2.0)
    ):
        payload["ranking_quantum_materiality"] = "material"
        payload["ranking_materiality_reason"] = "quantum scores materially alter the KG+omics candidate order"
    else:
        payload["ranking_quantum_materiality"] = "minor"
        payload["ranking_materiality_reason"] = "quantum scores perturb rankings but do not clearly alter top candidates"
    return payload


def _write_markdown(metrics: pd.DataFrame, ranking_metrics: pd.DataFrame, verdict: Dict[str, Any], path: Path) -> None:
    evidence_level = ""
    if not ranking_metrics.empty and "ranking_evidence_level" in ranking_metrics.columns:
        evidence_level = str(ranking_metrics["ranking_evidence_level"].dropna().iloc[0])
    lines = [
        "# RNA-seq Quantum Benchmark",
        "",
        "## Verdict",
        "",
        f"- Classifier verdict: `{verdict.get('classifier_verdict')}`",
        f"- Quantum adds value: `{verdict.get('quantum_adds_value')}`",
        f"- Best quantum config: `{verdict.get('best_quantum_model')}`",
        f"- Best classical model: `{verdict.get('best_classical_model')}`",
        f"- Delta ROC-AUC: `{verdict.get('delta_roc_auc')}`",
        f"- Delta balanced accuracy: `{verdict.get('delta_balanced_accuracy')}`",
        f"- Classifier evidence grade: `{verdict.get('classifier_evidence_grade')}`",
        f"- Classifier sample-size warning: `{verdict.get('classifier_sample_size_warning')}`",
        f"- Best quantum ROC-AUC permutation p-value: "
        f"`{verdict.get('permutation_auc_best_quantum', {}).get('p_value_auc_ge_observed')}`",
        f"- Best classical ROC-AUC permutation p-value: "
        f"`{verdict.get('permutation_auc_best_classical', {}).get('p_value_auc_ge_observed')}`",
        f"- Full retraining permutation available: "
        f"`{verdict.get('full_retraining_permutation', {}).get('available')}`",
        f"- Full retraining quantum ROC-AUC p-value: "
        f"`{verdict.get('full_retraining_permutation', {}).get('p_value_quantum_auc_ge_observed')}`",
        f"- Full retraining classical ROC-AUC p-value: "
        f"`{verdict.get('full_retraining_permutation', {}).get('p_value_best_classical_auc_ge_observed')}`",
        f"- Bootstrap ROC-AUC delta 95% CI: "
        f"`{verdict.get('bootstrap_delta_roc_auc', {}).get('delta_ci95_low')}` to "
        f"`{verdict.get('bootstrap_delta_roc_auc', {}).get('delta_ci95_high')}`",
        f"- Evidence scope: `{verdict.get('evidence_scope')}`",
        f"- Ranking evidence level: `{evidence_level or 'not_evaluable'}`",
        f"- Ranking quantum materiality: `{verdict.get('ranking_quantum_materiality')}`",
        f"- Ranking materiality reason: `{verdict.get('ranking_materiality_reason')}`",
        "",
        "## Classifier Metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Ranking Metrics",
        "",
        ranking_metrics.to_markdown(index=False) if not ranking_metrics.empty else "Ranking benchmark was not evaluable.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-counts", default="artifacts/single_cell/airway_pilot/normalized_counts.csv")
    parser.add_argument("--metadata", default="artifacts/external/airway/converted/airway_metadata.csv")
    parser.add_argument("--signature", default="artifacts/signatures/airway_pilot/disease_signature.json")
    parser.add_argument("--out-dir", default="artifacts/benchmarks/rnaseq_quantum")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--case-label", default="trt")
    parser.add_argument("--control-label", default="untrt")
    parser.add_argument("--top-genes", type=int, default=24)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--qml-dim", type=int, default=4)
    parser.add_argument(
        "--qml-dims",
        default=None,
        help="Comma-separated QSVC feature dimensions to sweep. Overrides --qml-dim when set.",
    )
    parser.add_argument("--qsvc-reps", type=int, default=1)
    parser.add_argument(
        "--qsvc-reps-list",
        default=None,
        help="Comma-separated ZZFeatureMap repetition counts to sweep. Overrides --qsvc-reps when set.",
    )
    parser.add_argument("--quantum-mode", choices=["local", "ibm"], default="local")
    parser.add_argument("--allow-ibm-submit", action="store_true")
    parser.add_argument("--candidate-profiles", default=None)
    parser.add_argument(
        "--cmap-signatures",
        default=None,
        help="Tidy real compound perturbation CSV with compound/gene/score columns, or CMap defaults.",
    )
    parser.add_argument(
        "--creeds-signatures",
        default=None,
        help="Public CREEDS single_drug_perturbations JSON file.",
    )
    parser.add_argument(
        "--gene-map",
        default=None,
        help="Optional gene ID to symbol map for matching Ensembl RNA-seq features to profile genes.",
    )
    parser.add_argument("--creeds-organism", default="human")
    parser.add_argument("--min-profile-gene-overlap", type=int, default=3)
    parser.add_argument("--max-ranking-profiles", type=int, default=100)
    parser.add_argument("--demo-ranking", action="store_true")
    parser.add_argument("--min-delta", type=float, default=0.02)
    parser.add_argument("--bootstrap", type=int, default=1000, help="Paired bootstrap resamples for verdict uncertainty.")
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Fixed-score label permutations for ROC-AUC context; not a full model-retraining permutation test.",
    )
    parser.add_argument(
        "--full-permutations",
        type=int,
        default=0,
        help="Full label permutations that rerun feature selection and CV; slower but stronger than fixed-score permutations.",
    )
    parser.add_argument(
        "--full-permutation-qml-dims",
        default=None,
        help="Optional comma-separated QSVC dimensions for full retraining permutations. Defaults to --qml-dims.",
    )
    parser.add_argument(
        "--full-permutation-qsvc-reps-list",
        default=None,
        help="Optional comma-separated QSVC reps for full retraining permutations. Defaults to --qsvc-reps-list.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qml_dims = _parse_int_list(args.qml_dims if args.qml_dims is not None else args.qml_dim)
    qsvc_reps_list = _parse_int_list(args.qsvc_reps_list if args.qsvc_reps_list is not None else args.qsvc_reps)
    full_perm_qml_dims = _parse_int_list(args.full_permutation_qml_dims) if args.full_permutation_qml_dims else qml_dims
    full_perm_qsvc_reps_list = (
        _parse_int_list(args.full_permutation_qsvc_reps_list)
        if args.full_permutation_qsvc_reps_list
        else qsvc_reps_list
    )

    signature_genes = load_signature_genes(args.signature, args.top_genes)
    X_df, metadata, y, measured_genes = load_expression_inputs(
        args.normalized_counts,
        args.metadata,
        sample_id_col=args.sample_id_col,
        condition_col=args.condition_col,
        case_label=args.case_label,
        control_label=args.control_label,
        signature_genes=None,
    )
    genes = [gene for gene in signature_genes if gene in X_df.columns]
    if not genes:
        raise ValueError("None of the signature genes are present in normalized counts for ranking.")

    metrics_df, predictions_df, classifier_context = run_classifier_benchmark(
        X_df,
        y,
        cv_folds=args.cv_folds,
        top_n=args.top_genes,
        qml_dims=qml_dims,
        qsvc_reps_list=qsvc_reps_list,
        quantum_mode=args.quantum_mode,
        allow_ibm_submit=args.allow_ibm_submit,
        random_state=args.random_state,
    )
    verdict = build_value_verdict(
        metrics_df,
        min_delta=args.min_delta,
        predictions_df=predictions_df,
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        random_state=args.random_state,
    )
    full_permutation_summary, full_permutation_df = run_full_retraining_permutation(
        X_df,
        y,
        metrics_df,
        cv_folds=args.cv_folds,
        top_n=args.top_genes,
        qml_dims=full_perm_qml_dims,
        qsvc_reps_list=full_perm_qsvc_reps_list,
        quantum_mode=args.quantum_mode,
        allow_ibm_submit=args.allow_ibm_submit,
        n_permutations=args.full_permutations,
        random_state=args.random_state + 1000,
    )
    verdict["full_retraining_permutation"] = full_permutation_summary

    ranking_df, ranking_metrics_df, ranking_context = run_ranking_benchmark(
        X_df,
        y,
        args.signature,
        genes,
        candidate_profiles_path=args.candidate_profiles,
        cmap_signatures_path=args.cmap_signatures,
        creeds_signatures_path=args.creeds_signatures,
        gene_map_path=args.gene_map,
        creeds_organism=args.creeds_organism,
        demo_ranking=args.demo_ranking,
        min_profile_gene_overlap=args.min_profile_gene_overlap,
        max_ranking_profiles=args.max_ranking_profiles,
        qml_dim=qml_dims[0],
        qsvc_reps=qsvc_reps_list[0],
        quantum_mode=args.quantum_mode,
        random_state=args.random_state,
    )
    ranking_level = ranking_context.get("ranking_evidence_level")
    verdict["ranking_evidence_level"] = ranking_level
    verdict["ranking_is_real_evidence"] = ranking_level in {"candidate_profiles", "cmap_signatures", "creeds_signatures"}
    verdict.update(build_ranking_materiality(ranking_metrics_df, ranking_is_real_evidence=verdict["ranking_is_real_evidence"]))
    verdict["evidence_scope"] = (
        "real_classifier_and_real_ranking"
        if verdict["ranking_is_real_evidence"]
        else "real_classifier_only_ranking_not_evaluable"
    )
    if not verdict["ranking_is_real_evidence"]:
        verdict["ranking_limitation"] = (
            "Drug-ranking quantum value is not established without real compound perturbation profiles. "
            "Use --cmap-signatures or --candidate-profiles for ranking evidence."
        )

    metrics_path = out_dir / "classifier_metrics.csv"
    predictions_path = out_dir / "classifier_predictions.csv"
    full_permutation_path = out_dir / "classifier_full_permutation.csv"
    ranking_path = out_dir / "ranking_comparison.csv"
    ranking_metrics_path = out_dir / "ranking_metrics.csv"
    verdict_path = out_dir / "quantum_value_verdict.json"
    manifest_path = out_dir / "benchmark_manifest.json"
    md_path = out_dir / "benchmark_report.md"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    full_permutation_df.to_csv(full_permutation_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)
    ranking_metrics_df.to_csv(ranking_metrics_path, index=False)
    verdict_path.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "normalized_counts": str(args.normalized_counts),
        "metadata": str(args.metadata),
        "signature": str(args.signature),
        "selected_genes": genes,
        "classifier_feature_universe": {
            "n_measured_genes": int(len(measured_genes)),
            "source": "all_measured_genes",
            "selection_timing": "inside_each_training_fold",
            "cohort_wide_signature_used_for_classifier": False,
        },
        "quantum_sweep": {
            "qml_dims": qml_dims,
            "qsvc_reps_list": qsvc_reps_list,
            "ranking_uses_first_config": {"qml_dim": qml_dims[0], "qsvc_reps": qsvc_reps_list[0]},
            "full_permutation_sweep": {
                "qml_dims": full_perm_qml_dims,
                "qsvc_reps_list": full_perm_qsvc_reps_list,
            },
        },
        "classifier": classifier_context,
        "ranking": ranking_context,
        "statistical_guardrails": {
            "bootstrap_resamples": int(args.bootstrap),
            "score_label_permutations": int(args.permutations),
            "full_retraining_permutations": int(args.full_permutations),
            "full_retraining_permutation_available": bool(full_permutation_summary.get("available")),
            "classifier_evidence_grade": verdict.get("classifier_evidence_grade"),
            "ranking_quantum_materiality": verdict.get("ranking_quantum_materiality"),
        },
        "outputs": {
            "classifier_metrics": str(metrics_path),
            "classifier_predictions": str(predictions_path),
            "classifier_full_permutation": str(full_permutation_path),
            "ranking_comparison": str(ranking_path),
            "ranking_metrics": str(ranking_metrics_path),
            "quantum_value_verdict": str(verdict_path),
            "benchmark_report": str(md_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_markdown(metrics_df, ranking_metrics_df, verdict, md_path)

    print(
        json.dumps(
            {
                "verdict": verdict,
                "outputs": manifest["outputs"],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
