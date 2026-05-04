#!/usr/bin/env python3
"""Compute paired-bootstrap CIs (H1 and H1b) on real Hetionet CtD.

Reproduces the headline configuration (PauliFeatureMap reps=2, RotatE 128D
pair features, hard negative sampling, 5-fold stratified CV), persists per-fold
out-of-fold predictions per model to ``results/cv_predictions/``, runs the
paired-bootstrap-with-conjunction-across-baselines decision rule from
``utils/bootstrap_ci.py``, and emits ``docs/results/bootstrap_ci_analysis.md``.

Usage:
    python scripts/run_bootstrap_ci.py --dry_run        # verify imports only
    python scripts/run_bootstrap_ci.py --skip_qsvc      # classical baselines only (~minutes)
    python scripts/run_bootstrap_ci.py                  # full eval (multi-hour)
    python scripts/run_bootstrap_ci.py --resume_from_cache   # re-emit report from cached folds

Per ``preregistration/osf_preregistration_v1.md`` §8.1 the bootstrap is run
on out-of-fold (OOF) predictions concatenated across the 5 folds — every
test instance contributes once, the bootstrap resamples instances at the
OOF level. This is the standard CI-on-stacking-OOF approach. A per-fold
paired-bootstrap mode is left as future work.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.bootstrap_ci import conjunction_across_baselines  # noqa: E402
from utils.preregistered_constants import (  # noqa: E402
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    EMBEDDING_DIM,
    EMBEDDING_METHOD,
    NEGATIVE_SAMPLE_RATIO,
    QSVC_C_BEST,
    QSVC_FEATURE_MAP_REPS,
    QSVC_FEATURE_MAP_TYPE,
    QSVC_PRE_PCA_DIM,
    QSVC_QML_DIM,
    SPLIT_SEED,
)

CACHE_DIR_DEFAULT = os.path.join(REPO_ROOT, "results", "cv_predictions")
REPORT_PATH = os.path.join(REPO_ROOT, "docs", "results", "bootstrap_ci_analysis.md")
HETIONET_SNAPSHOT_PATH = "docs/reproducibility/hetionet_snapshot.md"

# GridSearchCV grids reproduced from scripts/run_optimized_pipeline.py.
RF_PARAM_GRID = {
    "n_estimators": [100, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 5],
}
ET_PARAM_GRID = {
    "n_estimators": [250, 500],
    "max_depth": [None, 20, 40],
    "min_samples_split": [2, 4, 8],
    "min_samples_leaf": [1, 2],
}


@dataclass
class FoldResult:
    """Per-model OOF artifacts for one fold."""

    fold_idx: int
    test_indices: np.ndarray  # global indices into the full instance set
    labels: np.ndarray  # (n_test,)
    scores: dict[str, np.ndarray]  # model_name -> (n_test,) probabilities


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--n_resamples", type=int, default=BOOTSTRAP_N_RESAMPLES)
    p.add_argument("--cache_dir", type=str, default=CACHE_DIR_DEFAULT)
    p.add_argument("--skip_qsvc", action="store_true", help="Skip QSVC training (debug, classical-only)")
    p.add_argument("--skip_ensemble", action="store_true", help="Skip stacking ensemble (debug)")
    p.add_argument("--resume_from_cache", action="store_true",
                   help="Skip CV training; re-emit report from cached fold_*.npz files")
    p.add_argument("--dry_run", action="store_true",
                   help="Verify imports + minimal setup; do NOT load data or train models")
    p.add_argument("--subsample", type=int, default=0,
                   help="Debug: cap positive count (0 = full dataset)")
    p.add_argument("--gpu", action="store_true",
                   help="Use GPU-backed Aer simulator (cuStateVec) for QSVC kernel evaluation. "
                        "Requires qiskit-aer-gpu installed against your CUDA version. "
                        "Verify with `python scripts/verify_qiskit_gpu.py` first.")
    p.add_argument("--quantum_config_path", type=str, default=None,
                   help="Override the quantum executor config (YAML). "
                        "If --gpu is set and this is unset, defaults to config/quantum_config_gpu.yaml. "
                        "Otherwise uses QMLLinkPredictor's default (CPU statevector).")
    return p.parse_args()


def _resolve_quantum_config(args: argparse.Namespace) -> str | None:
    """Apply the --gpu / --quantum_config_path precedence rules.

    Mirrors scripts/run_optimized_pipeline.py: --gpu forces the GPU YAML
    unless an explicit path is given. Returns the resolved path (or None
    if no path is configured).
    """
    if args.quantum_config_path:
        return args.quantum_config_path
    if args.gpu:
        return os.path.join("config", "quantum_config_gpu.yaml")
    return None


def _log_gpu_availability() -> None:
    """Probe the quantum executor for GPU support and log the result."""
    try:
        from quantum_layer.quantum_executor import QuantumExecutor  # noqa: E402
        if QuantumExecutor.gpu_available():
            print("[gpu] QuantumExecutor reports NVIDIA GPU available (cuStateVec)")
        else:
            print("[gpu] No GPU-backed Aer available; falling back to CPU statevector simulation")
    except Exception as e:
        print(f"[gpu] could not check GPU availability: {e}")


def _aer_gpu_truly_available() -> bool:
    """Strict check: AerSimulator.available_devices() actually includes 'GPU'.

    More reliable than QuantumExecutor.gpu_available() (which can return
    True on CPU-only systems). Used to gate `--gpu` so the multi-hour
    bootstrap CI doesn't silently fall through to CPU.
    """
    try:
        from qiskit_aer import AerSimulator
        return "GPU" in AerSimulator().available_devices()
    except Exception:
        return False


def _gate_gpu_or_abort(args: argparse.Namespace) -> None:
    """If --gpu is set, abort early unless qiskit-aer-gpu is genuinely working."""
    if not args.gpu:
        return
    if not _aer_gpu_truly_available():
        print(
            "[gpu] ABORT: --gpu was requested but AerSimulator.available_devices() "
            "does not include 'GPU'.\n"
            "       qiskit-aer-gpu is either not installed or not seeing CUDA.\n"
            "       Run `python scripts/verify_qiskit_gpu.py` for a full diagnosis.\n"
            "       To proceed on CPU instead, drop the --gpu flag.",
            file=sys.stderr,
        )
        sys.exit(2)
    print("[gpu] AerSimulator confirmed GPU device — proceeding with GPU CV.")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _hetionet_sha() -> str:
    """Return the SHA-256 of edges.sif from the snapshot doc, or 'unknown'."""
    snap = os.path.join(REPO_ROOT, HETIONET_SNAPSHOT_PATH)
    if not os.path.isfile(snap):
        return "unknown (run scripts/record_hetionet_hash.py)"
    with open(snap, encoding="utf-8") as f:
        for line in f:
            if "hetionet-v1.0-edges.sif" in line:
                # Markdown row: | `path` | size | mtime | `sha256` |
                parts = line.strip().split("|")
                if len(parts) >= 5:
                    return parts[4].strip().strip("`")
    return "unknown"


def _load_hetionet_and_embeddings(rel: str = "CtD"):
    """Load Hetionet + cached RotatE 128D embeddings; return positives + lookup."""
    from kg_layer.kg_loader import (  # noqa: E402
        extract_task_edges,
        get_hard_negatives,
        load_hetionet_edges,
    )

    print("[load] Hetionet edges...")
    df_edges = load_hetionet_edges()
    print(f"[load] Hetionet edges: {len(df_edges):,} rows")

    print(f"[load] Extracting {rel} task edges...")
    task_edges, entity_to_id, _id_to_entity = extract_task_edges(df_edges, relation_type=rel)
    print(f"[load] {rel} positives: {len(task_edges):,}")

    print(f"[load] {EMBEDDING_METHOD} {EMBEDDING_DIM}D embeddings...")
    emb_path = os.path.join(REPO_ROOT, "data", f"rotate_{EMBEDDING_DIM}d_entity_embeddings.npy")
    ids_path = os.path.join(REPO_ROOT, "data", f"rotate_{EMBEDDING_DIM}d_entity_ids.json")
    if not (os.path.isfile(emb_path) and os.path.isfile(ids_path)):
        raise FileNotFoundError(
            f"Cached RotatE embeddings missing: {emb_path} or {ids_path}. "
            "Run the embedding training step first."
        )
    embeddings = np.load(emb_path)
    with open(ids_path) as f:
        emb_entity_ids = json.load(f)
    print(f"[load] embeddings: {embeddings.shape}; ids: {len(emb_entity_ids):,} entries")

    return task_edges, entity_to_id, emb_entity_ids, embeddings, get_hard_negatives


def _build_pair_features(
    pos_neg_df,
    emb_entity_ids: dict,
    embeddings: np.ndarray,
):
    """Build (concat + diff + Hadamard + scalars) pair features per row.

    Drops rows whose source/target are missing from the embedding ID map.
    Returns (X, y, valid_mask).
    """
    from kg_layer.enhanced_features import EnhancedFeatureBuilder  # noqa: E402

    builder = EnhancedFeatureBuilder(
        include_graph_features=False,
        include_domain_features=False,
        normalize=True,
    )

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    valid_mask = np.zeros(len(pos_neg_df), dtype=bool)
    missing = 0
    for i, (_idx, row) in enumerate(pos_neg_df.iterrows()):
        s = str(row["source"])
        t = str(row["target"])
        if s not in emb_entity_ids or t not in emb_entity_ids:
            missing += 1
            continue
        h_emb = embeddings[emb_entity_ids[s]]
        t_emb = embeddings[emb_entity_ids[t]]
        X_rows.append(builder.build_embedding_features(h_emb, t_emb))
        y_rows.append(int(row["label"]))
        valid_mask[i] = True

    if missing:
        print(f"[features] dropped {missing:,} rows with missing embeddings")
    X = np.stack(X_rows, axis=0)
    y = np.array(y_rows, dtype=int)
    print(f"[features] X.shape = {X.shape}, y.shape = {y.shape}")
    return X, y


def _train_classical(name: str, grid: dict, X_train, y_train):
    """Tuned RF/ET/LR via GridSearchCV with 5-fold inner CV on PR-AUC."""
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    if name == "RF":
        base = RandomForestClassifier(class_weight="balanced", random_state=0, n_jobs=-1)
    elif name == "ET":
        base = ExtraTreesClassifier(class_weight="balanced", random_state=0, n_jobs=-1)
    elif name == "LR":
        base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    else:
        raise ValueError(name)

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    gs = GridSearchCV(
        base, grid, cv=inner_cv, scoring="average_precision", n_jobs=-1, refit=True
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def _train_qsvc(X_train, y_train, *, quantum_config_path: str | None = None):
    """Train QSVC with double-PCA pipeline matching the headline.

    Optionally routes through a non-default quantum executor config (e.g.,
    config/quantum_config_gpu.yaml for cuStateVec on DGX). The double-PCA
    pipeline (521D -> 24D -> 16D) is unchanged regardless of backend.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    from quantum_layer.qml_model import QMLLinkPredictor  # noqa: E402

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    pca_pre = PCA(n_components=QSVC_PRE_PCA_DIM, random_state=0).fit(X_train_std)
    X_train_pre = pca_pre.transform(X_train_std)
    pca_qml = PCA(n_components=QSVC_QML_DIM, random_state=0).fit(X_train_pre)
    X_train_qml = pca_qml.transform(X_train_pre)

    qsvc_kwargs = dict(
        model_type="QSVC",
        feature_map_type=QSVC_FEATURE_MAP_TYPE.replace("FeatureMap", ""),  # "Pauli" / "ZZ"
        feature_map_reps=QSVC_FEATURE_MAP_REPS,
        num_qubits=QSVC_QML_DIM,
        random_state=0,
        C=QSVC_C_BEST,
    )
    if quantum_config_path is not None:
        qsvc_kwargs["quantum_config_path"] = quantum_config_path

    qsvc = QMLLinkPredictor(**qsvc_kwargs)
    qsvc.fit(X_train_qml, y_train)
    return qsvc, scaler, pca_pre, pca_qml


def _qsvc_predict(qsvc, scaler, pca_pre, pca_qml, X) -> np.ndarray:
    X_std = scaler.transform(X)
    X_pre = pca_pre.transform(X_std)
    X_qml = pca_qml.transform(X_pre)
    proba = qsvc.predict_proba(X_qml)
    return proba[:, 1]


def _train_ensemble_oof(oof_scores: dict, oof_labels: np.ndarray, base_names: list[str]) -> np.ndarray:
    """OOF stacking: train a meta-LR on per-base OOF predictions, return OOF ensemble probas.

    Uses LeaveOneOut-style logic via 5-fold cross_val_predict so that the
    meta-learner's OOF predictions are themselves out-of-sample.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X_meta = np.stack([oof_scores[name] for name in base_names], axis=1)  # (n_total, n_bases)
    meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SPLIT_SEED)
    meta = LogisticRegression(max_iter=1000, random_state=0, class_weight="balanced")
    ensemble_oof = cross_val_predict(meta, X_meta, oof_labels, cv=meta_cv, method="predict_proba")[:, 1]
    return ensemble_oof


def run_cv(args: argparse.Namespace) -> tuple[dict, np.ndarray]:
    """Run 5-fold CV; return (oof_scores: name -> array, oof_labels)."""
    from sklearn.model_selection import StratifiedKFold

    task_edges, _entity_to_id, emb_entity_ids, embeddings, get_hard_negatives = (
        _load_hetionet_and_embeddings("CtD")
    )

    pos_df = task_edges.copy()
    pos_df["label"] = 1

    if args.subsample > 0:
        print(f"[subsample] capping positives at {args.subsample}")
        pos_df = pos_df.sample(n=min(args.subsample, len(pos_df)), random_state=SPLIT_SEED).reset_index(drop=True)

    print(f"[negatives] generating hard negatives 1:{NEGATIVE_SAMPLE_RATIO}...")
    neg_df = get_hard_negatives(
        pos_df, strategy="hard", num_negatives=NEGATIVE_SAMPLE_RATIO * len(pos_df), random_state=SPLIT_SEED
    )
    full_df = pd.concat([pos_df[["source", "target", "label"]], neg_df[["source", "target", "label"]]],
                        ignore_index=True)
    print(f"[dataset] {len(full_df):,} instances ({pos_df.shape[0]:,} pos, {neg_df.shape[0]:,} neg)")

    X, y = _build_pair_features(full_df, emb_entity_ids, embeddings)
    n = len(y)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SPLIT_SEED)

    oof_scores: dict[str, np.ndarray] = {name: np.full(n, np.nan) for name in ("LR", "RF", "ET")}
    if not args.skip_qsvc:
        oof_scores["QSVC"] = np.full(n, np.nan)

    os.makedirs(args.cache_dir, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"\n[fold {fold_idx + 1}/{args.n_folds}] train={len(train_idx):,}, test={len(test_idx):,}")

        fold_artifacts: dict[str, np.ndarray] = {"labels": y_test, "test_indices": test_idx}

        for name, grid in [("LR", {"C": [0.01, 0.1, 1.0, 10.0]}), ("RF", RF_PARAM_GRID), ("ET", ET_PARAM_GRID)]:
            print(f"  [{name}] training...")
            model = _train_classical(name, grid, X_train, y_train)
            scores = model.predict_proba(X_test)[:, 1]
            oof_scores[name][test_idx] = scores
            fold_artifacts[name] = scores

        if not args.skip_qsvc:
            print("  [QSVC] training...")
            qsvc, sc, p1, p2 = _train_qsvc(
                X_train, y_train, quantum_config_path=args._quantum_config_path
            )
            scores = _qsvc_predict(qsvc, sc, p1, p2, X_test)
            oof_scores["QSVC"][test_idx] = scores
            fold_artifacts["QSVC"] = scores

        np.savez(os.path.join(args.cache_dir, f"fold_{fold_idx}.npz"), **fold_artifacts)
        print(f"[fold {fold_idx + 1}] done in {time.time() - t0:.1f}s; cache -> fold_{fold_idx}.npz")

    return oof_scores, y


def load_from_cache(cache_dir: str) -> tuple[dict, np.ndarray]:
    """Reconstruct OOF scores + labels from cached fold_*.npz files."""
    fold_files = sorted(f for f in os.listdir(cache_dir) if f.startswith("fold_") and f.endswith(".npz"))
    if not fold_files:
        raise FileNotFoundError(f"No fold_*.npz in {cache_dir}")
    sample = np.load(os.path.join(cache_dir, fold_files[0]))
    n_total = max(int(np.load(os.path.join(cache_dir, f))["test_indices"].max()) for f in fold_files) + 1
    model_names = [k for k in sample.files if k not in ("labels", "test_indices")]
    oof_scores = {name: np.full(n_total, np.nan) for name in model_names}
    oof_labels = np.full(n_total, -1, dtype=int)
    for f in fold_files:
        data = np.load(os.path.join(cache_dir, f))
        idx = data["test_indices"]
        oof_labels[idx] = data["labels"]
        for name in model_names:
            oof_scores[name][idx] = data[name]
    return oof_scores, oof_labels


def emit_report(
    oof_scores: dict,
    oof_labels: np.ndarray,
    base_names: list[str],
    *,
    n_resamples: int,
    has_ensemble: bool,
    has_qsvc: bool,
    quantum_config_path: str | None = None,
) -> None:
    """Compute H1 / H1b CIs and emit the markdown report."""
    from sklearn.metrics import average_precision_score

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    today = dt.date.today().isoformat()
    commit = _git_commit()
    sha_edges = _hetionet_sha()

    pr_aucs = {name: float(average_precision_score(oof_labels, scores)) for name, scores in oof_scores.items()}

    lines = [
        "# Bootstrap CI Analysis — H1 and H1b decision rules",
        "",
        f"**Run date:** {today}",
        f"**Git commit:** `{commit}`",
        f"**Hetionet snapshot:** [`{HETIONET_SNAPSHOT_PATH}`](../{HETIONET_SNAPSHOT_PATH}) "
        f"(edges sha256 `{sha_edges}`)",
        f"**Configuration:** PauliFeatureMap reps=2, RotatE 128D pair features (concat + diff + Hadamard + scalars), "
        f"hard negatives 1:1, 5-fold stratified CV",
        f"**Quantum backend:** {quantum_config_path or 'default (CPU statevector)'}",
        f"**Bootstrap:** {n_resamples:,} resamples, seed `{BOOTSTRAP_SEED}`, "
        f"{int(BOOTSTRAP_CONFIDENCE * 100)}% confidence",
        "",
        "## OOF point estimates (PR-AUC)",
        "",
        "| Model | OOF PR-AUC |",
        "|---|---|",
    ]
    for name in oof_scores:
        lines.append(f"| {name} | {pr_aucs[name]:.4f} |")
    lines.append("")

    # H1 — QSVC alone vs each baseline
    if has_qsvc:
        baseline_names = [n for n in base_names if n != "QSVC"]
        baseline_dict = {n: oof_scores[n] for n in baseline_names}
        h1 = conjunction_across_baselines(
            oof_scores["QSVC"], oof_labels, baseline_dict, n_resamples=n_resamples
        )
        lines += [
            "## H1 — QSVC alone vs each baseline (paired bootstrap on OOF predictions)",
            "",
            "| Baseline | Point (QSVC − base) | 95% CI | Supports H1 (lo > 0)? |",
            "|---|---|---|---|",
        ]
        for name in baseline_names:
            r = h1["per_baseline"][name]
            lines.append(
                f"| {name} | {r['point']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                f"{'✓' if r['supported'] else '✗'} |"
            )
        lines += [
            "",
            f"**Conjunction:** {h1['n_baselines_supporting']} of {h1['n_baselines_total']} support H1.",
            f"**H1 supported:** **{'YES' if h1['h1_supported'] else 'NO'}**.",
            "",
        ]
    else:
        lines += [
            "## H1 — QSVC alone vs each baseline",
            "",
            "_QSVC was skipped in this run; H1 not computed. Re-run without `--skip_qsvc`._",
            "",
        ]

    # H1b — ensemble vs each baseline (if computed)
    if has_ensemble:
        baseline_names = [n for n in base_names if n not in ("QSVC", "Ensemble")]
        baseline_dict = {n: oof_scores[n] for n in baseline_names}
        h1b = conjunction_across_baselines(
            oof_scores["Ensemble"], oof_labels, baseline_dict, n_resamples=n_resamples
        )
        lines += [
            "## H1b — Stacking ensemble vs each baseline",
            "",
            "| Baseline | Point (Ens − base) | 95% CI | Supports H1b (lo > 0)? |",
            "|---|---|---|---|",
        ]
        for name in baseline_names:
            r = h1b["per_baseline"][name]
            lines.append(
                f"| {name} | {r['point']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | "
                f"{'✓' if r['supported'] else '✗'} |"
            )
        lines += [
            "",
            f"**Conjunction:** {h1b['n_baselines_supporting']} of {h1b['n_baselines_total']} support H1b.",
            f"**H1b supported:** **{'YES' if h1b['h1_supported'] else 'NO'}**.",
            "",
        ]
    else:
        lines += [
            "## H1b — Stacking ensemble vs each baseline",
            "",
            "_Ensemble was skipped in this run; H1b not computed._",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- Forward-looking baselines R-GCN and TransE are not yet implemented; "
        "H1/H1b are reported against the currently-implemented classical baselines (LR, RF, ET) "
        "and will be re-run after R-GCN and TransE are added (per preregistration §6.2).",
        "- Per-fold per-model predictions cached at `results/cv_predictions/fold_{0..4}.npz` for reproducibility.",
        f"- Bootstrap is on OOF predictions (each instance contributes once); "
        f"per-fold paired bootstrap is left as future work per preregistration §8.1's literal wording.",
        "",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nWrote {REPORT_PATH}")


def main() -> int:
    args = parse_args()
    args._quantum_config_path = _resolve_quantum_config(args)

    if args.dry_run:
        print("[dry_run] verifying imports only...")
        # Touch each module
        from kg_layer import kg_loader  # noqa: F401
        from kg_layer import enhanced_features  # noqa: F401
        if not args.skip_qsvc:
            from quantum_layer import qml_model  # noqa: F401
            _log_gpu_availability()
        from utils.bootstrap_ci import paired_bootstrap_pr_auc_difference  # noqa: F401
        print(f"[dry_run] OK. seed={BOOTSTRAP_SEED} resamples={args.n_resamples}")
        print(f"[dry_run] cache_dir={args.cache_dir}")
        print(f"[dry_run] report -> {REPORT_PATH}")
        if args._quantum_config_path:
            print(f"[dry_run] quantum_config_path={args._quantum_config_path}")
        return 0

    # Lazy import pandas (heavy)
    global pd
    import pandas as pd  # noqa: F811

    # Log GPU status if QSVC will be trained; abort hard if --gpu can't deliver
    if not args.skip_qsvc:
        _log_gpu_availability()
        _gate_gpu_or_abort(args)
        if args._quantum_config_path:
            print(f"[gpu] quantum_config_path={args._quantum_config_path}")

    if args.resume_from_cache:
        print(f"[resume] loading from {args.cache_dir}...")
        oof_scores, oof_labels = load_from_cache(args.cache_dir)
    else:
        oof_scores, oof_labels = run_cv(args)

    has_qsvc = "QSVC" in oof_scores and not np.isnan(oof_scores["QSVC"]).any()

    # Train ensemble (OOF stacking on top of base OOF predictions).
    has_ensemble = False
    if not args.skip_ensemble:
        base_names_for_ens = [n for n in ("LR", "RF", "ET", "QSVC") if n in oof_scores]
        if len(base_names_for_ens) >= 2:
            print("\n[ensemble] training OOF meta-learner...")
            oof_scores["Ensemble"] = _train_ensemble_oof(oof_scores, oof_labels, base_names_for_ens)
            has_ensemble = True
        else:
            print("[ensemble] skipped (not enough base learners)")

    base_names = list(oof_scores.keys())
    emit_report(
        oof_scores, oof_labels, base_names,
        n_resamples=args.n_resamples,
        has_ensemble=has_ensemble,
        has_qsvc=has_qsvc,
        quantum_config_path=args._quantum_config_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
