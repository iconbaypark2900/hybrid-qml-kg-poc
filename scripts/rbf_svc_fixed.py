import os
import json
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, Callable

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report
)

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder


# --------------------
# Column / metric helpers
# --------------------
def _find_cols(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Robustly infer (src, dst, label) column names.
    Handles variants like source_id/target_id and fuzzy matches.
    """
    cols = list(df.columns)
    low2orig = {c.lower(): c for c in cols}
    lowers = [c.lower() for c in cols]

    # label first (exact then fuzzy)
    label_exact = ["label", "y", "is_positive", "class", "target_y"]
    y = next((low2orig[c] for c in label_exact if c in low2orig), None)
    if y is None:
        # any small-cardinality 0/1 numeric column
        for c in cols:
            s = df[c]
            if np.issubdtype(s.dtype, np.number):
                vals = pd.unique(s.dropna())
                if len(vals) <= 3 and set(vals).issubset({0, 1, 0.0, 1.0}):
                    y = c
                    break

    # src/dst exact candidates (added *_id forms)
    src_exact = ["source", "source_id", "src", "src_id", "u", "head", "subject", "left", "entity1"]
    dst_exact = ["target", "target_id", "dst", "dst_id", "v", "tail", "object", "right", "entity2"]

    src = next((low2orig[c] for c in src_exact if c in low2orig), None)
    dst = next((low2orig[c] for c in dst_exact if c in low2orig), None)

    # Fuzzy fallback (substring contains)
    def fuzzy_find(needles):
        for c in lowers:
            if any(n in c for n in needles):
                return low2orig[c]
        return None

    if src is None:
        src = fuzzy_find(["source", "src"])
    if dst is None:
        dst = fuzzy_find(["target", "dst"])

    # Guard against picking the same column twice
    if src == dst:
        # try to pick a second best fuzzy for dst
        candidates = [low2orig[c] for c in lowers if ("target" in c or "dst" in c) and low2orig[c] != src]
        if candidates:
            dst = candidates[0]
        else:
            dst = None

    if not (src and dst and y):
        raise ValueError(
            f"Could not infer (src, dst, label) from columns: {cols}. "
            "Expected names like source/source_id, target/target_id, and label (or provide a 0/1 numeric label)."
        )
    return src, dst, y


def _scores_to_continuous(estimator, X):
    if hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X)
        if isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] == 2:
            s = s[:, 1]
        return s
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.predict(X).astype(float)


def _compute_metrics(y_true, y_pred, y_score) -> Dict[str, float]:
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        m["roc_auc"] = float("nan")
    try:
        m["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        m["pr_auc"] = float("nan")
    return m


# --------------------
# Embedding access (robust)
# --------------------
def _make_embedding_getter(embedder):
    """
    Return (getter, dim_hint) where getter(eid) -> 1D np.ndarray for the entity.
    Tries, in order:
      1) direct accessor methods on the embedder
      2) matrix + mapping dict
      3) matrix + aligned id list
      4) **row-index fallback**: if eid can be cast to int and is in [0, nrows)
    """
    # 0) try to load saved embeddings if the embedder supports it
    M = None
    load_fn = getattr(embedder, "load_saved_embeddings", None)
    if callable(load_fn):
        try:
            load_fn()
        except Exception:
            pass
    # pick up a likely embeddings matrix from common attribute names or any 2-D array
    candidates = []
    for name in ("embeddings_", "embeddings", "embedding_matrix"):
        v = getattr(embedder, name, None)
        if isinstance(v, np.ndarray) and v.ndim == 2:
            candidates.append((name, v))
    if not candidates and hasattr(embedder, "__dict__"):
        for k, v in vars(embedder).items():
            if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.number):
                if 8 <= v.shape[1] <= 4096:
                    candidates.append((k, v))
    candidates.sort(key=lambda kv: kv[1].shape[1], reverse=True)
    if candidates:
        _, M = candidates[0]

    # 1) direct accessor methods
    for name in ("get_embedding", "get_entity_embedding", "embedding_for",
                 "transform_entity", "transform_node", "vector_for"):
        fn = getattr(embedder, name, None)
        if callable(fn):
            def _getter(eid, _fn=fn):
                return np.asarray(_fn(eid)).ravel()
            return _getter, None

    # 2) find a mapping dict (id -> row idx)
    mapping = None
    for name in ("entity_to_idx", "entity2idx", "id_to_idx", "name_to_idx",
                 "node_to_idx", "index_map", "id_map", "entity_index"):
        cand = getattr(embedder, name, None)
        if isinstance(cand, dict) and cand:
            mapping = cand
            break
    if mapping is None and hasattr(embedder, "__dict__"):
        # heuristic: any dict with int-like values
        for k, v in vars(embedder).items():
            if isinstance(v, dict) and v and all(isinstance(iv, (int, np.integer)) for iv in v.values()):
                mapping = v
                break

    # 3) aligned id list
    id_list = None
    if M is not None:
        nrows = M.shape[0]
        for name in ("entity_ids", "ids", "entities", "node_ids", "index_to_id", "idx_to_id"):
            cand = getattr(embedder, name, None)
            if isinstance(cand, (list, np.ndarray)) and len(cand) == nrows:
                id_list = cand
                break
        if id_list is None and hasattr(embedder, "__dict__"):
            for k, v in vars(embedder).items():
                if isinstance(v, (list, np.ndarray)) and len(v) == nrows:
                    id_list = v
                    break

    # helper: row-index fallback into M
    def _row_index_fallback(eid, _M=M):
        if _M is None:
            raise KeyError  # no matrix to fall back to
        try:
            i = int(eid)
        except Exception:
            raise KeyError
        if 0 <= i < _M.shape[0]:
            return _M[i].ravel()
        raise KeyError

    # if we have mapping, build getter that tries mapping then row-index fallback
    if mapping is not None and M is not None:
        def _lookup_idx(eid, _map=mapping):
            if eid in _map:
                return int(_map[eid])
            sid = str(eid)
            if sid in _map:
                return int(_map[sid])
            # give up; caller will try row-index fallback
            raise KeyError

        def _getter(eid, _M=M):
            try:
                idx = _lookup_idx(eid)
                return _M[idx].ravel()
            except KeyError:
                # <- crucial fix for your error: try treating eid as a row index
                return _row_index_fallback(eid, _M)
        return _getter, int(M.shape[1])

    # matrix + aligned id list
    if id_list is not None and M is not None:
        idx_map = {}
        for i, e in enumerate(id_list):
            idx_map[e] = i
            idx_map[str(e)] = i

        def _getter(eid, _M=M, _imap=idx_map):
            i = _imap.get(eid, _imap.get(str(eid), None))
            if i is None:
                # fallback to row index
                return _row_index_fallback(eid, _M)
            return _M[i].ravel()
        return _getter, int(M.shape[1])

    # if we *only* have the matrix, rely entirely on row-index fallback
    if M is not None:
        def _getter(eid, _M=M):
            return _row_index_fallback(eid, _M)
        return _getter, int(M.shape[1])

    raise AttributeError("Could not find a way to retrieve embeddings from HetionetEmbedder.")


def _probe_dim_single(getter: Callable[[Any], Optional[np.ndarray]]) -> int:
    # we can only get dim once we see a vector; defer to runtime by returning -1 if unknown
    return -1


def _infer_dim_from_data(getter: Callable[[Any], Optional[np.ndarray]],
                         df: pd.DataFrame, src_col: str, dst_col: str) -> int:
    for e in pd.concat([df[src_col], df[dst_col]], ignore_index=True):
        v = getter(e)
        if v is not None:
            return int(v.shape[0])
    raise RuntimeError("Could not infer embedding dimension from any entity.")


# --------------------
# Feature construction
# --------------------
def _pair_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """128-D classical features for 32-D (or 4*dim in general): [u, v, |u−v|, u*v]."""
    return np.concatenate([u, v, np.abs(u - v), u * v], axis=-1)


def _build_features_with_getter(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    y_col: str,
    getter: Callable[[Any], Optional[np.ndarray]],
    known_dim: int = -1,
) -> Tuple[np.ndarray, np.ndarray]:
    labels = df[y_col].astype(int).to_numpy()

    dim = known_dim if known_dim > 0 else _infer_dim_from_data(getter, df, src_col, dst_col)
    feats = np.zeros((len(df), dim * 4), dtype=np.float32)

    keep_mask = np.ones(len(df), dtype=bool)
    for i, (s, t) in enumerate(zip(df[src_col].tolist(), df[dst_col].tolist())):
        us = getter(s)
        vt = getter(t)
        if us is None or vt is None:
            keep_mask[i] = False
            continue
        if us.shape[0] != dim or vt.shape[0] != dim:
            keep_mask[i] = False
            continue
        feats[i] = _pair_features(us, vt)

    if not keep_mask.all():
        feats = feats[keep_mask]
        labels = labels[keep_mask]
    return feats, labels


# --------------------
# Main
# --------------------
def main():
    # Config
    relation = "CtD"
    max_entities = 300
    embedding_dim = 32
    qml_dim = 5  # only to satisfy embedder ctor signature
    random_state = 42
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Data
    print("\nLoading data...")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(df, relation_type=relation, max_entities=max_entities)
    train_df, test_df = prepare_link_prediction_dataset(task_edges)

    # Column names
    src_col, dst_col, y_col = _find_cols(train_df)

    # Embeddings
    embedder = HetionetEmbedder(embedding_dim=embedding_dim, qml_dim=qml_dim)
    if not embedder.load_saved_embeddings():
        if hasattr(embedder, "train_or_load_embeddings"):
            embedder.train_or_load_embeddings()
        elif hasattr(embedder, "train_embeddings"):
            embedder.train_embeddings()
        elif hasattr(embedder, "fit"):
            embedder.fit()
        else:
            raise RuntimeError("No embeddings found and no training method available on HetionetEmbedder.")

    getter, dim_hint = _make_embedding_getter(embedder)

    # Build classical 4*dim features
    X_train, y_train = _build_features_with_getter(train_df, src_col, dst_col, y_col, getter, dim_hint)
    X_test,  y_test  = _build_features_with_getter(test_df,  src_col, dst_col, y_col, getter, dim_hint)
    print(f"Shapes: train {X_train.shape}, test {X_test.shape}")

    # Model & search
    pipe = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", probability=False, random_state=random_state),
    )
    param_grid = {
        "svc__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "svc__gamma": [0.01, 0.03, 0.1, 0.3],
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="average_precision",
        cv=skf,
        refit=True,
        n_jobs=-1,
        verbose=0,
        return_train_score=False,
    )

    t0 = time.time()
    gs.fit(X_train, y_train)
    fit_s = time.time() - t0
    print(f"Best params: {gs.best_params_} (cv PR-AUC={gs.best_score_:.4f}, fit in {fit_s:.1f}s)")

    # Train metrics – OUT-OF-FOLD (no optimistic train=1.0000)
    best_C = gs.best_params_["svc__C"]
    best_gamma = gs.best_params_["svc__gamma"]
    best_pipe = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=best_C, gamma=best_gamma, probability=False, random_state=random_state),
    )
    oof_scores = cross_val_predict(best_pipe, X_train, y_train, cv=skf, method="decision_function", n_jobs=-1)
    oof_preds = (oof_scores >= 0).astype(int)
    train_m = _compute_metrics(y_train, oof_preds, oof_scores)

    print("\nTrain metrics (out-of-fold):")
    print(f"  accuracy : {train_m['accuracy']:.4f}")
    print(f"  precision: {train_m['precision']:.4f}")
    print(f"  recall   : {train_m['recall']:.4f}")
    print(f"  f1       : {train_m['f1']:.4f}")
    print(f"  roc_auc  : {train_m['roc_auc']:.4f}")
    print(f"  pr_auc   : {train_m['pr_auc']:.4f}")
    print("\n", classification_report(y_train, oof_preds, digits=2))

    # Test metrics – held out
    best_est = gs.best_estimator_
    test_scores = _scores_to_continuous(best_est, X_test)
    test_preds = (test_scores >= 0).astype(int)
    test_m = _compute_metrics(y_test, test_preds, test_scores)

    print("\nTest metrics:")
    print(f"  accuracy : {test_m['accuracy']:.4f}")
    print(f"  precision: {test_m['precision']:.4f}")
    print(f"  recall   : {test_m['recall']:.4f}")
    print(f"  f1       : {test_m['f1']:.4f}")
    print(f"  roc_auc  : {test_m['roc_auc']:.4f}")
    print(f"  pr_auc   : {test_m['pr_auc']:.4f}")
    print("\n", classification_report(y_test, test_preds, digits=2))

    # Save
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(results_dir, f"rbf_svc_128d_fixed_{stamp}.json")
    payload = {
        "relation": relation,
        "max_entities": max_entities,
        "embedding_dim": embedding_dim,
        "feature_dim": int(X_train.shape[1]),
        "cv": {"n_splits": skf.get_n_splits(), "best_params": gs.best_params_, "best_score_pr_auc": gs.best_score_},
        "train_metrics_oof": train_m,
        "test_metrics": test_m,
        "fit_seconds": fit_s,
        "random_state": random_state,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote → {out_path}")


if __name__ == "__main__":
    main()
