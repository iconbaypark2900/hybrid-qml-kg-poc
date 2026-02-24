from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PyKEENRunResult:
    model_name: str
    pr_auc: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_seconds: float
    n_train_pos: int
    n_test_pos: int
    n_train_total: int
    n_test_total: int


def _as_triples(df: pd.DataFrame) -> np.ndarray:
    """Return triples as shape (n, 3) ndarray of strings: (head, relation, tail)."""
    return df[["source", "metaedge", "target"]].astype(str).to_numpy()


def _score_binary(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    # Threshold scores at median (robust) if they are not probabilities
    thr = 0.5 if (np.nanmin(y_score) >= 0.0 and np.nanmax(y_score) <= 1.0) else float(np.nanmedian(y_score))
    y_pred = (y_score >= thr).astype(int)
    out = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return out


def run_pykeen_direct_scoring(
    *,
    df_edges: pd.DataFrame,
    relation: str,
    train_pos_edges: pd.DataFrame,
    test_pos_edges: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: str = "RotatE",
    embedding_dim: int = 128,
    epochs: int = 100,
    random_state: int = 42,
    device: Optional[str] = None,
) -> Tuple[PyKEENRunResult, Dict[str, str]]:
    """
    Train a PyKEEN model (RotatE/ComplEx) directly as a link predictor and score
    the held-out test set (positives + sampled negatives) with PR-AUC.

    Notes:
    - We remove test positives for the target `relation` from the training triples to avoid leakage.
    - We still allow training on other relations (full-graph context).
    """
    import time
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    # Build training triple set: all edges, but exclude held-out relation positives in test.
    df_all = df_edges[["source", "metaedge", "target"]].copy()
    df_all = df_all.astype(str)

    # Remove held-out test positives for the target relation (by exact triple match)
    test_pos = test_pos_edges[["source", "metaedge", "target"]].astype(str)
    if len(test_pos) > 0:
        key = test_pos.assign(_k=1).merge(df_all.assign(_k=1), on=["source", "metaedge", "target"], how="right")
        df_train_triples = key[key["_k_x"].isna()][["source", "metaedge", "target"]].copy()
    else:
        df_train_triples = df_all

    train_triples = _as_triples(df_train_triples)
    tf = TriplesFactory.from_labeled_triples(train_triples)

    t0 = time.time()
    res = pipeline(
        training=tf,
        model=str(model),
        model_kwargs={"embedding_dim": int(embedding_dim)},
        training_kwargs={"num_epochs": int(epochs)},
        random_seed=int(random_state),
        device=device,
    )
    train_seconds = float(time.time() - t0)

    # Score binary evaluation set (test positives + test negatives)
    # We use the exact same entity/relation labels as in df_edges.
    # For consistency with the rest of the pipeline, evaluate PR-AUC over the sampled negatives.
    y_true = test_df["label"].astype(int).to_numpy()

    # Build hrt triples for scoring (use metaedge for relation)
    # test_df may only have source_id/target_id; prefer existing 'source'/'target' if present.
    if "source" in test_df.columns and "target" in test_df.columns:
        df_score = pd.DataFrame(
            {
                "source": test_df["source"].astype(str),
                "metaedge": str(relation),
                "target": test_df["target"].astype(str),
            }
        )
    else:
        # Fall back: require train_pos_edges/test_pos_edges to have string columns and map IDs there.
        raise ValueError("test_df missing 'source'/'target' columns required for PyKEEN scoring.")

    hrt = _as_triples(df_score)
    scores = res.model.score_hrt(hrt)
    scores = np.asarray(scores, dtype=float)
    m = _score_binary(y_true, scores)

    run = PyKEENRunResult(
        model_name=f"PyKEEN-{model}",
        pr_auc=float(m["pr_auc"]),
        roc_auc=float(m["roc_auc"]),
        accuracy=float(m["accuracy"]),
        precision=float(m["precision"]),
        recall=float(m["recall"]),
        f1=float(m["f1"]),
        train_seconds=train_seconds,
        n_train_pos=int(train_pos_edges.shape[0]),
        n_test_pos=int(test_pos_edges.shape[0]),
        n_train_total=int(train_df.shape[0]),
        n_test_total=int(test_df.shape[0]),
    )

    meta = {
        "pykeen_model": str(model),
        "pykeen_embedding_dim": str(int(embedding_dim)),
        "pykeen_epochs": str(int(epochs)),
        "pykeen_device": str(device) if device else "",
    }
    return run, meta

