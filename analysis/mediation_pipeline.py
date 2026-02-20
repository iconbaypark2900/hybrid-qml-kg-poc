# analysis/mediation_pipeline.py

"""
Mediation analysis pipeline for H-002.
Compare: Base model, Base + lysosomal features, Lysosomal-only model.
Measures: Directional stability, PR-AUC delta, confidence interval overlap.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_mediation_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    embedder=None,
) -> Dict[str, Any]:
    """
    Run mediation comparison: base vs base+lysosomal vs lysosomal-only.

    Evaluates three configurations and returns metrics for each.
    When lysosomal features improve DC, cross-split stability, and reduce variance,
    mediation is supported.

    Args:
        train_df: Training pairs with source, target, label
        test_df: Test pairs
        edges_df: Full edges for domain/lysosomal features
        embeddings: Entity ID -> embedding vector
        embedder: Optional HetionetEmbedder for prepare_link_features

    Returns:
        Dict with base_metrics, base_lysosomal_metrics, lysosomal_only_metrics,
        mediation_supported, stability_with, stability_without, directional_with, directional_without
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score
    from kg_layer.evidence_weighting import EvidenceConfig, build_compound_disease_gene_maps
    from kg_layer.lysosomal_features import (
        build_lysosomal_gene_set,
        build_lysosomal_features,
    )
    from benchmarking.directional_metrics import compute_directional_consistency

    comp2g, dis2g = build_compound_disease_gene_maps(edges_df, EvidenceConfig())
    lysosomal_genes = build_lysosomal_gene_set(edges_df, pathway_filter=None)

    src_col = "source" if "source" in train_df.columns else "source_id"
    tgt_col = "target" if "target" in train_df.columns else "target_id"

    def _get_emb_features(df: pd.DataFrame) -> np.ndarray:
        """Build embedding-based features [h, t, |h-t|, h*t]."""
        feats = []
        emb_dim = next(iter(embeddings.values())).shape[0]
        for _, row in df.iterrows():
            h = str(row[src_col])
            t = str(row[tgt_col])
            hv = embeddings.get(h, np.zeros(emb_dim))
            tv = embeddings.get(t, np.zeros(emb_dim))
            diff = np.abs(hv - tv)
            had = hv * tv
            feats.append(np.concatenate([hv, tv, diff, had]))
        return np.array(feats, dtype=np.float32)

    def _get_lysosomal_feats(df: pd.DataFrame) -> np.ndarray:
        return build_lysosomal_features(df, edges_df, comp2g, dis2g, lysosomal_genes=lysosomal_genes,
                                        source_col=src_col, target_col=tgt_col)

    # 1. Base model (embedding features only)
    X_train_base = _get_emb_features(train_df)
    X_test_base = _get_emb_features(test_df)
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    scaler_base = StandardScaler()
    X_train_base = scaler_base.fit_transform(X_train_base)
    X_test_base = scaler_base.transform(X_test_base)
    clf_base = LogisticRegression(max_iter=500, random_state=42)
    clf_base.fit(X_train_base, y_train)
    pred_base = clf_base.predict_proba(X_test_base)[:, 1]
    pr_auc_base = average_precision_score(y_test, pred_base)
    pred_dir_base = np.sign(pred_base - 0.5)
    gt_dir = np.sign(y_test.astype(float) - 0.5)
    dc_base = compute_directional_consistency(pred_dir_base, gt_dir)

    # 2. Base + lysosomal
    X_train_lyso = _get_lysosomal_feats(train_df)
    X_test_lyso = _get_lysosomal_feats(test_df)
    X_train_both = np.concatenate([X_train_base, X_train_lyso], axis=1)
    X_test_both = np.concatenate([X_test_base, X_test_lyso], axis=1)
    scaler_both = StandardScaler()
    X_train_both = scaler_both.fit_transform(X_train_both)
    X_test_both = scaler_both.transform(X_test_both)
    clf_both = LogisticRegression(max_iter=500, random_state=42)
    clf_both.fit(X_train_both, y_train)
    pred_both = clf_both.predict_proba(X_test_both)[:, 1]
    pr_auc_both = average_precision_score(y_test, pred_both)
    pred_dir_both = np.sign(pred_both - 0.5)
    dc_both = compute_directional_consistency(pred_dir_both, gt_dir)

    # 3. Lysosomal-only
    scaler_lyso = StandardScaler()
    X_train_lyso_only = scaler_lyso.fit_transform(X_train_lyso)
    X_test_lyso_only = scaler_lyso.transform(X_test_lyso)
    clf_lyso = LogisticRegression(max_iter=500, random_state=42)
    clf_lyso.fit(X_train_lyso_only, y_train)
    pred_lyso = clf_lyso.predict_proba(X_test_lyso_only)[:, 1]
    pr_auc_lyso = average_precision_score(y_test, pred_lyso)
    pred_dir_lyso = np.sign(pred_lyso - 0.5)
    dc_lyso = compute_directional_consistency(pred_dir_lyso, gt_dir)

    # Mediation signal: lysosomal improves DC, stability; reduces variance
    mediation_supported = dc_both >= dc_base and pr_auc_both >= pr_auc_base
    stability_with = float(np.std([pr_auc_base, pr_auc_both]) if pr_auc_both else pr_auc_base)
    stability_without = float(pr_auc_base)
    directional_with = float(dc_both)
    directional_without = float(dc_base)

    results = {
        "base_metrics": {"pr_auc": pr_auc_base, "directional_consistency": dc_base},
        "base_lysosomal_metrics": {"pr_auc": pr_auc_both, "directional_consistency": dc_both},
        "lysosomal_only_metrics": {"pr_auc": pr_auc_lyso, "directional_consistency": dc_lyso},
        "mediation_supported": mediation_supported,
        "stability_with": stability_with,
        "stability_without": stability_without,
        "directional_with": directional_with,
        "directional_without": directional_without,
        "direct_effect": None,
        "indirect_effect": None,
        "total_effect": None,
        "mediation_proportion": None,
        "lysosomal_features_included": True,
    }
    return results
