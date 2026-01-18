#!/usr/bin/env python3
"""
Quick test of optimizations using existing embeddings + enhanced features.
This demonstrates the power of feature engineering improvements alone.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = metrics["pr_auc"] = float("nan")
    return metrics


def test_model(name, model, X_train, y_train, X_test, y_test):
    logger.info(f"Testing {name}...")
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0

    # Get predictions
    if hasattr(model, "predict_proba"):
        test_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        test_scores = model.decision_function(X_test)
    else:
        test_scores = model.predict(X_test).astype(float)

    test_preds = (test_scores >= 0.5).astype(int) if hasattr(model, "predict_proba") else model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds, test_scores)

    logger.info(f"  ✅ Test PR-AUC: {test_metrics['pr_auc']:.4f} (fit: {fit_time:.1f}s)")
    return test_metrics, fit_time


print("="*80)
print("QUICK OPTIMIZATION TEST: Enhanced Features vs Baseline")
print("="*80)

# Load data
logger.info("Loading CtD data...")
df = load_hetionet_edges()
task_edges, entity_to_id, id_to_entity = extract_task_edges(df, relation_type="CtD")
train_df, test_df = prepare_link_prediction_dataset(task_edges)
logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

# Load embeddings (32D from existing cache)
embedder = HetionetEmbedder(embedding_dim=32, qml_dim=5)
if not embedder.load_saved_embeddings():
    logger.info("Training embeddings...")
    embedder.train_embeddings(task_edges)

# Get embeddings dict
embeddings = {}
for ent_id, idx in embedder.entity_to_id.items():
    embeddings[ent_id] = embedder.entity_embeddings[idx]
logger.info(f"Loaded {len(embeddings)} embeddings")

# ========== TEST 1: BASELINE FEATURES ==========
print("\n" + "="*80)
print("TEST 1: BASELINE FEATURES (Original)")
print("="*80)

def build_baseline_features(df, embeddings, id_to_entity):
    """Original feature building: [h, t, |h-t|, h*t]"""
    features = []
    src_col = 'source_id' if 'source_id' in df.columns else 'source'
    tgt_col = 'target_id' if 'target_id' in df.columns else 'target'

    for _, row in df.iterrows():
        h_id = str(id_to_entity.get(row[src_col], row[src_col]))
        t_id = str(id_to_entity.get(row[tgt_col], row[tgt_col]))
        h = embeddings.get(h_id, np.zeros(32))
        t = embeddings.get(t_id, np.zeros(32))
        feat = np.concatenate([h, t, np.abs(h - t), h * t])
        features.append(feat)
    return np.array(features)

X_train_baseline = build_baseline_features(train_df, embeddings, id_to_entity)
X_test_baseline = build_baseline_features(test_df, embeddings, id_to_entity)
y_train = train_df['label'].values
y_test = test_df['label'].values

logger.info(f"Baseline features: {X_train_baseline.shape}")

baseline_results = {}
models_baseline = {
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'SVM-RBF': SVC(kernel='rbf', C=3.0, gamma=0.1, class_weight='balanced', probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
}

for name, model in models_baseline.items():
    metrics, fit_time = test_model(name, model, X_train_baseline, y_train, X_test_baseline, y_test)
    baseline_results[name] = {'metrics': metrics, 'fit_time': fit_time}

# ========== TEST 2: ENHANCED FEATURES ==========
print("\n" + "="*80)
print("TEST 2: ENHANCED FEATURES (Optimized)")
print("="*80)

feature_builder = EnhancedFeatureBuilder(
    include_graph_features=True,
    include_domain_features=True,
    normalize=True
)

logger.info("Building graph...")
feature_builder.build_graph(task_edges)

# Convert IDs to entity names for feature builder
train_df_names = train_df.copy()
test_df_names = test_df.copy()
if 'source_id' in train_df.columns:
    train_df_names['source'] = train_df['source_id'].map(id_to_entity)
    test_df_names['source'] = test_df['source_id'].map(id_to_entity)
if 'target_id' in train_df.columns:
    train_df_names['target'] = train_df['target_id'].map(id_to_entity)
    test_df_names['target'] = test_df['target_id'].map(id_to_entity)

logger.info("Building enhanced features...")
X_train_enhanced, feature_names = feature_builder.build_features(train_df_names, embeddings, task_edges)
X_test_enhanced, _ = feature_builder.build_features(test_df_names, embeddings, task_edges)

logger.info(f"Enhanced features: {X_train_enhanced.shape} ({len(feature_names)} features)")

enhanced_results = {}
models_enhanced = {
    'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    'SVM-RBF': SVC(kernel='rbf', C=3.0, gamma=0.1, class_weight='balanced', probability=True, random_state=42),
    'LogisticRegression': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42),
}

for name, model in models_enhanced.items():
    metrics, fit_time = test_model(name, model, X_train_enhanced, y_train, X_test_enhanced, y_test)
    enhanced_results[name] = {'metrics': metrics, 'fit_time': fit_time}

# ========== COMPARISON REPORT ==========
print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

print(f"\n{'Model':<25} | {'Baseline PR-AUC':<15} | {'Enhanced PR-AUC':<15} | {'Improvement':<12} | {'% Gain':<10}")
print("-" * 90)

for name in baseline_results.keys():
    baseline_pr = baseline_results[name]['metrics']['pr_auc']
    enhanced_pr = enhanced_results[name]['metrics']['pr_auc']
    improvement = enhanced_pr - baseline_pr
    pct_gain = (improvement / baseline_pr * 100) if baseline_pr > 0 else 0

    print(f"{name:<25} | {baseline_pr:<15.4f} | {enhanced_pr:<15.4f} | {improvement:<12.4f} | {pct_gain:<10.1f}%")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

best_baseline = max(baseline_results.items(), key=lambda x: x[1]['metrics']['pr_auc'])
best_enhanced = max(enhanced_results.items(), key=lambda x: x[1]['metrics']['pr_auc'])

print(f"🏆 Best Baseline: {best_baseline[0]} - PR-AUC: {best_baseline[1]['metrics']['pr_auc']:.4f}")
print(f"🏆 Best Enhanced: {best_enhanced[0]} - PR-AUC: {best_enhanced[1]['metrics']['pr_auc']:.4f}")

overall_improvement = best_enhanced[1]['metrics']['pr_auc'] - best_baseline[1]['metrics']['pr_auc']
overall_pct = (overall_improvement / best_baseline[1]['metrics']['pr_auc'] * 100)

print(f"\n📊 Overall Best Improvement: {overall_improvement:+.4f} ({overall_pct:+.1f}%)")

print(f"\nFeature Dimensions:")
print(f"  Baseline: {X_train_baseline.shape[1]} features")
print(f"  Enhanced: {X_train_enhanced.shape[1]} features ({X_train_enhanced.shape[1] - X_train_baseline.shape[1]} new features)")

print("\n✅ Test complete!")
