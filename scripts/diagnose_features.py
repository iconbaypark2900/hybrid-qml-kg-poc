"""
Diagnostic script to investigate why enhanced features might be failing.

Usage:
    python scripts/diagnose_features.py --relation CtD
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder, validate_no_leakage

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def diagnose_features(X_train, y_train, X_test, y_test, feature_names=None):
    """Diagnose why features might be failing."""
    
    logger.info(f"\n{'='*80}")
    logger.info("FEATURE DIAGNOSTICS")
    logger.info(f"{'='*80}")
    logger.info(f"Feature matrix shape: {X_train.shape}")
    
    # 1. Check for NaN/Inf
    nan_count = np.isnan(X_train).sum()
    inf_count = np.isinf(X_train).sum()
    logger.info(f"NaN values: {nan_count}, Inf values: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        logger.error("⚠️  Found NaN or Inf values! This will cause model failures.")
        return
    
    # 2. Check feature variance
    feature_std = np.std(X_train, axis=0)
    zero_variance = np.sum(feature_std < 1e-10)
    low_variance = np.sum(feature_std < 1e-6)
    very_low_variance = np.sum(feature_std < 1e-4)
    
    logger.info(f"\nVariance Analysis:")
    logger.info(f"  Zero variance features (<1e-10): {zero_variance}/{len(feature_std)}")
    logger.info(f"  Low variance features (<1e-6): {low_variance}/{len(feature_std)}")
    logger.info(f"  Very low variance features (<1e-4): {very_low_variance}/{len(feature_std)}")
    logger.info(f"  Mean std: {feature_std.mean():.6f}, Min std: {feature_std.min():.6f}, Max std: {feature_std.max():.6f}")
    
    if zero_variance > 0:
        zero_var_indices = np.where(feature_std < 1e-10)[0]
        logger.warning(f"\n⚠️  Zero variance feature indices (first 20): {zero_var_indices[:20].tolist()}")
        if feature_names:
            logger.warning(f"Zero variance feature names (first 10): {[feature_names[i] for i in zero_var_indices[:10]]}")
    
    # 3. Check for perfect correlation (only if not too many features)
    if X_train.shape[1] < 2000:
        logger.info(f"\nChecking correlations (this may take a moment)...")
        # Sample features if too many
        if X_train.shape[1] > 500:
            sample_indices = np.random.choice(X_train.shape[1], 500, replace=False)
            X_sample = X_train[:, sample_indices]
            corr_matrix = np.corrcoef(X_sample.T)
        else:
            corr_matrix = np.corrcoef(X_train.T)
            sample_indices = np.arange(X_train.shape[1])
        
        # Find near-perfect correlations (excluding diagonal)
        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
        high_corr = np.sum(np.abs(corr_matrix[mask]) > 0.99)
        very_high_corr = np.sum(np.abs(corr_matrix[mask]) > 0.999)
        
        logger.info(f"  High correlations (>0.99): {high_corr}")
        logger.info(f"  Very high correlations (>0.999): {very_high_corr}")
        
        if very_high_corr > 0:
            logger.warning(f"⚠️  Found {very_high_corr} near-perfect correlations! This can cause numerical issues.")
    
    # 4. Check model predictions
    logger.info(f"\nTesting RandomForest model...")
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"\nPrediction Analysis:")
    logger.info(f"  Train prediction range: [{train_preds.min():.4f}, {train_preds.max():.4f}]")
    logger.info(f"  Test prediction range: [{test_preds.min():.4f}, {test_preds.max():.4f}]")
    logger.info(f"  Train prediction mean: {train_preds.mean():.4f}, std: {train_preds.std():.4f}")
    logger.info(f"  Test prediction mean: {test_preds.mean():.4f}, std: {test_preds.std():.4f}")
    
    # Check if all predictions are ~0.5
    if np.abs(test_preds.mean() - 0.5) < 0.01 and test_preds.std() < 0.01:
        logger.error("⚠️  All predictions are ~0.5! Model is not learning.")
    elif test_preds.std() < 0.05:
        logger.warning(f"⚠️  Very low prediction variance ({test_preds.std():.4f}). Model may not be learning well.")
    
    # Calculate PR-AUC
    try:
        pr_auc = average_precision_score(y_test, test_preds)
        logger.info(f"  Test PR-AUC: {pr_auc:.4f}")
        if pr_auc < 0.51:
            logger.error("⚠️  PR-AUC is near random (0.5)! Model is not learning.")
    except Exception as e:
        logger.error(f"  Failed to calculate PR-AUC: {e}")
    
    # 5. Feature importance
    importances = model.feature_importances_
    top_features = np.argsort(importances)[-20:][::-1]
    logger.info(f"\nTop 20 Feature Importances:")
    logger.info(f"  Range: [{importances.min():.6f}, {importances.max():.6f}]")
    logger.info(f"  Mean: {importances.mean():.6f}")
    logger.info(f"  Top 10 values: {importances[top_features[:10]].tolist()}")
    
    if feature_names:
        logger.info(f"  Top 10 feature names:")
        for idx in top_features[:10]:
            logger.info(f"    {feature_names[idx]}: {importances[idx]:.6f}")
    
    # Check if importances are uniform (bad sign)
    if importances.std() < 1e-6:
        logger.error("⚠️  Feature importances are uniform! Model is not learning from features.")
    
    # 6. Check class balance
    logger.info(f"\nClass Balance:")
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    logger.info(f"  Train: {dict(zip(range(len(train_counts)), train_counts))}")
    logger.info(f"  Test: {dict(zip(range(len(test_counts)), test_counts))}")
    
    # 7. Check feature-to-sample ratio
    ratio = X_train.shape[1] / X_train.shape[0]
    logger.info(f"\nFeature-to-Sample Ratio: {ratio:.2f}")
    if ratio > 1.0:
        logger.warning(f"⚠️  More features than samples! This can cause overfitting and numerical issues.")
    elif ratio > 0.5:
        logger.warning(f"⚠️  High feature-to-sample ratio ({ratio:.2f}). Consider feature selection.")
    
    return {
        'zero_variance_features': int(zero_variance),
        'low_variance_features': int(low_variance),
        'train_pred_range': (float(train_preds.min()), float(train_preds.max())),
        'test_pred_range': (float(test_preds.min()), float(test_preds.max())),
        'test_pred_mean': float(test_preds.mean()),
        'test_pred_std': float(test_preds.std()),
        'pr_auc': float(pr_auc) if 'pr_auc' in locals() else None,
        'feature_to_sample_ratio': float(ratio),
        'mean_feature_importance': float(importances.mean()),
        'std_feature_importance': float(importances.std())
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose feature issues")
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type")
    parser.add_argument("--use_cached_embeddings", action="store_true", help="Use cached embeddings")
    parser.add_argument("--full_graph_embeddings", action="store_true", help="Use full graph embeddings")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--embedding_method", type=str, default="ComplEx", help="Embedding method")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading data...")
    df = load_hetionet_edges()
    task_edges, entity_to_id, id_to_entity = extract_task_edges(df, relation_type=args.relation)
    train_df, test_df = prepare_link_prediction_dataset(task_edges, random_state=args.random_state)
    
    # Train/load embeddings
    logger.info("Loading embeddings...")
    embedder = AdvancedKGEmbedder(
        embedding_dim=args.embedding_dim,
        method=args.embedding_method,
        work_dir="data",
        random_state=args.random_state
    )
    
    if args.use_cached_embeddings and embedder.load_embeddings():
        logger.info("Using cached embeddings")
    else:
        if args.full_graph_embeddings:
            from kg_layer.kg_loader import prepare_full_graph_for_embeddings
            task_entities = list(entity_to_id.keys())
            embedding_training_edges = prepare_full_graph_for_embeddings(df, task_entities)
            embedding_training_edges["source_id"] = embedding_training_edges["source"].map(entity_to_id)
            embedding_training_edges["target_id"] = embedding_training_edges["target"].map(entity_to_id)
            embedding_training_edges = embedding_training_edges.dropna(subset=["source_id", "target_id"])
            embedding_training_edges["source_id"] = embedding_training_edges["source_id"].astype(int)
            embedding_training_edges["target_id"] = embedding_training_edges["target_id"].astype(int)
        else:
            embedding_training_edges = task_edges
        
        embedder.train_embeddings(embedding_training_edges)
    
    embeddings = embedder.get_all_embeddings()
    
    # Build enhanced features
    logger.info("Building enhanced features...")
    feature_builder = EnhancedFeatureBuilder(
        include_graph_features=True,
        include_domain_features=True,
        normalize=True
    )
    
    train_edges_only = train_df[train_df['label'] == 1].copy()
    train_edges_only['source'] = train_edges_only['source_id'].map(id_to_entity)
    train_edges_only['target'] = train_edges_only['target_id'].map(id_to_entity)
    
    feature_builder.build_graph(train_edges_only)
    
    X_train, feature_names = feature_builder.build_features(
        train_df, embeddings, edges_df=train_edges_only, fit_scaler=True
    )
    X_test, _ = feature_builder.build_features(
        test_df, embeddings, edges_df=train_edges_only, fit_scaler=False
    )
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # Run diagnostics
    results = diagnose_features(X_train, y_train, X_test, y_test, feature_names)
    
    logger.info(f"\n{'='*80}")
    logger.info("SUMMARY")
    logger.info(f"{'='*80}")
    if results:
        logger.info(f"Zero variance features: {results['zero_variance_features']}")
        logger.info(f"Test PR-AUC: {results.get('pr_auc', 'N/A'):.4f}")
        logger.info(f"Feature-to-sample ratio: {results['feature_to_sample_ratio']:.2f}")
        logger.info(f"Prediction std: {results['test_pred_std']:.4f}")
    
    logger.info("\n✅ Diagnostics complete!")


if __name__ == "__main__":
    main()

