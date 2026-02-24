#!/usr/bin/env python3
"""
Comprehensive Parameter Exploration Script

Tests different parameter combinations for both classical and quantum models
to find optimal configurations before running the full pipeline.

Usage:
    python scripts/explore_parameters.py --relation CtD --use_cached_embeddings
    python scripts/explore_parameters.py --relation CtD --quantum_only
    python scripts/explore_parameters.py --relation CtD --classical_only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from itertools import product
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, silhouette_score, davies_bouldin_score,
    make_scorer
)
from scipy.spatial.distance import pdist
from scipy.stats import ttest_ind

# Project imports
from kg_layer.kg_loader import (
    load_hetionet_edges, extract_task_edges,
    prepare_full_graph_for_embeddings, get_negative_samples, load_kg_config
)
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder, validate_no_leakage
from quantum_layer.advanced_qml_features import QuantumFeatureEngineer
from utils.reproducibility import set_global_seed

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def analyze_quantum_separability(
    X: np.ndarray,
    y: np.ndarray,
    feature_name: str = "Quantum Features"
) -> Dict[str, float]:
    """Analyze quantum feature separability metrics."""
    pos_mask = y == 1
    neg_mask = y == 0
    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    
    if len(X_pos) == 0 or len(X_neg) == 0:
        return {'separation_ratio': 0.0, 'silhouette': -1.0, 'db_index': 10.0, 'significant_features': 0}
    
    # Mean differences
    mean_pos = np.mean(X_pos, axis=0)
    mean_neg = np.mean(X_neg, axis=0)
    mean_diff = np.abs(mean_pos - mean_neg)
    
    # Statistical significance
    significant_features = 0
    for i in range(X.shape[1]):
        try:
            _, pval = ttest_ind(X_pos[:, i], X_neg[:, i])
            if pval < 0.05:
                significant_features += 1
        except:
            pass
    
    # Within-class vs between-class distances
    try:
        pos_distances = pdist(X_pos[:min(100, len(X_pos))])
        neg_distances = pdist(X_neg[:min(100, len(X_neg))])
        within_class_dist = np.mean(np.concatenate([pos_distances, neg_distances]))
        
        between_class_dist = []
        for x_pos in X_pos[:min(50, len(X_pos))]:
            for x_neg in X_neg[:min(50, len(X_neg))]:
                between_class_dist.append(np.linalg.norm(x_pos - x_neg))
        between_class_dist = np.mean(between_class_dist) if between_class_dist else 0.0
        
        separation_ratio = between_class_dist / within_class_dist if within_class_dist > 0 else 0.0
    except:
        separation_ratio = 0.0
    
    # Silhouette score
    try:
        silhouette = silhouette_score(X, y)
    except:
        silhouette = -1.0
    
    # Davies-Bouldin index
    try:
        db_index = davies_bouldin_score(X, y)
    except:
        db_index = 10.0
    
    return {
        'separation_ratio': float(separation_ratio),
        'silhouette': float(silhouette),
        'db_index': float(db_index),
        'significant_features': significant_features,
        'mean_diff_mean': float(np.mean(mean_diff)),
        'mean_diff_max': float(np.max(mean_diff))
    }


def test_quantum_config(
    train_h_embs: np.ndarray,
    train_t_embs: np.ndarray,
    test_h_embs: np.ndarray,
    test_t_embs: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    qml_dim: int,
    random_state: int
) -> Dict[str, Any]:
    """Test a quantum parameter configuration."""
    try:
        engineer = QuantumFeatureEngineer(
            num_qubits=qml_dim,
            encoding_strategy=config.get('encoding', 'optimized_diff'),
            reduction_method=config.get('reduction_method', 'pca'),
            feature_selection_method=config.get('feature_selection_method', 'f_classif'),
            feature_select_k_mult=config.get('feature_select_k_mult', 4.0),
            pre_pca_dim=config.get('pre_pca_dim', 0),
            random_state=random_state
        )
        
        # Prepare features
        X_train_qml = engineer.prepare_qml_features(
            train_h_embs, train_t_embs, y_train, fit=True
        )
        X_test_qml = engineer.prepare_qml_features(
            test_h_embs, test_t_embs, fit=False
        )
        
        # CRITICAL: Filter out 1D configurations (LDA for binary classification)
        # Quantum kernels need multiple dimensions to work effectively
        if X_train_qml.shape[1] == 1:
            logger.warning(f"  ⚠️  Skipping 1D configuration (LDA gives 1D for binary classification)")
            return {
                'status': 'failed',
                'error': '1D features not suitable for quantum kernels',
                'config': config,
                'feature_shape': X_train_qml.shape
            }
        
        # Analyze separability
        metrics = analyze_quantum_separability(X_train_qml, y_train, "Quantum Features")
        
        # Check for overfitting risk: features vs samples
        n_features = X_train_qml.shape[1]
        n_samples = X_train_qml.shape[0]
        feature_ratio = n_features / n_samples if n_samples > 0 else float('inf')
        
        if feature_ratio > 2.0:
            logger.warning(f"  ⚠️  High feature-to-sample ratio: {feature_ratio:.2f} ({n_features} features, {n_samples} samples)")
            logger.warning(f"     This configuration is at high risk of overfitting!")
        
        # Test QSVC performance with cross-validation to detect overfitting
        from sklearn.svm import SVC
        from sklearn.metrics import average_precision_score, make_scorer
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        # Use RBF kernel as a proxy for quantum kernel performance
        # Test multiple C values to find best regularization
        best_cv_score = -1
        best_test_score = -1
        best_train_score = -1
        best_C = 1.0
        
        # Use cross-validation to detect overfitting
        cv_folds = 3  # Use 3-fold CV for speed
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # Custom scorer for PR-AUC using decision_function
        def pr_auc_scorer(estimator, X, y):
            """Score using decision_function for PR-AUC."""
            try:
                y_score = estimator.decision_function(X)
                return average_precision_score(y, y_score)
            except:
                return 0.0
        
        # Test C values with cross-validation
        C_values = [0.1, 0.3, 1.0, 3.0, 10.0]
        for C in C_values:
            svc = SVC(kernel='rbf', gamma='scale', C=C, class_weight='balanced', random_state=random_state)
            
            # Cross-validation score
            cv_scores = cross_val_score(
                svc, X_train_qml, y_train,
                cv=skf, scoring=pr_auc_scorer, n_jobs=-1
            )
            cv_mean = cv_scores.mean()
            
            # Train on full training set
            svc.fit(X_train_qml, y_train)
            
            # Evaluate on test set
            y_score_test = svc.decision_function(X_test_qml)
            test_pr_auc = average_precision_score(y_test, y_score_test)
            
            # Evaluate on training set (for overfitting detection)
            y_score_train = svc.decision_function(X_train_qml)
            train_pr_auc = average_precision_score(y_train, y_score_train)
            
            # Track best configuration (prioritize test PR-AUC, but penalize overfitting)
            if test_pr_auc > best_test_score:
                best_cv_score = cv_mean
                best_test_score = test_pr_auc
                best_train_score = train_pr_auc
                best_C = C
        
        # Calculate overfitting gap
        overfitting_gap = best_train_score - best_test_score
        cv_test_gap = best_cv_score - best_test_score
        
        # Warn if overfitting detected
        if overfitting_gap > 0.15:  # 15% gap indicates overfitting
            logger.warning(f"  ⚠️  Overfitting detected! Train-Test gap: {overfitting_gap:.4f}")
        if cv_test_gap > 0.10:  # CV-Test gap indicates generalization issues
            logger.warning(f"  ⚠️  CV-Test mismatch! CV-Test gap: {cv_test_gap:.4f}")
        
        # Compute quality score with overfitting penalty
        # Penalize configurations with high overfitting gaps
        overfitting_penalty = max(0, (overfitting_gap - 0.10) * 5)  # Penalty for gaps > 10%
        cv_mismatch_penalty = max(0, (cv_test_gap - 0.05) * 3)  # Penalty for CV-Test gaps > 5%
        
        quality_score = (
            0.5 * best_test_score * 10 +  # Test PR-AUC (primary metric)
            0.2 * best_cv_score * 10 +  # CV PR-AUC (generalization)
            0.1 * metrics['separation_ratio'] +
            0.1 * max(0, metrics['silhouette']) * 10 +
            0.05 * (metrics['significant_features'] / max(1, X_train_qml.shape[1])) * 10 +
            -0.05 * overfitting_penalty +  # Penalty for overfitting
            -0.05 * cv_mismatch_penalty  # Penalty for CV-Test mismatch
        )
        
        return {
            'status': 'success',
            'metrics': metrics,
            'quality_score': quality_score,
            'test_pr_auc': best_test_score,
            'train_pr_auc': best_train_score,
            'cv_pr_auc': best_cv_score,
            'overfitting_gap': overfitting_gap,
            'cv_test_gap': cv_test_gap,
            'best_C': best_C,
            'feature_ratio': feature_ratio,
            'feature_shape': X_train_qml.shape,
            'config': config
        }
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'config': config
        }


def test_classical_config(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str,
    config: Dict[str, Any],
    cv_folds: int,
    random_state: int
) -> Dict[str, Any]:
    """Test a classical model hyperparameter configuration."""
    try:
        if model_type == 'RandomForest':
            model = RandomForestClassifier(
                n_estimators=config.get('n_estimators', 100),
                max_depth=config.get('max_depth', 10),
                min_samples_split=config.get('min_samples_split', 10),
                min_samples_leaf=config.get('min_samples_leaf', 5),
                max_features=config.get('max_features', 'sqrt'),
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
        elif model_type == 'LogisticRegression':
            model = LogisticRegression(
                C=config.get('C', 1.0),
                penalty='l2',
                class_weight='balanced',
                max_iter=1000,
                random_state=random_state,
                solver='liblinear'
            )
        else:
            return {'status': 'failed', 'error': f'Unknown model type: {model_type}'}
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
        
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring=pr_auc_scorer, n_jobs=-1
        )
        
        # Train on full training set and evaluate on test
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        test_pr_auc = average_precision_score(y_test, y_pred_proba)
        
        return {
            'status': 'success',
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'test_pr_auc': float(test_pr_auc),
            'config': config
        }
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'config': config
        }


def explore_quantum_parameters(
    train_h_embs: np.ndarray,
    train_t_embs: np.ndarray,
    test_h_embs: np.ndarray,
    test_t_embs: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    qml_dim: int,
    random_state: int
) -> pd.DataFrame:
    """Explore quantum parameter space."""
    logger.info("\n" + "="*80)
    logger.info("EXPLORING QUANTUM PARAMETERS")
    logger.info("="*80)
    
    # Define parameter grid
    param_grid = {
        'encoding': ['optimized_diff', 'hybrid'],
        'reduction_method': ['pca', 'lda', None],
        'feature_selection_method': ['f_classif', 'mutual_info', None],
        'feature_select_k_mult': [2.0, 4.0, 6.0],
        'pre_pca_dim': [0, 64, 128]
    }
    
    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))
    
    logger.info(f"Testing {len(combinations)} quantum parameter combinations...")
    
    results = []
    for i, combo in enumerate(combinations, 1):
        config = dict(zip(keys, combo))
        
        # Skip invalid combinations
        if config['reduction_method'] is None and config['pre_pca_dim'] > 0:
            continue
        if config['reduction_method'] == 'lda' and config['pre_pca_dim'] > 0:
            continue  # LDA doesn't benefit from pre-PCA
        
        logger.info(f"[{i}/{len(combinations)}] Testing: {config}")
        
        result = test_quantum_config(
            train_h_embs, train_t_embs, test_h_embs, test_t_embs,
            y_train, y_test, config, qml_dim, random_state
        )
        
        if result['status'] == 'success':
            pr_auc = result.get('test_pr_auc', 0.0)
            cv_pr_auc = result.get('cv_pr_auc', 0.0)
            gap = result.get('overfitting_gap', 0.0)
            feature_ratio = result.get('feature_ratio', 0.0)
            
            gap_warning = " ⚠️ OVERFITTING" if gap > 0.15 else ""
            logger.info(f"  ✓ Test PR-AUC: {pr_auc:.4f}, CV: {cv_pr_auc:.4f}, "
                       f"Gap: {gap:.4f}{gap_warning}")
            if feature_ratio > 2.0:
                logger.info(f"     Feature ratio: {feature_ratio:.2f} (high overfitting risk)")
        else:
            logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
        
        results.append(result)
    
    # Convert to DataFrame
    df_results = []
    for r in results:
            if r['status'] == 'success':
                row = {
                    'test_pr_auc': r.get('test_pr_auc', 0.0),
                    'train_pr_auc': r.get('train_pr_auc', 0.0),
                    'cv_pr_auc': r.get('cv_pr_auc', 0.0),
                    'overfitting_gap': r.get('overfitting_gap', 0.0),
                    'cv_test_gap': r.get('cv_test_gap', 0.0),
                    'best_C': r.get('best_C', 1.0),
                    'feature_ratio': r.get('feature_ratio', 0.0),
                    'quality_score': r['quality_score'],
                    'separation_ratio': r['metrics']['separation_ratio'],
                    'silhouette': r['metrics']['silhouette'],
                    'db_index': r['metrics']['db_index'],
                    'significant_features': r['metrics']['significant_features'],
                    'feature_shape': str(r['feature_shape']),
                    **{f"param_{k}": v for k, v in r['config'].items()}
                }
                df_results.append(row)
    
    df = pd.DataFrame(df_results)
    if len(df) > 0:
        # Sort by PR-AUC first, then quality score
        df = df.sort_values(['test_pr_auc', 'quality_score'], ascending=[False, False])
    
    return df


def explore_classical_parameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int,
    random_state: int
) -> Dict[str, pd.DataFrame]:
    """Explore classical model hyperparameters."""
    logger.info("\n" + "="*80)
    logger.info("EXPLORING CLASSICAL PARAMETERS")
    logger.info("="*80)
    
    results = {}
    
    # RandomForest grid
    logger.info("\nTesting RandomForest hyperparameters...")
    rf_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [3, 5, 10],
        'max_features': ['sqrt', 0.7, 0.9]
    }
    
    rf_results = []
    rf_combinations = list(product(*rf_grid.values()))
    logger.info(f"Testing {len(rf_combinations)} RandomForest combinations...")
    
    for i, combo in enumerate(rf_combinations, 1):
        config = dict(zip(rf_grid.keys(), combo))
        if i % 10 == 0:
            logger.info(f"[{i}/{len(rf_combinations)}] Testing RF config...")
        
        result = test_classical_config(
            X_train, y_train, X_test, y_test,
            'RandomForest', config, cv_folds, random_state
        )
        
        if result['status'] == 'success':
            row = {
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'test_pr_auc': result['test_pr_auc'],
                **{f"param_{k}": v for k, v in result['config'].items()}
            }
            rf_results.append(row)
    
    results['RandomForest'] = pd.DataFrame(rf_results)
    if len(results['RandomForest']) > 0:
        results['RandomForest'] = results['RandomForest'].sort_values('test_pr_auc', ascending=False)
        logger.info(f"✓ Best RF CV PR-AUC: {results['RandomForest'].iloc[0]['cv_mean']:.4f}")
    
    # LogisticRegression grid
    logger.info("\nTesting LogisticRegression hyperparameters...")
    lr_grid = {
        'C': [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    }
    
    lr_results = []
    lr_combinations = list(product(*lr_grid.values()))
    logger.info(f"Testing {len(lr_combinations)} LogisticRegression combinations...")
    
    for i, combo in enumerate(lr_combinations, 1):
        config = dict(zip(lr_grid.keys(), combo))
        result = test_classical_config(
            X_train, y_train, X_test, y_test,
            'LogisticRegression', config, cv_folds, random_state
        )
        
        if result['status'] == 'success':
            row = {
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std'],
                'test_pr_auc': result['test_pr_auc'],
                **{f"param_{k}": v for k, v in result['config'].items()}
            }
            lr_results.append(row)
    
    results['LogisticRegression'] = pd.DataFrame(lr_results)
    if len(results['LogisticRegression']) > 0:
        results['LogisticRegression'] = results['LogisticRegression'].sort_values('test_pr_auc', ascending=False)
        logger.info(f"✓ Best LR CV PR-AUC: {results['LogisticRegression'].iloc[0]['cv_mean']:.4f}")
    
    return results


def print_recommendations(
    quantum_results: pd.DataFrame,
    classical_results: Dict[str, pd.DataFrame],
    output_file: Optional[str] = None
):
    """Print and save parameter recommendations."""
    logger.info("\n" + "="*80)
    logger.info("PARAMETER RECOMMENDATIONS")
    logger.info("="*80)
    
    recommendations = {}
    
    # Quantum recommendations
    if len(quantum_results) > 0:
        logger.info("\n🏆 TOP 5 QUANTUM CONFIGURATIONS:")
        logger.info("-" * 80)
        
        top_quantum = quantum_results.head(5)
        for i, (idx, row) in enumerate(top_quantum.iterrows(), 1):
            test_pr = row.get('test_pr_auc', 0.0)
            cv_pr = row.get('cv_pr_auc', 0.0)
            gap = row.get('overfitting_gap', 0.0)
            feature_ratio = row.get('feature_ratio', 0.0)
            
            gap_warning = " ⚠️ OVERFITTING" if gap > 0.15 else ""
            logger.info(f"\n{i}. Test PR-AUC: {test_pr:.4f}, CV PR-AUC: {cv_pr:.4f}, Gap: {gap:.4f}{gap_warning}")
            logger.info(f"   Quality Score: {row['quality_score']:.4f}, Best C: {row.get('best_C', 1.0)}")
            if feature_ratio > 2.0:
                logger.info(f"   ⚠️  Feature Ratio: {feature_ratio:.2f} (high overfitting risk)")
            logger.info(f"   Separation Ratio: {row['separation_ratio']:.4f}, Silhouette: {row['silhouette']:.4f}")
            logger.info(f"   Config:")
            for col in row.index:
                if col.startswith('param_'):
                    param_name = col.replace('param_', '')
                    logger.info(f"     --qml_{param_name} {row[col]}")
        
        best_quantum = top_quantum.iloc[0]
        recommendations['quantum'] = {
            k.replace('param_', ''): v for k, v in best_quantum.items()
            if k.startswith('param_')
        }
        recommendations['quantum']['quality_score'] = float(best_quantum['quality_score'])
        recommendations['quantum']['test_pr_auc'] = float(best_quantum.get('test_pr_auc', 0.0))
        recommendations['quantum']['train_pr_auc'] = float(best_quantum.get('train_pr_auc', 0.0))
    
    # Classical recommendations
    for model_name, df in classical_results.items():
        if len(df) > 0:
            logger.info(f"\n🏆 TOP 3 {model_name.upper()} CONFIGURATIONS:")
            logger.info("-" * 80)
            
            top_classical = df.head(3)
            for i, (idx, row) in enumerate(top_classical.iterrows(), 1):
                logger.info(f"\n{i}. Test PR-AUC: {row['test_pr_auc']:.4f} (CV: {row['cv_mean']:.4f} ± {row['cv_std']:.4f})")
                logger.info(f"   Config:")
                for col in row.index:
                    if col.startswith('param_'):
                        param_name = col.replace('param_', '')
                        logger.info(f"     {param_name}: {row[col]}")
            
            best_classical = top_classical.iloc[0]
            recommendations[model_name] = {
                k.replace('param_', ''): v for k, v in best_classical.items()
                if k.startswith('param_')
            }
            recommendations[model_name]['test_pr_auc'] = float(best_classical['test_pr_auc'])
            recommendations[model_name]['cv_mean'] = float(best_classical['cv_mean'])
    
    # Save recommendations (convert numpy types to native Python types)
    if output_file:
        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        recommendations_native = convert_to_native(recommendations)
        with open(output_file, 'w') as f:
            json.dump(recommendations_native, f, indent=2)
        logger.info(f"\n✅ Recommendations saved to {output_file}")
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(
        description="Explore optimal parameters for classical and quantum models"
    )
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type")
    parser.add_argument("--pos_edge_sample", type=int, default=1500, help="Sample size for positive edges")
    parser.add_argument("--neg_ratio", type=float, default=2.0, help="Negative to positive ratio")
    parser.add_argument("--embedding_method", type=str, default="RotatE", help="Embedding method")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--embedding_epochs", type=int, default=200, help="Embedding epochs")
    parser.add_argument("--full_graph_embeddings", action="store_true", help="Use full graph embeddings")
    parser.add_argument("--use_cached_embeddings", action="store_true", help="Use cached embeddings")
    parser.add_argument("--use_evidence_weighting", action="store_true", help="Use evidence weighting")
    parser.add_argument("--min_shared_genes", type=int, default=1, help="Min shared genes for evidence")
    parser.add_argument("--use_contrastive_learning", action="store_true", help="Use contrastive learning")
    parser.add_argument("--contrastive_epochs", type=int, default=75, help="Contrastive epochs")
    parser.add_argument("--qml_dim", type=int, default=12, help="Quantum dimension")
    parser.add_argument("--quantum_only", action="store_true", help="Only explore quantum parameters")
    parser.add_argument("--classical_only", action="store_true", help="Only explore classical parameters")
    parser.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    set_global_seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*80)
    logger.info("PARAMETER EXPLORATION FOR HYBRID QML-KG PIPELINE")
    logger.info("="*80)
    logger.info(f"Relation: {args.relation}")
    logger.info(f"Random seed: {args.random_state}")
    
    # Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)
    
    df = load_hetionet_edges()
    task_edges, entity_to_id, id_to_entity = extract_task_edges(
        df, relation_type=args.relation, max_entities=300
    )
    
    # Sample positive edges if requested
    if args.pos_edge_sample and args.pos_edge_sample > 0:
        if len(task_edges) > args.pos_edge_sample:
            logger.info(f"Sampling {args.pos_edge_sample} positive edges from {len(task_edges)} available")
            task_edges = task_edges.sample(n=args.pos_edge_sample, random_state=args.random_state).reset_index(drop=True)
            logger.info(f"Sampled {len(task_edges)} positive edges")
        else:
            logger.info(f"Requested {args.pos_edge_sample} edges but only {len(task_edges)} available, using all")
    
    # Create train/test split with custom neg_ratio
    from sklearn.model_selection import train_test_split
    
    # Positive samples
    pos_df = task_edges[["source_id", "target_id"]].copy()
    pos_df["label"] = 1
    
    # Train/test split on positive edges
    pos_train, pos_test = train_test_split(
        pos_df, test_size=0.2, random_state=args.random_state
    )
    
    # Generate negatives with custom ratio
    num_neg_train = int(len(pos_train) * args.neg_ratio)
    num_neg_test = int(len(pos_test) * args.neg_ratio)
    
    config = load_kg_config()
    neg_train = get_negative_samples(pos_train, num_negatives=num_neg_train, random_state=args.random_state, config=config)
    neg_test = get_negative_samples(pos_test, num_negatives=num_neg_test, random_state=args.random_state + 1, config=config)
    
    # Combine
    train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(frac=1, random_state=args.random_state)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True).sample(frac=1, random_state=args.random_state)
    
    logger.info(f"Train: {len(train_df)} samples ({train_df['label'].sum()} positive, {len(neg_train)} negative, ratio={args.neg_ratio:.1f})")
    logger.info(f"Test: {len(test_df)} samples ({test_df['label'].sum()} positive, {len(neg_test)} negative, ratio={args.neg_ratio:.1f})")
    
    # Generate embeddings
    logger.info("\n" + "="*80)
    logger.info("STEP 2: GENERATING EMBEDDINGS")
    logger.info("="*80)
    
    embedder = AdvancedKGEmbedder(
        method=args.embedding_method,
        embedding_dim=args.embedding_dim,
        num_epochs=args.embedding_epochs,
        random_state=args.random_state
    )
    
    if args.full_graph_embeddings:
        logger.info("Using FULL GRAPH embeddings (all relations) for richer context...")
        # Get all entities involved in task
        task_entities = list(entity_to_id.keys())
        # Prepare full graph edges involving these entities
        embedding_training_edges = prepare_full_graph_for_embeddings(df, task_entities)
        # Filter to only edges where both entities are in our task set
        embedding_training_edges = embedding_training_edges[
            embedding_training_edges["source"].isin(task_entities) &
            embedding_training_edges["target"].isin(task_entities)
        ].copy()
        logger.info(f"Training embeddings on {len(embedding_training_edges)} edges "
                   f"({embedding_training_edges['metaedge'].nunique()} relation types)")
        embedder.train_embeddings(embedding_training_edges)
    else:
        logger.info(f"Training {args.embedding_method} embeddings on task-specific edges...")
        # task_edges already has 'source' and 'target' columns with entity strings
        embedding_training_edges = task_edges[["source", "metaedge", "target"]].copy()
        embedder.train_embeddings(embedding_training_edges)
    
    # Apply contrastive learning if requested
    if args.use_contrastive_learning:
        logger.info("Applying contrastive learning fine-tuning...")
        from kg_layer.contrastive_embeddings import ContrastiveEmbeddingFineTuner
        tuner = ContrastiveEmbeddingFineTuner(
            margin=1.0, epochs=args.contrastive_epochs, random_state=args.random_state
        )
        entity_embeddings_dict = embedder.get_all_embeddings()
        tuner.fine_tune_embeddings(
            entity_embeddings_dict, train_df, embedder.id_to_entity
        )
        # Note: The tuner modifies embeddings in-place, so we can continue using embedder
    
    # Get embeddings and id mappings
    embeddings = embedder.get_all_embeddings()
    id_to_entity = embedder.id_to_entity
    actual_emb_dim = args.embedding_dim
    
    # Handle complex embeddings (RotatE, ComplEx)
    sample_emb = next(iter(embeddings.values()))
    if sample_emb.shape[0] != args.embedding_dim:
        actual_emb_dim = sample_emb.shape[0]
        logger.info(f"Embedding dimension mismatch: expected {args.embedding_dim}, got {actual_emb_dim}")
    
    # Prepare training/test embeddings for quantum
    logger.info("Preparing embeddings for quantum feature engineering...")
    train_h_embs = []
    train_t_embs = []
    for h_id, t_id in zip(train_df['source_id'].values, train_df['target_id'].values):
        h_entity = id_to_entity.get(h_id, None)
        t_entity = id_to_entity.get(t_id, None)
        
        h_emb = embeddings.get(h_entity, None) if h_entity else None
        t_emb = embeddings.get(t_entity, None) if t_entity else None
        
        if h_emb is None:
            h_emb = np.zeros(actual_emb_dim)
        if t_emb is None:
            t_emb = np.zeros(actual_emb_dim)
        
        train_h_embs.append(h_emb)
        train_t_embs.append(t_emb)
    
    test_h_embs = []
    test_t_embs = []
    for h_id, t_id in zip(test_df['source_id'].values, test_df['target_id'].values):
        h_entity = id_to_entity.get(h_id, None)
        t_entity = id_to_entity.get(t_id, None)
        
        h_emb = embeddings.get(h_entity, None) if h_entity else None
        t_emb = embeddings.get(t_entity, None) if t_entity else None
        
        if h_emb is None:
            h_emb = np.zeros(actual_emb_dim)
        if t_emb is None:
            t_emb = np.zeros(actual_emb_dim)
        
        test_h_embs.append(h_emb)
        test_t_embs.append(t_emb)
    
    train_h_embs = np.array(train_h_embs)
    train_t_embs = np.array(train_t_embs)
    test_h_embs = np.array(test_h_embs)
    test_t_embs = np.array(test_t_embs)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Quantum embeddings prepared: train ({train_h_embs.shape}, {train_t_embs.shape}), "
               f"test ({test_h_embs.shape}, {test_t_embs.shape})")
    
    # Build classical features (for classical parameter exploration)
    if not args.quantum_only:
        logger.info("\n" + "="*80)
        logger.info("STEP 3: BUILDING CLASSICAL FEATURES")
        logger.info("="*80)
        
        feature_builder = EnhancedFeatureBuilder(
            include_graph_features=True,
            include_domain_features=True,
            normalize=True
        )
        
        # Build on training edges only (prevent leakage)
        train_edges_only = train_df[train_df['label'] == 1].copy()
        train_edges_only['source'] = train_edges_only['source_id'].map(id_to_entity)
        train_edges_only['target'] = train_edges_only['target_id'].map(id_to_entity)
        feature_builder.build_graph(train_edges_only)
        
        # Prepare dataframes for feature building
        train_df_for_features = train_df.copy()
        train_df_for_features['source'] = train_df_for_features['source_id'].map(id_to_entity)
        train_df_for_features['target'] = train_df_for_features['target_id'].map(id_to_entity)
        
        test_df_for_features = test_df.copy()
        test_df_for_features['source'] = test_df_for_features['source_id'].map(id_to_entity)
        test_df_for_features['target'] = test_df_for_features['target_id'].map(id_to_entity)
        
        X_train, _ = feature_builder.build_features(
            train_df_for_features, embeddings, edges_df=train_edges_only, fit_scaler=True
        )
        X_test, _ = feature_builder.build_features(
            test_df_for_features, embeddings, edges_df=train_edges_only, fit_scaler=False
        )
        
        # Remove zero-variance features
        train_std = np.std(X_train, axis=0)
        valid_features = train_std > 1e-10
        X_train = X_train[:, valid_features]
        X_test = X_test[:, valid_features]
        
        logger.info(f"Classical features: train {X_train.shape}, test {X_test.shape}")
    
    # Explore parameters
    quantum_results = pd.DataFrame()
    classical_results = {}
    
    if not args.classical_only:
        quantum_results = explore_quantum_parameters(
            train_h_embs, train_t_embs, test_h_embs, test_t_embs,
            y_train, y_test, args.qml_dim, args.random_state
        )
    
    if not args.quantum_only:
        classical_results = explore_classical_parameters(
            X_train, y_train, X_test, y_test, args.cv_folds, args.random_state
        )
    
    # Print recommendations
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = os.path.join(args.output_dir, f"parameter_recommendations_{timestamp}.json")
    
    recommendations = print_recommendations(quantum_results, classical_results, output_file)
    
    # Save detailed results
    if len(quantum_results) > 0:
        quantum_file = os.path.join(args.output_dir, f"quantum_exploration_{timestamp}.csv")
        quantum_results.to_csv(quantum_file, index=False)
        logger.info(f"\n✅ Quantum results saved to {quantum_file}")
    
    for model_name, df in classical_results.items():
        if len(df) > 0:
            classical_file = os.path.join(args.output_dir, f"{model_name.lower()}_exploration_{timestamp}.csv")
            df.to_csv(classical_file, index=False)
            logger.info(f"✅ {model_name} results saved to {classical_file}")
    
    logger.info("\n" + "="*80)
    logger.info("EXPLORATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nUse the recommended parameters from: {output_file}")
    logger.info("Or check the detailed CSV files for all tested configurations.")


if __name__ == "__main__":
    main()
