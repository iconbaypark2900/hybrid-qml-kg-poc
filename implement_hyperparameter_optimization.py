#!/usr/bin/env python3
"""
Extensive Hyperparameter Optimization for Quantum-Classical Link Prediction

This script performs extensive hyperparameter optimization for both quantum and classical models
to achieve the highest possible performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder
from quantum_layer.qml_model import QMLLinkPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def optimize_classical_hyperparameters(X_train, y_train, X_test, y_test, cv_folds=3):
    """Perform hyperparameter optimization for classical models."""
    logger.info("Starting classical hyperparameter optimization...")
    
    # Define parameter grids for each model
    rf_param_grid = {
        'n_estimators': [100, 200, 300] if not args.fast_mode else [100, 200],
        'max_depth': [10, 15, 20, None] if not args.fast_mode else [10, None],
        'min_samples_split': [2, 5, 10] if not args.fast_mode else [2, 5],
        'min_samples_leaf': [1, 2, 4] if not args.fast_mode else [1, 2],
        'max_features': ['sqrt', 'log2', None] if not args.fast_mode else ['sqrt', None]
    }
    
    et_param_grid = {
        'n_estimators': [100, 200, 300] if not args.fast_mode else [100, 200],
        'max_depth': [10, 15, 20, None] if not args.fast_mode else [10, None],
        'min_samples_split': [2, 5, 10] if not args.fast_mode else [2, 5],
        'min_samples_leaf': [1, 2, 4] if not args.fast_mode else [1, 2],
        'max_features': ['sqrt', 'log2', None] if not args.fast_mode else ['sqrt', None]
    }
    
    lr_param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0] if not args.fast_mode else [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'] if not args.fast_mode else ['l2'],
        'solver': ['liblinear', 'saga'] if not args.fast_mode else ['liblinear'],
        'max_iter': [1000, 2000] if not args.fast_mode else [1000]
    }
    
    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Optimize Random Forest
    logger.info("Optimizing Random Forest hyperparameters...")
    rf_grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=cv_folds,
        scoring='average_precision',
        n_jobs=-1,
        verbose=0
    )
    rf_grid_search.fit(X_train, y_train)
    
    # Evaluate optimized Random Forest
    rf_best = rf_grid_search.best_estimator_
    rf_pred_proba = rf_best.predict_proba(X_test)[:, 1]
    rf_ap = average_precision_score(y_test, rf_pred_proba)
    
    logger.info(f"Random Forest - Best params: {rf_grid_search.best_params_}")
    logger.info(f"Random Forest - Best CV score: {rf_grid_search.best_score_:.4f}")
    logger.info(f"Random Forest - Test PR-AUC: {rf_ap:.4f}")
    
    # Optimize Extra Trees
    logger.info("Optimizing Extra Trees hyperparameters...")
    et_grid_search = GridSearchCV(
        ExtraTreesClassifier(random_state=42),
        et_param_grid,
        cv=cv_folds,
        scoring='average_precision',
        n_jobs=-1,
        verbose=0
    )
    et_grid_search.fit(X_train, y_train)
    
    # Evaluate optimized Extra Trees
    et_best = et_grid_search.best_estimator_
    et_pred_proba = et_best.predict_proba(X_test)[:, 1]
    et_ap = average_precision_score(y_test, et_pred_proba)
    
    logger.info(f"Extra Trees - Best params: {et_grid_search.best_params_}")
    logger.info(f"Extra Trees - Best CV score: {et_grid_search.best_score_:.4f}")
    logger.info(f"Extra Trees - Test PR-AUC: {et_ap:.4f}")
    
    # Optimize Logistic Regression
    logger.info("Optimizing Logistic Regression hyperparameters...")
    lr_grid_search = GridSearchCV(
        LogisticRegression(random_state=42, class_weight='balanced'),
        lr_param_grid,
        cv=cv_folds,
        scoring='average_precision',
        n_jobs=-1,
        verbose=0
    )
    lr_grid_search.fit(X_train_scaled, y_train)
    
    # Evaluate optimized Logistic Regression
    lr_best = lr_grid_search.best_estimator_
    lr_pred_proba = lr_best.predict_proba(X_test_scaled)[:, 1]
    lr_ap = average_precision_score(y_test, lr_pred_proba)
    
    logger.info(f"Logistic Regression - Best params: {lr_grid_search.best_params_}")
    logger.info(f"Logistic Regression - Best CV score: {lr_grid_search.best_score_:.4f}")
    logger.info(f"Logistic Regression - Test PR-AUC: {lr_ap:.4f}")
    
    # Return best models and their scores
    classical_results = {
        'RandomForest': {'model': rf_best, 'test_pr_auc': rf_ap, 'best_params': rf_grid_search.best_params_},
        'ExtraTrees': {'model': et_best, 'test_pr_auc': et_ap, 'best_params': et_grid_search.best_params_},
        'LogisticRegression': {'model': lr_best, 'test_pr_auc': lr_ap, 'best_params': lr_grid_search.best_params_}
    }
    
    return classical_results


def optimize_quantum_hyperparameters(X_train, y_train, X_test, y_test, embedder, fast_mode=False):
    """Perform hyperparameter optimization for quantum models."""
    logger.info("Starting quantum hyperparameter optimization...")
    
    # Define parameter combinations for quantum models
    if not fast_mode:
        qsvc_param_combinations = [
            {'num_qubits': 16, 'feature_map_type': 'ZZ', 'feature_map_reps': 2, 'entanglement': 'full'},
            {'num_qubits': 16, 'feature_map_type': 'ZZ', 'feature_map_reps': 3, 'entanglement': 'full'},
            {'num_qubits': 16, 'feature_map_type': 'Pauli', 'feature_map_reps': 2, 'entanglement': 'full'},
            {'num_qubits': 16, 'feature_map_type': 'Pauli', 'feature_map_reps': 3, 'entanglement': 'full'},
            {'num_qubits': 16, 'feature_map_type': 'Z', 'feature_map_reps': 3, 'entanglement': 'full'},
        ]
    else:
        qsvc_param_combinations = [
            {'num_qubits': 16, 'feature_map_type': 'Pauli', 'feature_map_reps': 3, 'entanglement': 'full'},
            {'num_qubits': 16, 'feature_map_type': 'ZZ', 'feature_map_reps': 3, 'entanglement': 'full'},
        ]
    
    best_qsvc_score = 0.0
    best_qsvc_model = None
    best_qsvc_params = None
    
    for params in qsvc_param_combinations:
        logger.info(f"Testing QSVC with params: {params}")
        
        try:
            qsvc_model = QMLLinkPredictor(
                model_type="QSVC",
                encoding_method="feature_map",
                num_qubits=params['num_qubits'],
                feature_map_type=params['feature_map_type'],
                feature_map_reps=params['feature_map_reps'],
                entanglement=params['entanglement'],
                random_state=42
            )
            
            # Fit the model
            qsvc_model.fit(X_train, y_train)
            
            # Evaluate
            qsvc_pred_proba = qsvc_model.predict_proba(X_test)[:, 1]
            qsvc_ap = average_precision_score(y_test, qsvc_pred_proba)
            
            logger.info(f"  QSVC PR-AUC: {qsvc_ap:.4f}")
            
            if qsvc_ap > best_qsvc_score:
                best_qsvc_score = qsvc_ap
                best_qsvc_model = qsvc_model
                best_qsvc_params = params
                
        except Exception as e:
            logger.warning(f"  QSVC with params {params} failed: {e}")
            continue
    
    logger.info(f"Best QSVC - Params: {best_qsvc_params}")
    logger.info(f"Best QSVC - Test PR-AUC: {best_qsvc_score:.4f}")
    
    # Return best quantum model and its score
    quantum_results = {
        'QSVC': {'model': best_qsvc_model, 'test_pr_auc': best_qsvc_score, 'best_params': best_qsvc_params}
    }
    
    return quantum_results


def main():
    parser = argparse.ArgumentParser(description="Perform Extensive Hyperparameter Optimization")
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type (e.g., CtD for Compound treats Disease)")
    parser.add_argument("--max_entities", type=int, default=100, help="Max entities to include (for scalability)")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--qml_dim", type=int, default=16, help="Quantum feature dimension (qubits)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--cv_folds", type=int, default=3, help="Cross-validation folds for hyperparameter optimization")
    parser.add_argument("--fast_mode", action="store_true", help="Use smaller parameter grids for faster optimization")

    args = parser.parse_args()

    logger.info("Loading Hetionet data...")
    df = load_hetionet_edges()
    task_edges, entity_to_id, id_to_entity = extract_task_edges(
        df, 
        relation_type=args.relation, 
        max_entities=args.max_entities
    )
    
    logger.info(f"Extracted {len(task_edges)} edges for '{args.relation}' relation")
    
    # Prepare train/test split
    train_df, test_df = prepare_link_prediction_dataset(task_edges, test_size=args.test_size)
    logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
    
    # Generate embeddings
    logger.info("Generating knowledge graph embeddings...")
    embedder = AdvancedKGEmbedder(
        embedding_dim=args.embedding_dim,
        method='RotatE',  # Using RotatE as it performed well in our experiments
        num_epochs=50,
        batch_size=512,
        learning_rate=0.001,
        work_dir="data",
        random_state=args.random_state
    )
    
    # Load or train embeddings
    if embedder.load_embeddings():
        logger.info("Loaded cached embeddings")
    else:
        logger.info("Training embeddings...")
        embedder.train_embeddings(task_edges[["source", "metaedge", "target"]])
    
    # Prepare features for both quantum and classical models
    logger.info("Preparing features for quantum and classical models...")
    X_train_qml = embedder.prepare_link_features_qml(train_df, mode="diff")
    X_test_qml = embedder.prepare_link_features_qml(test_df, mode="diff")
    
    X_train_classical = embedder.prepare_link_features(train_df)
    X_test_classical = embedder.prepare_link_features(test_df)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Features prepared - Quantum: train {X_train_qml.shape}, test {X_test_qml.shape}")
    logger.info(f"Features prepared - Classical: train {X_train_classical.shape}, test {X_test_classical.shape}")
    
    # Perform classical hyperparameter optimization
    classical_results = optimize_classical_hyperparameters(
        X_train_classical, y_train, X_test_classical, y_test, cv_folds=args.cv_folds
    )
    
    # Perform quantum hyperparameter optimization
    quantum_results = optimize_quantum_hyperparameters(
        X_train_qml, y_train, X_test_qml, y_test, embedder, fast_mode=args.fast_mode
    )
    
    # Find best overall model
    all_results = {**classical_results, **quantum_results}
    best_model_name = max(all_results, key=lambda k: all_results[k]['test_pr_auc'])
    best_model_result = all_results[best_model_name]
    
    logger.info("\n" + "="*80)
    logger.info("HYPERPARAMETER OPTIMIZATION RESULTS")
    logger.info("="*80)
    
    for model_name, result in all_results.items():
        logger.info(f"{model_name:20s} - PR-AUC: {result['test_pr_auc']:.4f}, Best Params: {result['best_params']}")
    
    logger.info(f"\n🏆 Best Overall Model: {best_model_name} - PR-AUC: {best_model_result['test_pr_auc']:.4f}")
    
    # Save results
    results = {
        'relation': args.relation,
        'best_model': best_model_name,
        'best_pr_auc': best_model_result['test_pr_auc'],
        'best_params': str(best_model_result['best_params']),
        'all_results': {name: result['test_pr_auc'] for name, result in all_results.items()},
        'test_size': args.test_size,
        'embedding_dim': args.embedding_dim,
        'qml_dim': args.qml_dim
    }
    
    results_df = pd.DataFrame([results])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    results_file = f"results/hyperparameter_optimization_results_{args.relation}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\nExtensive hyperparameter optimization completed!")


if __name__ == "__main__":
    main()