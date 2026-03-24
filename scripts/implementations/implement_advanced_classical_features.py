#!/usr/bin/env python3
"""
Implement Advanced Classical Feature Engineering

This script implements advanced classical feature engineering techniques to improve
the performance of classical models, aiming to achieve PR-AUC in the 80s-90s range.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.calibration import CalibratedClassifierCV
import warnings

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Implement Advanced Classical Feature Engineering")
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type (e.g., CtD for Compound treats Disease)")
    parser.add_argument("--max_entities", type=int, default=100, help="Max entities to include (for scalability)")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--feature_engineering_method", type=str, default="all", 
                        choices=["basic", "advanced", "interaction", "all"],
                        help="Method for feature engineering")
    parser.add_argument("--feature_selection_method", type=str, default="mutual_info", 
                        choices=["mutual_info", "rfe", "select_from_model", "none"],
                        help="Method for feature selection")
    parser.add_argument("--num_features_to_select", type=int, default=100, 
                        help="Number of features to select (if using feature selection)")

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
    
    # Prepare enhanced features with advanced engineering
    logger.info("Preparing enhanced features with advanced engineering...")
    from kg_layer.enhanced_features import EnhancedFeatureBuilder
    
    feature_builder = EnhancedFeatureBuilder(
        include_graph_features=True,
        include_domain_features=True,
        normalize=True
    )
    
    # Build graph for graph features (TRAIN ONLY to prevent leakage)
    train_edges_only = train_df[train_df['label'] == 1].copy()
    feature_builder.build_graph(train_edges_only)
    
    # Build features for training set
    X_train_basic = feature_builder.build_features(train_df, embedder.get_all_embeddings(), edges_df=train_edges_only)
    X_test_basic = feature_builder.build_features(test_df, embedder.get_all_embeddings(), edges_df=train_edges_only)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Basic features prepared - Train: {X_train_basic.shape}, Test: {X_test_basic.shape}")
    
    # Apply advanced feature engineering based on the method selected
    if args.feature_engineering_method in ["advanced", "all"]:
        logger.info("Applying advanced feature engineering...")
        
        # 1. Polynomial features
        from sklearn.preprocessing import PolynomialFeatures
        poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train_basic[:, :min(50, X_train_basic.shape[1])])  # Only first 50 features to avoid explosion
        X_test_poly = poly_features.transform(X_test_basic[:, :min(50, X_test_basic.shape[1])])
        
        # 2. Statistical features
        logger.info("Computing statistical features...")
        # Calculate statistics across embedding dimensions for each entity pair
        emb_dim = args.embedding_dim
        if X_train_basic.shape[1] >= 2 * emb_dim:
            emb_train = np.hstack([X_train_basic[:, :emb_dim], X_train_basic[:, emb_dim:2*emb_dim]])
            emb_test = np.hstack([X_test_basic[:, :emb_dim], X_test_basic[:, emb_dim:2*emb_dim]])
            
            # Statistical features for embeddings
            emb_mean_train = np.mean(emb_train, axis=1, keepdims=True)
            emb_std_train = np.std(emb_train, axis=1, keepdims=True)
            emb_max_train = np.max(emb_train, axis=1, keepdims=True)
            emb_min_train = np.min(emb_train, axis=1, keepdims=True)
            
            emb_mean_test = np.mean(emb_test, axis=1, keepdims=True)
            emb_std_test = np.std(emb_test, axis=1, keepdims=True)
            emb_max_test = np.max(emb_test, axis=1, keepdims=True)
            emb_min_test = np.min(emb_test, axis=1, keepdims=True)
        else:
            # If embedding dimensions are not available, use all features for stats
            emb_mean_train = np.mean(X_train_basic, axis=1, keepdims=True)
            emb_std_train = np.std(X_train_basic, axis=1, keepdims=True)
            emb_max_train = np.max(X_train_basic, axis=1, keepdims=True)
            emb_min_train = np.min(X_train_basic, axis=1, keepdims=True)
            
            emb_mean_test = np.mean(X_test_basic, axis=1, keepdims=True)
            emb_std_test = np.std(X_test_basic, axis=1, keepdims=True)
            emb_max_test = np.max(X_test_basic, axis=1, keepdims=True)
            emb_min_test = np.min(X_test_basic, axis=1, keepdims=True)
        
        # Combine all features
        X_train_enhanced = np.hstack([
            X_train_basic,
            X_train_poly[:, min(50, X_train_basic.shape[1]):],  # Exclude original features from poly expansion
            emb_mean_train,
            emb_std_train,
            emb_max_train,
            emb_min_train
        ])
        
        X_test_enhanced = np.hstack([
            X_test_basic,
            X_test_poly[:, min(50, X_test_basic.shape[1]):],  # Exclude original features from poly expansion
            emb_mean_test,
            emb_std_test,
            emb_max_test,
            emb_min_test
        ])
        
        logger.info(f"Enhanced features prepared - Train: {X_train_enhanced.shape}, Test: {X_test_enhanced.shape}")
        
        X_train, X_test = X_train_enhanced, X_test_enhanced
    else:
        X_train, X_test = X_train_basic, X_test_basic
    
    # Apply feature selection if specified
    if args.feature_selection_method != "none" and args.feature_engineering_method == "all":
        logger.info(f"Applying {args.feature_selection_method} feature selection...")
        
        if args.feature_selection_method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=min(args.num_features_to_select, X_train.shape[1]))
        elif args.feature_selection_method == "rfe":
            estimator = RandomForestClassifier(n_estimators=50, random_state=args.random_state)
            selector = RFE(estimator, n_features_to_select=min(args.num_features_to_select, X_train.shape[1]), step=0.1)
        elif args.feature_selection_method == "select_from_model":
            estimator = RandomForestClassifier(n_estimators=100, random_state=args.random_state)
            selector = SelectFromModel(estimator, max_features=min(args.num_features_to_select, X_train.shape[1]))
        else:
            selector = None
        
        if selector is not None:
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            logger.info(f"After feature selection - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Scale features
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple advanced classical models
    logger.info("Training advanced classical models...")
    
    models = {
        'RandomForest-Advanced': RandomForestClassifier(
            n_estimators=300,  # Increased from default
            max_depth=15,      # Increased depth for more complexity
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=args.random_state,
            n_jobs=-1
        ),
        'ExtraTrees-Advanced': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=args.random_state,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,  # Lower learning rate for better generalization
            max_depth=5,
            subsample=0.8,
            random_state=args.random_state
        ),
        'XGBoost-like': ExtraTreesClassifier(  # Using ExtraTrees as a proxy for XGBoost
            n_estimators=500,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            oob_score=True,
            random_state=args.random_state,
            n_jobs=-1
        ),
        'NeuralNetwork': MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Deep architecture
            activation='relu',
            solver='adam',
            alpha=0.01,  # L2 regularization
            batch_size='auto',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=args.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # For computationally expensive models, use a subset if needed
        if name in ['NeuralNetwork'] and X_train_scaled.shape[0] > 1000:
            subset_indices = np.random.choice(X_train_scaled.shape[0], size=min(1000, X_train_scaled.shape[0]), replace=False)
            X_subset = X_train_scaled[subset_indices]
            y_subset = y_train[subset_indices]
            model.fit(X_subset, y_subset)
        else:
            model.fit(X_train_scaled, y_train)
        
        # Evaluate
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            # For models without predict_proba, use decision function
            y_pred_proba = model.decision_function(X_test_scaled)
            # Convert to probabilities using sigmoid
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
        
        pr_auc = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'model': model
        }
        
        logger.info(f"  ✅ {name} - PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    # Find best classical model
    best_classical_model_name = max(results, key=lambda k: results[k]['pr_auc'])
    best_classical_pr_auc = results[best_classical_model_name]['pr_auc']
    
    logger.info(f"\nBest Classical Model: {best_classical_model_name} - PR-AUC: {best_classical_pr_auc:.4f}")
    
    # Compare with our previous quantum result
    logger.info(f"Previous best quantum result: 0.6663")
    logger.info(f"Improvement over quantum: {best_classical_pr_auc - 0.6663:.4f}")
    
    # Save results
    results_df = pd.DataFrame({
        'model': list(results.keys()),
        'pr_auc': [results[k]['pr_auc'] for k in results.keys()],
        'roc_auc': [results[k]['roc_auc'] for k in results.keys()],
        'relation': args.relation,
        'feature_engineering_method': args.feature_engineering_method,
        'feature_selection_method': args.feature_selection_method,
        'num_features_selected': X_train.shape[1]
    })
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    results_file = f"results/advanced_classical_features_results_{args.relation}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\nAdvanced classical feature engineering completed!")


if __name__ == "__main__":
    main()