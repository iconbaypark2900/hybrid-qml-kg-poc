#!/usr/bin/env python3
"""
Implement Quantum-Classical Feature Fusion

This script implements quantum-classical feature fusion to combine quantum and classical
representations for improved link prediction performance.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
import warnings

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder
from quantum_layer.qml_model import QMLLinkPredictor
from quantum_layer.advanced_qml_features import QuantumFeatureEngineer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_quantum_classical_features(X_quantum, X_classical, y_train, top_k=20):
    """
    Create fused features combining quantum and classical representations.
    
    Args:
        X_quantum: Quantum features [N, D_q]
        X_classical: Classical features [N, D_c]
        y_train: Training labels
        top_k: Number of top features to select from each modality
    
    Returns:
        Fused features [N, D_fused]
    """
    logger.info(f"Creating quantum-classical feature fusion (top_k={top_k})...")
    
    # Feature selection for quantum features
    if X_quantum.shape[1] > top_k:
        quantum_selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
        X_quantum_selected = quantum_selector.fit_transform(X_quantum, y_train)
    else:
        X_quantum_selected = X_quantum
    
    # Feature selection for classical features
    if X_classical.shape[1] > top_k:
        classical_selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
        X_classical_selected = classical_selector.fit_transform(X_classical, y_train)
    else:
        X_classical_selected = X_classical
    
    # Create interaction features between quantum and classical
    n_samples = X_quantum_selected.shape[0]
    n_quantum = X_quantum_selected.shape[1]
    n_classical = X_classical_selected.shape[1]
    
    # Element-wise multiplication features
    interaction_mult = X_quantum_selected[:, :min(n_quantum, n_classical)] * X_classical_selected[:, :min(n_quantum, n_classical)]
    
    # Element-wise addition features
    interaction_add = X_quantum_selected[:, :min(n_quantum, n_classical)] + X_classical_selected[:, :min(n_quantum, n_classical)]
    
    # Concatenate all features
    X_fused = np.hstack([
        X_quantum_selected,
        X_classical_selected,
        interaction_mult,
        interaction_add
    ])
    
    logger.info(f"Feature fusion completed: {X_quantum.shape[1]} + {X_classical.shape[1]} → {X_fused.shape[1]} features")
    
    return X_fused


def main():
    parser = argparse.ArgumentParser(description="Implement Quantum-Classical Feature Fusion")
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type (e.g., CtD for Compound treats Disease)")
    parser.add_argument("--max_entities", type=int, default=100, help="Max entities to include (for scalability)")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--qml_dim", type=int, default=16, help="Quantum feature dimension (qubits)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--top_k_features", type=int, default=20, help="Top k features to select from each modality")
    parser.add_argument("--fusion_method", type=str, default="concat_interaction", 
                        choices=["concat", "interaction", "concat_interaction", "weighted_sum"],
                        help="Method for feature fusion")

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
    
    # Build enhanced features for classical models
    feature_builder = EnhancedFeatureBuilder(
        include_graph_features=True,
        include_domain_features=True,
        normalize=True
    )
    
    # Build graph for graph features (TRAIN ONLY to prevent leakage)
    train_edges_only = train_df[train_df['label'] == 1].copy()
    feature_builder.build_graph(train_edges_only)
    
    # Build classical features
    X_train_classical = feature_builder.build_features(train_df, embedder.get_all_embeddings(), edges_df=train_edges_only)
    X_test_classical = feature_builder.build_features(test_df, embedder.get_all_embeddings(), edges_df=train_edges_only)
    
    # Prepare quantum features using the quantum feature engineer
    logger.info("Preparing quantum features...")
    try:
        from quantum_layer.advanced_qml_features import QuantumFeatureEngineer
        quantum_feature_eng = QuantumFeatureEngineer(
            num_qubits=args.qml_dim,
            encoding_strategy="hybrid",
            feature_map_type="Pauli",
            feature_map_reps=3,
            entanglement="full",
            random_state=args.random_state
        )
        
        # Prepare quantum features
        X_train_quantum = quantum_feature_eng.prepare_quantum_features(
            train_df, embedder.get_all_embeddings(), edges_df=train_edges_only
        )
        X_test_quantum = quantum_feature_eng.prepare_quantum_features(
            test_df, embedder.get_all_embeddings(), edges_df=train_edges_only
        )
    except ImportError:
        logger.warning("QuantumFeatureEngineer not available, using basic approach")
        # Fallback: use PCA to reduce classical features to quantum dimension
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.qml_dim)
        X_train_quantum = pca.fit_transform(X_train_classical)
        X_test_quantum = pca.transform(X_test_classical)
    
    logger.info(f"Quantum features prepared: train {X_train_quantum.shape}, test {X_test_quantum.shape}")
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Features prepared - Quantum: train {X_train_quantum.shape}, test {X_test_quantum.shape}")
    logger.info(f"Features prepared - Classical: train {X_train_classical.shape}, test {X_test_classical.shape}")
    
    # Create fused features
    logger.info("Creating quantum-classical fused features...")
    X_train_fused = create_quantum_classical_features(X_train_quantum, X_train_classical, y_train, args.top_k_features)
    X_test_fused = create_quantum_classical_features(X_test_quantum, X_test_classical, y_train, args.top_k_features)  # Use y_train for consistency in feature selection
    
    logger.info(f"Fused features - Train: {X_train_fused.shape}, Test: {X_test_fused.shape}")
    
    # Scale fused features
    scaler = StandardScaler()
    X_train_fused_scaled = scaler.fit_transform(X_train_fused)
    X_test_fused_scaled = scaler.transform(X_test_fused)
    
    # Train models on different feature sets for comparison
    logger.info("Training models on different feature sets for comparison...")
    
    # Model configurations
    rf_config = {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'class_weight': 'balanced',
        'random_state': args.random_state,
        'n_jobs': -1
    }
    
    # 1. Quantum-only model
    logger.info("Training quantum-only model...")
    quantum_model = QMLLinkPredictor(
        model_type="QSVC",
        encoding_method="feature_map",
        num_qubits=args.qml_dim,
        feature_map_type="Pauli",
        feature_map_reps=3,
        random_state=args.random_state
    )
    
    quantum_model.fit(X_train_quantum, y_train)
    quantum_pred_proba = quantum_model.predict_proba(X_test_quantum)[:, 1]
    quantum_ap = average_precision_score(y_test, quantum_pred_proba)
    logger.info(f"  Quantum-only PR-AUC: {quantum_ap:.4f}")
    
    # 2. Classical-only model
    logger.info("Training classical-only model...")
    classical_model = RandomForestClassifier(**rf_config)
    classical_model.fit(X_train_classical, y_train)
    classical_pred_proba = classical_model.predict_proba(X_test_classical)[:, 1]
    classical_ap = average_precision_score(y_test, classical_pred_proba)
    logger.info(f"  Classical-only PR-AUC: {classical_ap:.4f}")
    
    # 3. Fused features model
    logger.info("Training model on fused features...")
    fused_model = RandomForestClassifier(**rf_config)
    fused_model.fit(X_train_fused_scaled, y_train)
    fused_pred_proba = fused_model.predict_proba(X_test_fused_scaled)[:, 1]
    fused_ap = average_precision_score(y_test, fused_pred_proba)
    logger.info(f"  Fused features PR-AUC: {fused_ap:.4f}")
    
    # 4. Ensemble of quantum and classical
    logger.info("Training quantum-classical ensemble...")
    
    # Since we can't directly ensemble the quantum and classical models due to different interfaces,
    # we'll create a meta-learner that combines their predictions
    meta_features_train = np.column_stack([quantum_model.predict_proba(X_train_quantum)[:, 1], 
                                          classical_model.predict_proba(X_train_classical_scaled)[:, 1]])
    meta_features_test = np.column_stack([quantum_pred_proba, 
                                         classical_pred_proba])
    
    # Meta-learner
    meta_learner = LogisticRegression(random_state=args.random_state)
    meta_learner.fit(meta_features_train, y_train)
    ensemble_pred_proba = meta_learner.predict_proba(meta_features_test)[:, 1]
    ensemble_ap = average_precision_score(y_test, ensemble_pred_proba)
    logger.info(f"  Quantum-Classical Ensemble PR-AUC: {ensemble_ap:.4f}")
    
    # Compare all approaches
    logger.info("\n" + "="*80)
    logger.info("QUANTUM-CLASSICAL FEATURE FUSION RESULTS")
    logger.info("="*80)
    logger.info(f"Quantum-only model          : PR-AUC = {quantum_ap:.4f}")
    logger.info(f"Classical-only model        : PR-AUC = {classical_ap:.4f}")
    logger.info(f"Fused features model        : PR-AUC = {fused_ap:.4f}")
    logger.info(f"Quantum-Classical Ensemble  : PR-AUC = {ensemble_ap:.4f}")
    
    best_approach = max([
        ("Quantum-only", quantum_ap),
        ("Classical-only", classical_ap),
        ("Fused features", fused_ap),
        ("Quantum-Classical Ensemble", ensemble_ap)
    ], key=lambda x: x[1])
    
    logger.info(f"\n🏆 Best Approach: {best_approach[0]} - PR-AUC: {best_approach[1]:.4f}")
    
    # Calculate improvement over best of individual models
    best_individual = max(quantum_ap, classical_ap)
    improvement = best_approach[1] - best_individual
    logger.info(f"Improvement over best individual: {improvement:+.4f}")
    
    # Save results
    results = {
        'relation': args.relation,
        'best_approach': best_approach[0],
        'best_pr_auc': best_approach[1],
        'quantum_pr_auc': quantum_ap,
        'classical_pr_auc': classical_ap,
        'fused_pr_auc': fused_ap,
        'ensemble_pr_auc': ensemble_ap,
        'improvement_over_best_individual': improvement,
        'test_size': args.test_size,
        'embedding_dim': args.embedding_dim,
        'qml_dim': args.qml_dim,
        'top_k_features': args.top_k_features,
        'fusion_method': args.fusion_method
    }
    
    results_df = pd.DataFrame([results])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    results_file = f"results/quantum_classical_fusion_results_{args.relation}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\nQuantum-classical feature fusion implementation completed!")


if __name__ == "__main__":
    main()