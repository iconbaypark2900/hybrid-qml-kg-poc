#!/usr/bin/env python3
"""
Implement Quantum-Classical Ensemble for Knowledge Graph Link Prediction

This script implements the quantum-classical ensemble approach to combine
predictions from quantum and classical models for improved link prediction performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
from quantum_layer.quantum_classical_ensemble import QuantumClassicalEnsemble
from quantum_layer.qml_model import QMLLinkPredictor
from classical_baseline.train_baseline import ClassicalLinkPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Implement Quantum-Classical Ensemble for Link Prediction")
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type (e.g., CtD for Compound treats Disease)")
    parser.add_argument("--max_entities", type=int, default=100, help="Max entities to include (for scalability)")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--qml_dim", type=int, default=16, help="Quantum feature dimension (qubits)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    
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
    embedder = HetionetEmbedder(
        embedding_dim=args.embedding_dim,
        qml_dim=args.qml_dim,  # This will be used for quantum feature reduction
        work_dir="data"
    )
    
    # Load or train embeddings
    if embedder.load_saved_embeddings():
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
    
    # Initialize quantum and classical models with optimized configurations
    logger.info("Initializing optimized quantum and classical models...")
    
    # Quantum model with optimized parameters from our experiments
    quantum_model = QMLLinkPredictor(
        model_type="QSVC",
        encoding_method="feature_map",
        num_qubits=args.qml_dim,
        feature_map_type="ZZ",  # Only ZZ and Z are supported by the current implementation
        feature_map_reps=3,     # But we can still use 3 repetitions as found to be effective
        random_state=args.random_state
    )
    
    # Classical model (Random Forest performed well in our experiments)
    classical_model = ClassicalLinkPredictor(
        model_type="RandomForest",
        random_state=args.random_state
    )
    
    # Train individual models first to compare with ensemble
    logger.info("Training individual models...")
    
    # Train quantum model
    logger.info("Training quantum model...")
    quantum_model.fit(X_train_qml, y_train)
    quantum_pred_proba = quantum_model.predict_proba(X_test_qml)[:, 1]
    quantum_ap = average_precision_score(y_test, quantum_pred_proba)
    logger.info(f"Quantum model PR-AUC: {quantum_ap:.4f}")
    
    # For the classical model, we need to use a standard sklearn model since the ClassicalLinkPredictor
    # expects dataframes and embedders, not pre-computed features
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    logger.info("Training classical model (Random Forest)...")
    scaler = StandardScaler()
    X_train_classical_scaled = scaler.fit_transform(X_train_classical)
    X_test_classical_scaled = scaler.transform(X_test_classical)
    
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=args.random_state)
    rf_model.fit(X_train_classical_scaled, y_train)
    classical_pred_proba = rf_model.predict_proba(X_test_classical_scaled)[:, 1]
    classical_ap = average_precision_score(y_test, classical_pred_proba)
    logger.info(f"Classical model PR-AUC: {classical_ap:.4f}")
    
    # Create and evaluate the quantum-classical ensemble
    logger.info("Creating quantum-classical ensemble...")
    ensemble_pred_proba = 0.6 * quantum_pred_proba + 0.4 * classical_pred_proba
    ensemble_ap = average_precision_score(y_test, ensemble_pred_proba)
    logger.info(f"Quantum-Classical Ensemble PR-AUC: {ensemble_ap:.4f}")
    
    ensemble_ap = average_precision_score(y_test, ensemble_pred_proba)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
    
    logger.info(f"Quantum-Classic Ensemble PR-AUC: {ensemble_ap:.4f}")
    logger.info(f"Quantum-Classic Ensemble ROC-AUC: {ensemble_auc:.4f}")
    
    # Compare all approaches
    logger.info("\n" + "="*60)
    logger.info("COMPARISON OF APPROACHES")
    logger.info("="*60)
    logger.info(f"Classical Model (Random Forest)  : PR-AUC = {classical_ap:.4f}")
    logger.info(f"Quantum Model (QSVC-Pauli)      : PR-AUC = {quantum_ap:.4f}")
    logger.info(f"Quantum-Classic Ensemble        : PR-AUC = {ensemble_ap:.4f}")
    
    if ensemble_ap > max(classical_ap, quantum_ap):
        logger.info("✅ Ensemble outperforms individual models!")
    else:
        logger.info("ℹ️  Ensemble performance is comparable to best individual model")
    
    # Calculate improvement
    best_individual = max(classical_ap, quantum_ap)
    improvement = ensemble_ap - best_individual
    logger.info(f"Improvement over best individual: {improvement:+.4f}")
    
    # Save results
    results = {
        'relation': args.relation,
        'classical_pr_auc': classical_ap,
        'quantum_pr_auc': quantum_ap,
        'ensemble_pr_auc': ensemble_ap,
        'improvement_over_best': improvement,
        'test_size': args.test_size,
        'embedding_dim': args.embedding_dim,
        'qml_dim': args.qml_dim
    }
    
    results_df = pd.DataFrame([results])
    results_file = f"results/quantum_classical_ensemble_results_{args.relation}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\nQuantum-Classical Ensemble implementation completed!")


if __name__ == "__main__":
    main()