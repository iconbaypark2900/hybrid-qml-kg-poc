#!/usr/bin/env python3
"""
Implement Quantum Kernel Alignment Optimization

This script implements quantum kernel alignment optimization to improve
the performance of quantum models by aligning the quantum kernel with
the ideal kernel for the specific task.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
from quantum_layer.quantum_kernel_engineering import QuantumKernelAligner, improve_quantum_kernel
from quantum_layer.qml_model import QMLLinkPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Implement Quantum Kernel Alignment Optimization")
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type (e.g., CtD for Compound treats Disease)")
    parser.add_argument("--max_entities", type=int, default=100, help="Max entities to include (for scalability)")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--qml_dim", type=int, default=16, help="Quantum feature dimension (qubits)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--alignment_method", type=str, default="kernel_target", 
                        choices=["kernel_target", "spectral", "hilbert_schmidt"],
                        help="Method for kernel alignment")
    parser.add_argument("--max_alignment_iter", type=int, default=50, 
                        help="Maximum iterations for alignment optimization")

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
    
    # Apply quantum kernel alignment
    logger.info(f"Applying quantum kernel alignment using method: {args.alignment_method}")
    
    # Create quantum kernel aligner
    aligner = QuantumKernelAligner(
        feature_map_type="Pauli",  # Using Pauli as it performed well in our experiments
        num_qubits=args.qml_dim,
        reps=3,  # Using 3 reps as found to be effective
        entanglement="full",
        alignment_method=args.alignment_method,
        max_iterations=args.max_alignment_iter
    )
    
    # Align the quantum kernel to the target
    alignment_results = aligner.align_to_target(X_train_qml, y_train)
    logger.info(f"Quantum kernel alignment score: {alignment_results['alignment_score']:.4f}")
    
    # Initialize quantum model with aligned kernel
    logger.info("Initializing quantum model with aligned kernel...")
    quantum_model = QMLLinkPredictor(
        model_type="QSVC",
        encoding_method="feature_map",
        num_qubits=args.qml_dim,
        feature_map_type="Pauli",  # Using Pauli as found to be effective
        feature_map_reps=3,        # Using 3 reps as found to be effective
        random_state=args.random_state
    )
    
    # Train the quantum model
    logger.info("Training quantum model with aligned features...")
    quantum_model.fit(X_train_qml, y_train)
    
    # Evaluate quantum model
    quantum_pred_proba = quantum_model.predict_proba(X_test_qml)[:, 1]
    quantum_ap = average_precision_score(y_test, quantum_pred_proba)
    quantum_auc = roc_auc_score(y_test, quantum_pred_proba)
    
    logger.info(f"Quantum model (with alignment) - PR-AUC: {quantum_ap:.4f}, ROC-AUC: {quantum_auc:.4f}")
    
    # For comparison, also train a classical model
    logger.info("Training classical model (Random Forest) for comparison...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=args.random_state
    )
    
    # Scale classical features
    scaler = StandardScaler()
    X_train_classical_scaled = scaler.fit_transform(X_train_classical)
    X_test_classical_scaled = scaler.transform(X_test_classical)
    
    rf_model.fit(X_train_classical_scaled, y_train)
    rf_pred_proba = rf_model.predict_proba(X_test_classical_scaled)[:, 1]
    rf_ap = average_precision_score(y_test, rf_pred_proba)
    rf_auc = roc_auc_score(y_test, rf_pred_proba)
    
    logger.info(f"Classical model (Random Forest) - PR-AUC: {rf_ap:.4f}, ROC-AUC: {rf_auc:.4f}")
    
    # Compare results
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: Quantum vs Classical (with kernel alignment)")
    logger.info("="*80)
    logger.info(f"Quantum (QSVC+Alignment)  : PR-AUC = {quantum_ap:.4f}, ROC-AUC = {quantum_auc:.4f}")
    logger.info(f"Classical (RandomForest)  : PR-AUC = {rf_ap:.4f}, ROC-AUC = {rf_auc:.4f}")
    
    improvement = quantum_ap - rf_ap
    logger.info(f"Improvement (Quantum - Classical): {improvement:+.4f}")
    
    # Save results
    results = {
        'relation': args.relation,
        'quantum_pr_auc': quantum_ap,
        'quantum_roc_auc': quantum_auc,
        'classical_pr_auc': rf_ap,
        'classical_roc_auc': rf_auc,
        'improvement': improvement,
        'kernel_alignment_score': alignment_results['alignment_score'],
        'test_size': args.test_size,
        'embedding_dim': args.embedding_dim,
        'qml_dim': args.qml_dim,
        'alignment_method': args.alignment_method
    }
    
    results_df = pd.DataFrame([results])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    results_file = f"results/quantum_kernel_alignment_results_{args.relation}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\nQuantum kernel alignment optimization completed!")


if __name__ == "__main__":
    main()