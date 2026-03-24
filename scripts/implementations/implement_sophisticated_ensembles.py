#!/usr/bin/env python3
"""
Implement Sophisticated Ensemble Methods

This script implements advanced ensemble techniques that combine quantum and classical models
to achieve superior performance in link prediction tasks.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import warnings

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder
from quantum_layer.qml_model import QMLLinkPredictor
from quantum_layer.quantum_classical_ensemble import QuantumClassicalEnsemble

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sophisticated_ensembles(X_train, y_train, X_test, y_test, X_train_qml, X_test_qml):
    """Create sophisticated ensemble methods combining quantum and classical models."""
    logger.info("Creating sophisticated ensemble methods...")
    
    # Scale classical features
    scaler = StandardScaler()
    X_train_classical_scaled = scaler.fit_transform(X_train)
    X_test_classical_scaled = scaler.transform(X_test)
    
    # Train individual models that will be used in ensembles
    logger.info("Training individual models for ensemble...")
    
    # Quantum model (QSVC with optimized parameters)
    quantum_model = QMLLinkPredictor(
        model_type="QSVC",
        encoding_method="feature_map",
        num_qubits=16,  # Using 16 qubits as optimized
        feature_map_type="Pauli",  # Using Pauli as found to be effective
        feature_map_reps=3,        # Using 3 reps as found to be effective
        random_state=42
    )
    quantum_model.fit(X_train_qml, y_train)
    quantum_pred_proba = quantum_model.predict_proba(X_test_qml)[:, 1]
    quantum_ap = average_precision_score(y_test, quantum_pred_proba)
    
    # Classical models
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_classical_scaled, y_train)
    rf_pred_proba = rf_model.predict_proba(X_test_classical_scaled)[:, 1]
    rf_ap = average_precision_score(y_test, rf_pred_proba)
    
    et_model = ExtraTreesClassifier(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
    et_model.fit(X_train_classical_scaled, y_train)
    et_pred_proba = et_model.predict_proba(X_test_classical_scaled)[:, 1]
    et_ap = average_precision_score(y_test, et_pred_proba)
    
    gb_model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
    gb_model.fit(X_train_classical_scaled, y_train)
    gb_pred_proba = gb_model.predict_proba(X_test_classical_scaled)[:, 1]
    gb_ap = average_precision_score(y_test, gb_pred_proba)
    
    logger.info(f"Individual model performances:")
    logger.info(f"  Quantum (QSVC): {quantum_ap:.4f}")
    logger.info(f"  Random Forest: {rf_ap:.4f}")
    logger.info(f"  Extra Trees: {et_ap:.4f}")
    logger.info(f"  Gradient Boosting: {gb_ap:.4f}")
    
    # 1. Voting Ensemble (Simple combination)
    logger.info("Creating voting ensemble...")
    voting_ensemble = VotingClassifier(
        estimators=[
            ('quantum', quantum_model),
            ('rf', rf_model),
            ('et', et_model),
            ('gb', gb_model)
        ],
        voting='soft',  # Use probability averaging
        weights=[1.5, 1.0, 1.0, 0.8]  # Give quantum model higher weight based on performance
    )
    
    # Since QSVC doesn't work with VotingClassifier directly, we'll implement our own weighted combination
    # Calculate weighted average of probabilities
    weights = np.array([1.5, 1.0, 1.0, 0.8])  # Quantum gets higher weight
    weights = weights / weights.sum()  # Normalize
    
    # Combine probabilities
    ensemble_proba = (
        weights[0] * quantum_pred_proba +
        weights[1] * rf_pred_proba +
        weights[2] * et_pred_proba +
        weights[3] * gb_pred_proba
    )
    
    voting_ap = average_precision_score(y_test, ensemble_proba)
    logger.info(f"  Voting ensemble PR-AUC: {voting_ap:.4f}")
    
    # 2. Stacking Ensemble (Meta-learner approach)
    logger.info("Creating stacking ensemble...")
    
    # Prepare training data for meta-learner using cross-validation predictions
    from sklearn.model_selection import cross_val_predict
    
    # Get out-of-fold predictions for training the meta-learner
    quantum_oof = cross_val_predict(quantum_model, X_train_qml, y_train, cv=3, method='predict_proba')[:, 1]
    rf_oof = cross_val_predict(rf_model, X_train_classical_scaled, y_train, cv=3, method='predict_proba')[:, 1]
    et_oof = cross_val_predict(et_model, X_train_classical_scaled, y_train, cv=3, method='predict_proba')[:, 1]
    gb_oof = cross_val_predict(gb_model, X_train_classical_scaled, y_train, cv=3, method='predict_proba')[:, 1]
    
    # Create meta-features (predictions from base models)
    meta_X_train = np.column_stack([quantum_oof, rf_oof, et_oof, gb_oof])
    meta_X_test = np.column_stack([quantum_pred_proba, rf_pred_proba, et_pred_proba, gb_pred_proba])
    
    # Train meta-learner
    meta_learner = LogisticRegression(random_state=42)
    meta_learner.fit(meta_X_train, y_train)
    
    # Predict with meta-learner
    stacking_pred_proba = meta_learner.predict_proba(meta_X_test)[:, 1]
    stacking_ap = average_precision_score(y_test, stacking_pred_proba)
    logger.info(f"  Stacking ensemble PR-AUC: {stacking_ap:.4f}")
    
    # 3. Quantum-Classical Ensemble (already implemented)
    logger.info("Creating quantum-classical ensemble...")
    qc_ensemble = QuantumClassicalEnsemble(
        quantum_model=quantum_model,
        classical_model=rf_model,  # Using the best classical model
        ensemble_method="weighted_average",
        weights={"quantum": 0.6, "classical": 0.4},  # Adjust weights based on individual performance
        random_state=42
    )
    
    # For the QC ensemble, we'll use the individual model predictions
    qc_ensemble_proba = 0.6 * quantum_pred_proba + 0.4 * rf_pred_proba
    qc_ensemble_ap = average_precision_score(y_test, qc_ensemble_proba)
    logger.info(f"  Quantum-Classical ensemble PR-AUC: {qc_ensemble_ap:.4f}")
    
    # 4. Adaptive Ensemble (based on confidence)
    logger.info("Creating adaptive ensemble...")
    
    # Calculate uncertainty for each model (1 - max probability)
    quantum_uncertainty = 1 - np.abs(quantum_pred_proba - 0.5) * 2  # Closer to 0.5 = higher uncertainty
    rf_uncertainty = 1 - np.abs(rf_pred_proba - 0.5) * 2
    et_uncertainty = 1 - np.abs(et_pred_proba - 0.5) * 2
    gb_uncertainty = 1 - np.abs(gb_pred_proba - 0.5) * 2
    
    # For each sample, use the prediction from the most confident model
    uncertainties = np.column_stack([quantum_uncertainty, rf_uncertainty, et_uncertainty, gb_uncertainty])
    predictions = np.column_stack([quantum_pred_proba, rf_pred_proba, et_pred_proba, gb_pred_proba])
    
    # Select the prediction from the model with lowest uncertainty (highest confidence) for each sample
    best_model_idx = np.argmin(uncertainties, axis=1)
    adaptive_ensemble_proba = predictions[np.arange(len(predictions)), best_model_idx]
    adaptive_ap = average_precision_score(y_test, adaptive_ensemble_proba)
    logger.info(f"  Adaptive ensemble PR-AUC: {adaptive_ap:.4f}")
    
    # Compile results
    ensemble_results = {
        'voting': {'ap': voting_ap, 'proba': ensemble_proba},
        'stacking': {'ap': stacking_ap, 'proba': stacking_pred_proba},
        'quantum_classical': {'ap': qc_ensemble_ap, 'proba': qc_ensemble_proba},
        'adaptive': {'ap': adaptive_ap, 'proba': adaptive_ensemble_proba},
        'individual_performances': {
            'quantum': quantum_ap,
            'random_forest': rf_ap,
            'extra_trees': et_ap,
            'gradient_boosting': gb_ap
        }
    }
    
    return ensemble_results


def main():
    parser = argparse.ArgumentParser(description="Implement Sophisticated Ensemble Methods")
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
    
    # Build features for classical models
    feature_builder = EnhancedFeatureBuilder(
        include_graph_features=True,
        include_domain_features=True,
        normalize=True
    )
    
    # Build graph for graph features (TRAIN ONLY to prevent leakage)
    train_edges_only = train_df[train_df['label'] == 1].copy()
    feature_builder.build_graph(train_edges_only)
    
    X_train_classical = feature_builder.build_features(train_df, embedder.get_all_embeddings(), edges_df=train_edges_only)
    X_test_classical = feature_builder.build_features(test_df, embedder.get_all_embeddings(), edges_df=train_edges_only)
    
    # Prepare quantum features
    X_train_qml = embedder.prepare_link_features_qml(train_df, mode="diff")
    X_test_qml = embedder.prepare_link_features_qml(test_df, mode="diff")
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Classical features - Train: {X_train_classical.shape}, Test: {X_test_classical.shape}")
    logger.info(f"Quantum features - Train: {X_train_qml.shape}, Test: {X_test_qml.shape}")
    
    # Create sophisticated ensembles
    ensemble_results = create_sophisticated_ensembles(
        X_train_classical, y_train, X_test_classical, y_test,
        X_train_qml, X_test_qml
    )
    
    # Find best ensemble
    best_ensemble_name = max(ensemble_results.keys(), key=lambda k: ensemble_results[k]['ap'] if k != 'individual_performances' else 0)
    best_ensemble_ap = ensemble_results[best_ensemble_name]['ap']
    
    logger.info("\n" + "="*80)
    logger.info("ENSEMBLE METHODS COMPARISON")
    logger.info("="*80)
    
    # Display individual performances
    logger.info("Individual Model Performances:")
    for model_name, ap_score in ensemble_results['individual_performances'].items():
        logger.info(f"  {model_name:20s}: {ap_score:.4f}")
    
    logger.info("\nEnsemble Method Performances:")
    for ensemble_name, result in ensemble_results.items():
        if ensemble_name != 'individual_performances':
            logger.info(f"  {ensemble_name:20s}: {result['ap']:.4f}")
    
    logger.info(f"\n🏆 Best Ensemble: {best_ensemble_name} - PR-AUC: {best_ensemble_ap:.4f}")
    
    # Compare with best individual model
    best_individual_ap = max(ensemble_results['individual_performances'].values())
    improvement = best_ensemble_ap - best_individual_ap
    logger.info(f"  → Ensemble improvement over best individual: {improvement:+.4f}")
    
    # Save results
    results = {
        'relation': args.relation,
        'best_ensemble': best_ensemble_name,
        'best_ensemble_pr_auc': best_ensemble_ap,
        'best_individual_pr_auc': best_individual_ap,
        'ensemble_improvement': improvement,
        'test_size': args.test_size,
        'embedding_dim': args.embedding_dim,
        'qml_dim': args.qml_dim,
        'individual_performances': ensemble_results['individual_performances'],
        'ensemble_performances': {k: v['ap'] for k, v in ensemble_results.items() if k != 'individual_performances'}
    }
    
    results_df = pd.DataFrame([results])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    results_file = f"results/ensemble_methods_results_{args.relation}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\nSophisticated ensemble methods implementation completed!")


if __name__ == "__main__":
    main()