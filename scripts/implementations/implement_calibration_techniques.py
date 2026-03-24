#!/usr/bin/env python3
"""
Implement Calibration Techniques for Quantum and Classical Models

This script applies calibration techniques to improve the probability estimates
of both quantum and classical models, which can significantly improve PR-AUC scores.
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
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder
from quantum_layer.qml_model import QMLLinkPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_reliability_diagram(y_true, y_prob, model_name, n_bins=10):
    """Plot reliability diagram (calibration curve) for a model."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{model_name}")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Reliability Diagram for {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.savefig(f"results/reliability_diagram_{model_name.replace(' ', '_').replace('/', '_').lower()}.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Apply Calibration Techniques to Improve Model Performance")
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
    
    # Prepare quantum features
    try:
        X_train_qml = embedder.prepare_link_features_qml(train_df, mode="diff")
        X_test_qml = embedder.prepare_link_features_qml(test_df, mode="diff")
    except AttributeError:
        # If prepare_link_features_qml doesn't exist, use regular features and reduce dimension
        X_train_temp = embedder.prepare_link_features(train_df)
        X_test_temp = embedder.prepare_link_features(test_df)
        
        # Use PCA to reduce to quantum dimension
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.qml_dim)
        X_train_qml = pca.fit_transform(X_train_temp)
        X_test_qml = pca.transform(X_test_temp)
    
    X_train_classical = embedder.prepare_link_features(train_df)
    X_test_classical = embedder.prepare_link_features(test_df)
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Features prepared - Quantum: train {X_train_qml.shape}, test {X_test_qml.shape}")
    logger.info(f"Features prepared - Classical: train {X_train_classical.shape}, test {X_test_classical.shape}")
    
    # Scale classical features
    scaler = StandardScaler()
    X_train_classical_scaled = scaler.fit_transform(X_train_classical)
    X_test_classical_scaled = scaler.transform(X_test_classical)
    
    # Train quantum model (QSVC)
    logger.info("Training quantum model (QSVC)...")
    quantum_model = QMLLinkPredictor(
        model_type="QSVC",
        encoding_method="feature_map",
        num_qubits=args.qml_dim,
        feature_map_type="Pauli",  # Using Pauli as found to be effective
        feature_map_reps=3,        # Using 3 reps as found to be effective
        random_state=args.random_state
    )
    
    quantum_model.fit(X_train_qml, y_train)
    
    # Get uncalibrated quantum predictions
    quantum_pred_proba = quantum_model.predict_proba(X_test_qml)[:, 1]
    quantum_ap_uncal = average_precision_score(y_test, quantum_pred_proba)
    logger.info(f"Uncalibrated Quantum PR-AUC: {quantum_ap_uncal:.4f}")
    
    # Train classical model (Random Forest)
    logger.info("Training classical model (Random Forest)...")
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=args.random_state,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_classical_scaled, y_train)
    
    # Get uncalibrated classical predictions
    rf_pred_proba = rf_model.predict_proba(X_test_classical_scaled)[:, 1]
    rf_ap_uncal = average_precision_score(y_test, rf_pred_proba)
    logger.info(f"Uncalibrated Classical (RF) PR-AUC: {rf_ap_uncal:.4f}")
    
    # Apply calibration to quantum model
    logger.info("Applying calibration to quantum model...")
    quantum_cal_results = {}
    for cal_method in ['sigmoid', 'isotonic']:
        try:
            # Use CalibratedClassifierCV for quantum model
            calibrator = CalibratedClassifierCV(quantum_model, method=cal_method, cv=3)
            
            # Fit the calibrator on training data
            calibrator.fit(X_train_qml, y_train)
            
            # Get calibrated probabilities
            quantum_cal_proba = calibrator.predict_proba(X_test_qml)[:, 1]
            
            # Calculate metrics
            pr_auc_cal = average_precision_score(y_test, quantum_cal_proba)
            roc_auc_cal = roc_auc_score(y_test, quantum_cal_proba)
            brier_cal = brier_score_loss(y_test, quantum_cal_proba)
            
            cal_metrics = {
                'pr_auc': pr_auc_cal,
                'roc_auc': roc_auc_cal,
                'brier_score': brier_cal,
                'method': cal_method
            }
            
            # Plot reliability diagram
            plot_reliability_diagram(y_test, quantum_cal_proba, f"QSVC_{cal_method}")
            
            quantum_cal_results[cal_method] = {
                'proba': quantum_cal_proba,
                'metrics': cal_metrics
            }
            
            # Compare with uncalibrated
            improvement = pr_auc_cal - quantum_ap_uncal
            logger.info(f"  {cal_method} calibration improvement: {improvement:+.4f}")
            
        except Exception as e:
            logger.warning(f"  Quantum calibration with {cal_method} failed: {e}")
    
    # Apply calibration to classical model
    logger.info("Applying calibration to classical model...")
    rf_cal_results = {}
    for cal_method in ['sigmoid', 'isotonic']:
        try:
            # Use CalibratedClassifierCV for classical model
            calibrator = CalibratedClassifierCV(rf_model, method=cal_method, cv=3)
            
            # Fit the calibrator on training data
            calibrator.fit(X_train_classical_scaled, y_train)
            
            # Get calibrated probabilities
            rf_cal_proba = calibrator.predict_proba(X_test_classical_scaled)[:, 1]
            
            # Calculate metrics
            pr_auc_cal = average_precision_score(y_test, rf_cal_proba)
            roc_auc_cal = roc_auc_score(y_test, rf_cal_proba)
            brier_cal = brier_score_loss(y_test, rf_cal_proba)
            
            cal_metrics = {
                'pr_auc': pr_auc_cal,
                'roc_auc': roc_auc_cal,
                'brier_score': brier_cal,
                'method': cal_method
            }
            
            # Plot reliability diagram
            plot_reliability_diagram(y_test, rf_cal_proba, f"RandomForest_{cal_method}")
            
            rf_cal_results[cal_method] = {
                'proba': rf_cal_proba,
                'metrics': cal_metrics
            }
            
            # Compare with uncalibrated
            improvement = pr_auc_cal - rf_ap_uncal
            logger.info(f"  {cal_method} calibration improvement: {improvement:+.4f}")
            
        except Exception as e:
            logger.warning(f"  Classical calibration with {cal_method} failed: {e}")
    
    # Find best calibrated models
    best_quantum_cal = None
    best_quantum_ap = quantum_ap_uncal
    best_quantum_method = "none (uncalibrated)"
    
    for method, result in quantum_cal_results.items():
        if result['metrics']['pr_auc'] > best_quantum_ap:
            best_quantum_ap = result['metrics']['pr_auc']
            best_quantum_cal = result['proba']
            best_quantum_method = f"quantum_{method}"
    
    best_rf_cal = None
    best_rf_ap = rf_ap_uncal
    best_rf_method = "none (uncalibrated)"
    
    for method, result in rf_cal_results.items():
        if result['metrics']['pr_auc'] > best_rf_ap:
            best_rf_ap = result['metrics']['pr_auc']
            best_rf_cal = result['proba']
            best_rf_method = f"classical_{method}"
    
    # Compare all results
    logger.info("\n" + "="*80)
    logger.info("CALIBRATION RESULTS COMPARISON")
    logger.info("="*80)
    
    logger.info(f"Uncalibrated Quantum (QSVC)    : PR-AUC = {quantum_ap_uncal:.4f}")
    logger.info(f"Uncalibrated Classical (RF)    : PR-AUC = {rf_ap_uncal:.4f}")
    
    for method, result in quantum_cal_results.items():
        logger.info(f"Calibrated Quantum ({method})   : PR-AUC = {result['metrics']['pr_auc']:.4f}")
    
    for method, result in rf_cal_results.items():
        logger.info(f"Calibrated Classical ({method}): PR-AUC = {result['metrics']['pr_auc']:.4f}")
    
    logger.info(f"\nBest Quantum: {best_quantum_method} - PR-AUC: {best_quantum_ap:.4f}")
    logger.info(f"Best Classical: {best_rf_method} - PR-AUC: {best_rf_ap:.4f}")
    
    # Determine overall best
    if best_quantum_ap >= best_rf_ap:
        logger.info(f"🏆 Overall Best: Quantum ({best_quantum_method}) - PR-AUC: {best_quantum_ap:.4f}")
        best_overall = best_quantum_cal if best_quantum_cal is not None else quantum_pred_proba
        best_overall_method = best_quantum_method
    else:
        logger.info(f"🏆 Overall Best: Classical ({best_rf_method}) - PR-AUC: {best_rf_ap:.4f}")
        best_overall = best_rf_cal if best_rf_cal is not None else rf_pred_proba
        best_overall_method = best_rf_method
    
    # Save results
    results = {
        'relation': args.relation,
        'best_model_type': best_overall_method,
        'best_pr_auc': max(best_quantum_ap, best_rf_ap),
        'quantum_pr_auc': best_quantum_ap,
        'classical_pr_auc': best_rf_ap,
        'uncal_quantum_pr_auc': quantum_ap_uncal,
        'uncal_classical_pr_auc': rf_ap_uncal,
        'quantum_calibration_method': best_quantum_method,
        'classical_calibration_method': best_rf_method,
        'test_size': args.test_size,
        'embedding_dim': args.embedding_dim,
        'qml_dim': args.qml_dim
    }
    
    results_df = pd.DataFrame([results])
    timestamp = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
    results_file = f"results/calibration_results_{args.relation}_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    logger.info(f"Results saved to {results_file}")
    
    logger.info("\nCalibration techniques implementation completed!")


if __name__ == "__main__":
    main()