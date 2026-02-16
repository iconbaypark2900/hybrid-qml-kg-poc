"""
Robust Validation Script for Quantum Cure Prediction

This script performs cross-validation to properly assess model performance
and detect overfitting in the quantum models.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score, accuracy_score
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
from quantum_cure_prediction.quantum_enhanced_framework import compute_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_machine_learning.state_fidelities import ComputeUncompute
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    from qiskit.circuit.library import ZZFeatureMap
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False
    logger.warning("Quantum libraries not available for validation.")


def pr_auc_scorer(y_true, y_pred_proba):
    """Custom scorer for PR-AUC."""
    if len(np.unique(y_true)) < 2:
        return 0.0  # Return 0 if only one class is present
    return average_precision_score(y_true, y_pred_proba)


def validate_quantum_models_cv():
    """Perform cross-validation to assess quantum model performance."""
    if not QUANTUM_AVAILABLE:
        logger.error("Quantum libraries not available for validation.")
        return
    
    logger.info("Starting robust validation of quantum models with cross-validation...")
    
    # Load a small dataset for validation
    df_edges = load_hetionet_edges(data_dir="data")
    task_edges, entity_to_id, id_to_entity = extract_task_edges(
        df_edges, 
        relation_type="CtD", 
        max_entities=50  # Small dataset for validation
    )
    
    logger.info(f"Loaded {len(task_edges)} edges with {len(entity_to_id)} entities")
    
    # Prepare link prediction dataset
    train_df, test_df = prepare_link_prediction_dataset(
        task_edges,
        test_size=0.3,  # Larger test set for more robust evaluation
        random_state=42
    )
    
    logger.info(f"Dataset: train={len(train_df)}, test={len(test_df)}")
    
    # Train embeddings
    embedder = HetionetEmbedder(
        embedding_dim=64,
        work_dir="data"
    )
    
    # Prepare training data
    train_data = task_edges[['source', 'target']].copy()
    train_data.columns = ['source', 'target']
    train_data['metaedge'] = 'treats'
    
    embedder.train_embeddings(train_data)
    
    # Extract features for pairs
    def extract_pair_features(pairs_df):
        """Extract features for (compound, disease) pairs."""
        if embedder.entity_embeddings is None:
            raise ValueError("Embeddings not trained.")
        
        features = []
        for _, row in pairs_df.iterrows():
            source_id = int(row['source_id'])
            target_id = int(row['target_id'])
            
            # Get embeddings for source and target
            source_emb = embedder.entity_embeddings[source_id]
            target_emb = embedder.entity_embeddings[target_id]
            
            # Extract combined features
            concat_feat = np.concatenate([source_emb, target_emb])
            diff_feat = np.abs(source_emb - target_emb)
            hadamard_feat = source_emb * target_emb
            l2_dist = np.linalg.norm(source_emb - target_emb)
            cosine_sim = np.dot(source_emb, target_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8)
            
            all_features = np.concatenate([
                concat_feat, 
                diff_feat, 
                hadamard_feat, 
                [l2_dist, cosine_sim]
            ])
            
            features.append(all_features)
        
        return np.array(features)
    
    # Extract features
    X_train = extract_pair_features(train_df)
    y_train = train_df['label'].values
    
    # For quantum models, we need to reduce dimensions to match qubits
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10, random_state=42)  # 10 qubits
    X_train_reduced = pca.fit_transform(X_train)
    
    # Normalize to quantum-safe range
    X_train_reduced = np.tanh(X_train_reduced) * np.pi
    
    logger.info(f"Feature extraction completed: X_train shape = {X_train.shape}, reduced = {X_train_reduced.shape}")
    
    # Create quantum feature map
    feature_map = ZZFeatureMap(feature_dimension=10, reps=2, entanglement="linear")
    
    # Create quantum kernel
    sampler = AerSamplerV2()
    fidelity = ComputeUncompute(sampler=sampler)
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map.decompose(reps=10), fidelity=fidelity)
    
    # Create QSVC model
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    
    # Perform cross-validation
    logger.info("Performing cross-validation on QSVC model...")
    
    # Use StratifiedKFold for balanced splits
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduced folds due to small dataset
    
    # Define custom scoring functions
    def accuracy_scorer(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
    
    # Perform cross-validation for accuracy
    try:
        cv_accuracy_scores = cross_val_score(qsvc, X_train_reduced, y_train, 
                                           cv=skf, scoring='accuracy', n_jobs=1)
        logger.info(f"CV Accuracy scores: {cv_accuracy_scores}")
        logger.info(f"Mean CV Accuracy: {cv_accuracy_scores.mean():.4f} (+/- {cv_accuracy_scores.std() * 2:.4f})")
    except Exception as e:
        logger.error(f"Error in CV accuracy scoring: {e}")
        cv_accuracy_scores = []
    
    # Manual cross-validation for PR-AUC since it's not directly supported in older sklearn versions
    pr_auc_scores = []
    accuracy_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_reduced, y_train)):
        X_fold_train, X_fold_val = X_train_reduced[train_idx], X_train_reduced[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        try:
            # Train model
            qsvc_fold = QSVC(quantum_kernel=quantum_kernel)
            qsvc_fold.fit(X_fold_train, y_fold_train)
            
            # Predict
            y_pred = qsvc_fold.predict(X_fold_val)
            y_pred_proba = qsvc_fold.decision_function(X_fold_val)
            # Convert decision function to probabilities using sigmoid
            y_pred_proba = 1 / (1 + np.exp(-y_pred_proba))
            
            # Calculate metrics
            pr_auc = average_precision_score(y_fold_val, y_pred_proba)
            accuracy = accuracy_score(y_fold_val, y_pred)
            
            pr_auc_scores.append(pr_auc)
            accuracy_scores.append(accuracy)
            
            logger.info(f"Fold {fold_idx + 1}: PR-AUC = {pr_auc:.4f}, Accuracy = {accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Error in fold {fold_idx + 1}: {e}")
            pr_auc_scores.append(0.0)
            accuracy_scores.append(0.0)
    
    if pr_auc_scores:
        logger.info(f"\nCross-Validation Results:")
        logger.info(f"PR-AUC scores: {pr_auc_scores}")
        logger.info(f"Mean PR-AUC: {np.mean(pr_auc_scores):.4f} (+/- {np.std(pr_auc_scores) * 2:.4f})")
        logger.info(f"Accuracy scores: {accuracy_scores}")
        logger.info(f"Mean Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores) * 2:.4f})")
        
        # Check for overfitting indicators
        logger.info(f"\nOverfitting Analysis:")
        if np.mean(pr_auc_scores) > 0.95:
            logger.warning("⚠️  High mean PR-AUC suggests potential overfitting")
        if np.std(pr_auc_scores) > 0.15:  # High variance across folds
            logger.warning("⚠️  High variance across folds suggests potential overfitting")
        
        # Train final model on full dataset
        logger.info(f"\nTraining final model on full dataset...")
        qsvc.fit(X_train_reduced, y_train)
        
        # Evaluate on test set (previously unseen)
        X_test = extract_pair_features(test_df)
        X_test_reduced = pca.transform(X_test)
        X_test_reduced = np.tanh(X_test_reduced) * np.pi
        y_test = test_df['label'].values
        
        y_test_pred = qsvc.predict(X_test_reduced)
        y_test_pred_proba = qsvc.decision_function(X_test_reduced)
        y_test_pred_proba = 1 / (1 + np.exp(-y_test_pred_proba))
        
        test_pr_auc = average_precision_score(y_test, y_test_pred_proba)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        logger.info(f"Test Set Results:")
        logger.info(f"PR-AUC: {test_pr_auc:.4f}")
        logger.info(f"Accuracy: {test_accuracy:.4f}")
        
        # Overfitting detection
        cv_mean_pr_auc = np.mean(pr_auc_scores)
        if abs(cv_mean_pr_auc - test_pr_auc) > 0.15:
            logger.warning(f"⚠️  Large gap between CV PR-AUC ({cv_mean_pr_auc:.4f}) and Test PR-AUC ({test_pr_auc:.4f}) suggests overfitting")
        else:
            logger.info(f"✅ Reasonable agreement between CV and Test PR-AUC suggests good generalization")
        
        return {
            'cv_pr_auc_mean': cv_mean_pr_auc,
            'cv_pr_auc_std': np.std(pr_auc_scores),
            'cv_accuracy_mean': np.mean(accuracy_scores),
            'test_pr_auc': test_pr_auc,
            'test_accuracy': test_accuracy,
            'overfitting_indicators': {
                'high_cv_score': cv_mean_pr_auc > 0.95,
                'high_variance': np.std(pr_auc_scores) > 0.15,
                'cv_test_gap': abs(cv_mean_pr_auc - test_pr_auc) > 0.15
            }
        }
    
    else:
        logger.error("No valid CV scores obtained.")
        return None


def run_regularization_analysis():
    """Analyze the effect of regularization on quantum models."""
    logger.info("Running regularization analysis to prevent overfitting...")
    
    # This would involve testing different regularization approaches
    # For quantum models, regularization often involves:
    # - Reducing feature map complexity (fewer repetitions)
    # - Adding noise mitigation
    # - Using simpler ansatz circuits
    
    logger.info("Regularization recommendations:")
    logger.info("1. Reduce feature map repetitions from 2 to 1")
    logger.info("2. Use fewer qubits (reduce from 10 to 6-8)")
    logger.info("3. Apply noise mitigation techniques")
    logger.info("4. Increase training dataset size if possible")


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("ROBUST VALIDATION OF QUANTUM CURE PREDICTION MODELS")
    logger.info("="*80)
    
    # Run cross-validation analysis
    results = validate_quantum_models_cv()
    
    if results:
        logger.info("\nVALIDATION SUMMARY:")
        logger.info(f"Cross-Validation PR-AUC: {results['cv_pr_auc_mean']:.4f} ± {results['cv_pr_auc_std']:.4f}")
        logger.info(f"Test Set PR-AUC: {results['test_pr_auc']:.4f}")
        logger.info(f"Test Set Accuracy: {results['test_accuracy']:.4f}")
        
        logger.info("\nOVERFITTING ASSESSMENT:")
        indicators = results['overfitting_indicators']
        if any(indicators.values()):
            logger.warning("⚠️  OVERFITTING INDICATORS DETECTED:")
            if indicators['high_cv_score']:
                logger.warning("  - High cross-validation score (>0.95)")
            if indicators['high_variance']:
                logger.warning("  - High variance across folds")
            if indicators['cv_test_gap']:
                logger.warning("  - Large gap between CV and test scores")
        else:
            logger.info("✅ No significant overfitting indicators detected")
    
    # Run regularization analysis
    run_regularization_analysis()
    
    logger.info("\nValidation complete. Recommendations for reducing overfitting have been provided.")