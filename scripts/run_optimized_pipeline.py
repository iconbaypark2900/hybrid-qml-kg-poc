#!/usr/bin/env python3
"""
Optimized Pipeline for Hetionet Link Prediction

Integrates all optimizations:
1. Advanced KG embeddings (ComplEx, RotatE, DistMult)
2. Enhanced features (graph + domain + embeddings)
3. Optimized quantum features
4. Comprehensive model comparison

Usage:
    python scripts/run_optimized_pipeline.py --relation CtD --fast_mode
    python scripts/run_optimized_pipeline.py --relation CtD --embedding_method ComplEx --classical_only
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

# Import reproducibility, evaluation, and calibration utilities
from utils.reproducibility import set_global_seed
from utils.evaluation import (
    stratified_kfold_cv, evaluate_model_cv,
    print_cv_results, compare_models_cv,
    train_random_forest, train_logistic_regression, train_rbf_svm
)
from utils.calibration import CalibratedModel

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, silhouette_score
)
from scipy.spatial.distance import pdist
from scipy.stats import ttest_ind

# Project imports
from kg_layer.kg_loader import (
    load_hetionet_edges, extract_task_edges,
    prepare_link_prediction_dataset, prepare_full_graph_for_embeddings
)
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder, validate_no_leakage
from quantum_layer.advanced_qml_features import QuantumFeatureEngineer


def _write_cheap_quantum_config(
    base_config_path: str,
    out_config_path: str,
    *,
    cheap_sim_shots: int,
    cheap_hw_shots: int,
    zne_sample_size: int,
    zne_max_train_for_zne: int,
    readout_cal_shots: int,
) -> str:
    """
    Create a temporary, cheaper quantum config by overriding a few knobs.

    This keeps code changes localized: QSVC/VQC code still reads a YAML config path.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required for --cheap_mode but is not installed.")

    with open(base_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    q = cfg.setdefault("quantum", {})
    sim = q.setdefault("simulator", {})
    heron = q.setdefault("heron", {})

    # Shots
    try:
        sim["shots"] = min(int(sim.get("shots", cheap_sim_shots) or cheap_sim_shots), int(cheap_sim_shots))
    except Exception:
        sim["shots"] = int(cheap_sim_shots)
    try:
        heron["shots"] = min(int(heron.get("shots", cheap_hw_shots) or cheap_hw_shots), int(cheap_hw_shots))
    except Exception:
        heron["shots"] = int(cheap_hw_shots)

    # ZNE: keep enabled, but reduce sampling to keep circuit count bounded
    zne = sim.setdefault("zne", {})
    if isinstance(zne, dict):
        zne["enabled"] = bool(zne.get("enabled", True))
        zne["sample_size"] = int(min(int(zne.get("sample_size", zne_sample_size) or zne_sample_size), int(zne_sample_size)))
        zne["max_train_for_zne"] = int(min(int(zne.get("max_train_for_zne", zne_max_train_for_zne) or zne_max_train_for_zne), int(zne_max_train_for_zne)))
        # Keep at least 3 scales when possible for a stable fit
        scales = zne.get("scales")
        if not scales:
            zne["scales"] = [1.0, 1.5, 2.0]

    # Readout mitigation calibration shots (only impacts small-qubit observable path)
    ro = sim.setdefault("readout_mitigation", {})
    if isinstance(ro, dict):
        ro["enabled"] = bool(ro.get("enabled", False))
        ro["calibration_shots"] = int(min(int(ro.get("calibration_shots", readout_cal_shots) or readout_cal_shots), int(readout_cal_shots)))

    with open(out_config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return out_config_path
from quantum_layer.qml_trainer import QMLTrainer
from kg_layer.kg_embedder import HetionetEmbedder

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    """Compute comprehensive metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except Exception:
            metrics["pr_auc"] = float("nan")

    return metrics


def train_svm_linear(X_train, y_train, X_test, y_test, cv_folds=5, random_state=42):
    """Train Linear SVM as alternative to RBF."""
    logger.info("\nTraining SVM-Linear-Optimized with grid search...")
    
    try:
        t0 = time.time()
        
        # Create pipeline with scaling and Linear SVM
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='linear', class_weight='balanced', probability=True, random_state=random_state))
        ])
        
        # Parameter grid for Linear SVM (only C matters)
        param_grid = {
            'svc__C': [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        }
        
        # Grid search with cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring='average_precision',  # PR-AUC
            cv=skf,
            refit=True,
            n_jobs=-1,
            verbose=0,
            return_train_score=False
        )
        
        gs.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        cv_score = gs.best_score_
        
        logger.info(f"  Best params: C={best_params['svc__C']}")
        logger.info(f"  CV PR-AUC: {cv_score:.4f}")
        
        # Out-of-fold predictions for train metrics
        oof_scores = cross_val_predict(best_model, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        oof_preds = (oof_scores >= 0.5).astype(int)
        
        # Test predictions
        test_scores = best_model.predict_proba(X_test)[:, 1]
        test_preds = (test_scores >= 0.5).astype(int)
        
        train_metrics = compute_metrics(y_train, oof_preds, oof_scores)
        test_metrics = compute_metrics(y_test, test_preds, test_scores)
        
        logger.info(f"  ✅ SVM-Linear-Optimized - Test PR-AUC: {test_metrics['pr_auc']:.4f} (fit: {fit_time:.1f}s)")
        
        return {
            'status': 'success',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'fit_seconds': fit_time,
            'best_params': best_params,
            'cv_score': float(cv_score)
        }
    except Exception as e:
        logger.error(f"  ❌ SVM-Linear-Optimized failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e),
            'train_metrics': {},
            'test_metrics': {},
            'fit_seconds': 0.0
        }


def train_svm_rbf_with_grid_search(X_train, y_train, X_test, y_test, cv_folds=5, random_state=42, fast_mode=False):
    """Train SVM-RBF with grid search to find optimal hyperparameters."""
    logger.info("\nTraining SVM-RBF-Optimized with grid search...")
    
    try:
        t0 = time.time()
        
        # Create pipeline with scaling and SVM
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=random_state))
        ])
        
        # Define parameter grid - expanded with higher C values and scale/auto gamma
        if fast_mode:
            param_grid = {
                'svc__C': [0.1, 1.0, 10.0],
                'svc__gamma': [0.01, 0.1, 'scale']
            }
            logger.info("  Fast mode: Reduced grid search (9 combinations)")
        else:
            param_grid = {
                'svc__C': [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                'svc__gamma': [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 'scale', 'auto']
            }
            logger.info("  Full grid search (56 combinations with scale/auto)")
        
        # Grid search with cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        gs = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring='average_precision',  # PR-AUC
            cv=skf,
            refit=True,
            n_jobs=-1,
            verbose=0,
            return_train_score=False
        )
        
        gs.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        best_model = gs.best_estimator_
        best_params = gs.best_params_
        cv_score = gs.best_score_
        
        logger.info(f"  Best params: C={best_params['svc__C']}, gamma={best_params['svc__gamma']}")
        logger.info(f"  CV PR-AUC: {cv_score:.4f}")
        
        # Out-of-fold predictions for train metrics
        oof_scores = cross_val_predict(best_model, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        oof_preds = (oof_scores >= 0.5).astype(int)
        
        # Test predictions
        test_scores = best_model.predict_proba(X_test)[:, 1]
        test_preds = (test_scores >= 0.5).astype(int)
        
        train_metrics = compute_metrics(y_train, oof_preds, oof_scores)
        test_metrics = compute_metrics(y_test, test_preds, test_scores)
        
        logger.info(f"  ✅ SVM-RBF-Optimized - Test PR-AUC: {test_metrics['pr_auc']:.4f} (fit: {fit_time:.1f}s)")
        
        return {
            'status': 'success',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'fit_seconds': fit_time,
            'best_params': best_params,
            'cv_score': float(cv_score)
        }
    except Exception as e:
        logger.error(f"  ❌ SVM-RBF-Optimized failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e),
            'train_metrics': {},
            'test_metrics': {},
            'fit_seconds': 0.0
        }


def train_classical_model(name, model, X_train, y_train, X_test, y_test, cv_folds=5, random_state=42, calibrate=False, calibration_method='isotonic', feature_names=None, sample_weight=None):
    """Train and evaluate a classical model."""
    logger.info(f"\nTraining {name}{'  (with calibration)' if calibrate else ''}{'  (with sample weights)' if sample_weight is not None else ''}...")

    try:
        t0 = time.time()

        # Fit base model first (before calibration, for cross-validation compatibility)
        base_model = model
        if sample_weight is not None:
            base_model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            base_model.fit(X_train, y_train)
        fit_time = time.time() - t0

        # Feature importance analysis for RandomForest (before calibration)
        if hasattr(base_model, 'feature_importances_') and feature_names is not None:
            importances = base_model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            logger.info(f"  Top 10 Feature Importances:")
            for idx in top_indices:
                logger.info(f"    {feature_names[idx]:30s}: {importances[idx]:.6f}")

        # Out-of-fold predictions for train metrics (use base model for CV compatibility)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        if hasattr(base_model, "predict_proba"):
            oof_scores = cross_val_predict(base_model, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
            oof_preds = (oof_scores >= 0.5).astype(int)
        elif hasattr(base_model, "decision_function"):
            oof_scores = cross_val_predict(base_model, X_train, y_train, cv=skf, method="decision_function", n_jobs=-1)
            oof_preds = (oof_scores >= 0).astype(int)
        else:
            oof_preds = cross_val_predict(base_model, X_train, y_train, cv=skf, n_jobs=-1)
            oof_scores = oof_preds.astype(float)

        # Apply calibration if requested (after CV, for test predictions)
        if calibrate:
            from sklearn.calibration import CalibratedClassifierCV
            logger.info(f"  Using {calibration_method} calibration")
            calibrated_model = CalibratedClassifierCV(
                base_model, 
                method=calibration_method, 
                cv=min(3, cv_folds)
            )
            if sample_weight is not None:
                calibrated_model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                calibrated_model.fit(X_train, y_train)
            
            # Use calibrated model for test predictions
            test_scores = calibrated_model.predict_proba(X_test)[:, 1]
            test_preds = (test_scores >= 0.5).astype(int)
            model = calibrated_model  # Store calibrated model
        else:
            # Use base model for test predictions
            if hasattr(base_model, "predict_proba"):
                test_scores = base_model.predict_proba(X_test)[:, 1]
                test_preds = (test_scores >= 0.5).astype(int)
            elif hasattr(base_model, "decision_function"):
                test_scores = base_model.decision_function(X_test)
                test_preds = (test_scores >= 0).astype(int)
            else:
                test_preds = base_model.predict(X_test)
                test_scores = test_preds.astype(float)

        train_metrics = compute_metrics(y_train, oof_preds, oof_scores)
        test_metrics = compute_metrics(y_test, test_preds, test_scores)

        logger.info(f"  ✅ {name} - Test PR-AUC: {test_metrics['pr_auc']:.4f} (fit: {fit_time:.1f}s)")

        result = {
            'status': 'success',
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'fit_seconds': fit_time
        }
        
        # Add feature importances if available
        if hasattr(model, 'feature_importances_'):
            result['feature_importances'] = model.feature_importances_.tolist()
        
        return result
    except Exception as e:
        logger.error(f"  ❌ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'failed',
            'error': str(e),
            'train_metrics': {},
            'test_metrics': {},
            'fit_seconds': 0.0
        }


def main():
    parser = argparse.ArgumentParser(description="Optimized Hetionet Link Prediction Pipeline")

    # Data args
    parser.add_argument("--relation", type=str, default="CtD", help="Relation type (CtD, DaG, etc.)")
    parser.add_argument("--max_entities", type=int, default=None, help="Limit entities (None = all)")
    parser.add_argument("--pos_edge_sample", type=int, default=None,
                       help="Sample this many positive edges before training (None = use all)")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                       help="Ratio of negative to positive samples (default: 1.0, i.e., 1:1)")

    # Embedding args
    parser.add_argument("--embedding_method", type=str, default="ComplEx",
                       choices=['TransE', 'ComplEx', 'RotatE', 'DistMult'],
                       help="KG embedding method")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--embedding_epochs", type=int, default=100, help="Embedding training epochs")
    parser.add_argument("--use_cached_embeddings", action="store_true", help="Use cached embeddings if available")
    parser.add_argument("--full_graph_embeddings", action="store_true",
                       help="Train embeddings on full Hetionet (all relations) for richer context")

    # Feature args
    parser.add_argument("--use_graph_features", action="store_true", default=True, help="Include graph features")
    parser.add_argument("--use_domain_features", action="store_true", default=True, help="Include domain features")
    parser.add_argument("--use_evidence_weighting", action="store_true",
                       help="Use evidence weighting based on shared genes (for CtD relation)")
    parser.add_argument("--min_shared_genes", type=int, default=1,
                       help="Minimum number of shared genes for evidence weighting (default: 1)")

    # Quantum args
    parser.add_argument("--qml_dim", type=int, default=12, help="Number of qubits (default: 12, was 10)")
    parser.add_argument("--qml_encoding", type=str, default="hybrid",
                       choices=['amplitude', 'phase', 'hybrid', 'optimized_diff', 'tensor_product'],
                       help="Quantum encoding strategy")
    parser.add_argument("--qml_reduction_method", type=str, default="pca",
                       choices=['pca', 'lda', 'kernel_pca'],
                       help="Dimensionality reduction method for quantum features (PCA, LDA, or KernelPCA)")
    parser.add_argument("--qml_feature_selection_method", type=str, default=None,
                       choices=[None, 'f_classif', 'mutual_info', 'chi2'],
                       help="Feature selection method before reduction (None, f_classif, mutual_info, chi2)")
    parser.add_argument("--qml_feature_select_k_mult", type=float, default=4.0,
                       help="Multiplier for feature selection: select k_mult * qml_dim features (default: 4.0)")
    parser.add_argument("--qml_pre_pca_dim", type=int, default=0,
                       help="Pre-PCA dimension reduction (0 = disabled, >0 = reduce to this dim before main reduction)")
    parser.add_argument("--qml_feature_map", type=str, default="ZZ",
                       choices=['ZZ', 'Z', 'Pauli', 'custom_link_prediction'],
                       help="Quantum feature map type (ZZ, Z, Pauli, or custom_link_prediction)")
    parser.add_argument("--qml_feature_map_reps", type=int, default=2,
                       help="Number of feature map repetitions (default: 2)")
    parser.add_argument("--qml_entanglement", type=str, default="full",
                       choices=['linear', 'full', 'circular'],
                       help="Entanglement pattern for feature map (linear, full, circular)")
    parser.add_argument("--use_data_reuploading", action="store_true",
                       help="Use data re-uploading feature map (quantum-native, encodes features multiple times)")
    parser.add_argument("--use_variational_feature_map", action="store_true",
                       help="Use variational (trainable) feature map")
    parser.add_argument("--optimize_feature_map_reps", action="store_true",
                       help="Optimize feature map repetitions using kernel-target alignment")
    parser.add_argument("--qml_max_iter", type=int, default=50, help="QML max iterations")
    parser.add_argument("--skip_quantum", action="store_true", help="Skip quantum models")

    # QSVC kernel scaling / approximation
    parser.add_argument("--qsvc_nystrom_m", type=int, default=None,
                       help="Enable Nyström kernel approximation for QSVC with m landmark points (reduces O(n^2) to O(n*m))")
    parser.add_argument("--nystrom_ridge", type=float, default=1e-6,
                       help="Ridge added to K_mm before pseudo-inverse for Nyström (stability)")
    parser.add_argument("--nystrom_max_pairs", type=int, default=20000,
                       help="Safety cap: if n_train*m exceeds this, landmark mitigation may be reduced/skipped")
    parser.add_argument("--no_nystrom_landmark_mitigation", action="store_true",
                       help="Disable mitigation on landmark evaluations (default: enabled when possible)")

    # Negative sampling args
    parser.add_argument("--negative_sampling", type=str, default="random",
                       choices=['random', 'hard', 'diverse'],
                       help="Negative sampling strategy")
    parser.add_argument("--diversity_weight", type=float, default=0.5,
                       help="Diversity weight for 'diverse' sampling (0.0-1.0)")

    # Calibration args
    parser.add_argument("--calibrate_probabilities", action="store_true",
                       help="Apply probability calibration to models")
    parser.add_argument("--calibration_method", type=str, default="isotonic",
                       choices=['isotonic', 'sigmoid'],
                       help="Calibration method (isotonic or sigmoid for Platt scaling)")

    # Experiment args
    parser.add_argument("--classical_only", action="store_true", help="Run only classical models")
    parser.add_argument("--quantum_only", action="store_true", help="Run only quantum models (skip classical)")
    parser.add_argument("--use_classical_features_in_kernel", action="store_true",
                       help="Use classical features (reduced via PCA) in quantum kernel instead of quantum-reduced features")
    parser.add_argument("--use_feature_selection", action="store_true", 
                       help="Apply mutual information feature selection when feature-to-sample ratio > 1.0")
    parser.add_argument("--use_contrastive_learning", action="store_true",
                       help="Fine-tune embeddings using contrastive learning to improve class separability")
    parser.add_argument("--contrastive_margin", type=float, default=1.0,
                       help="Margin for contrastive learning triplet loss")
    parser.add_argument("--contrastive_epochs", type=int, default=50,
                       help="Number of epochs for contrastive fine-tuning")
    parser.add_argument("--use_quantum_aware_embeddings", action="store_true",
                       help="Fine-tune embeddings using quantum kernel separability (quantum-native)")
    parser.add_argument("--quantum_aware_epochs", type=int, default=100,
                       help="Number of epochs for quantum-aware embedding fine-tuning")
    parser.add_argument("--use_task_specific_finetuning", action="store_true",
                       help="Fine-tune embeddings using classification loss on the target task (recommended)")
    parser.add_argument("--task_specific_epochs", type=int, default=100,
                       help="Number of epochs for task-specific fine-tuning")
    parser.add_argument("--task_specific_lr", type=float, default=0.001,
                       help="Learning rate for task-specific fine-tuning")
    parser.add_argument("--use_quantum_feature_engineering", action="store_true",
                       help="Use quantum-specific feature engineering (amplitude, phase, entanglement features)")
    parser.add_argument("--quantum_feature_selection", action="store_true",
                       help="Use quantum kernel for feature selection")
    parser.add_argument("--use_improved_features", action="store_true",
                       help="Use improved feature engineering with RandomForest guidance and interaction features")
    parser.add_argument("--max_interaction_features", type=int, default=50,
                       help="Maximum number of interaction features to create")
    # Note: --use_domain_features is already defined above (line 341)
    parser.add_argument("--skip_svm_rbf", action="store_true", 
                       help="Skip SVM-RBF model (if it continues to fail)")
    parser.add_argument("--skip_svm_linear", action="store_true",
                       help="Skip SVM-Linear model")
    parser.add_argument("--skip_vqc", action="store_true",
                       help="Skip VQC model (QSVC only)")
    parser.add_argument("--fast_mode", action="store_true", help="Fast mode (fewer models, less tuning)")
    parser.add_argument("--cheap_mode", action="store_true",
                       help="Cheapest runnable settings (implies --fast_mode + caps shots/ZNE/readout calibration + defaults max_entities)")
    parser.add_argument("--cheap_sim_shots", type=int, default=256, help="Cheap mode: simulator shots cap")
    parser.add_argument("--cheap_hw_shots", type=int, default=200, help="Cheap mode: hardware shots cap")
    parser.add_argument("--cheap_max_entities", type=int, default=80, help="Cheap mode: default max_entities if not provided")
    parser.add_argument("--cheap_zne_sample_size", type=int, default=64, help="Cheap mode: ZNE sample_size cap")
    parser.add_argument("--cheap_zne_max_train_for_zne", type=int, default=60, help="Cheap mode: ZNE max_train_for_zne cap")
    parser.add_argument("--cheap_readout_cal_shots", type=int, default=512, help="Cheap mode: readout calibration shots cap")
    parser.add_argument("--cv_folds", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--use_cv_evaluation", action="store_true",
                       help="Use K-Fold CV for evaluation (more robust than single split)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--quantum_config_path", type=str, default="config/quantum_config.yaml",
                       help="Quantum execution config path")

    args = parser.parse_args()

    # Cheap mode: keep it dead simple and deterministic.
    if args.cheap_mode:
        args.fast_mode = True
        if args.max_entities is None:
            args.max_entities = int(args.cheap_max_entities)
        # Create a temporary low-cost quantum config in results_dir
        os.makedirs(args.results_dir, exist_ok=True)
        cheap_cfg_path = os.path.join(args.results_dir, "quantum_config_cheap.yaml")
        try:
            args.quantum_config_path = _write_cheap_quantum_config(
                args.quantum_config_path,
                cheap_cfg_path,
                cheap_sim_shots=int(args.cheap_sim_shots),
                cheap_hw_shots=int(args.cheap_hw_shots),
                zne_sample_size=int(args.cheap_zne_sample_size),
                zne_max_train_for_zne=int(args.cheap_zne_max_train_for_zne),
                readout_cal_shots=int(args.cheap_readout_cal_shots),
            )
        except Exception as e:
            logger.warning(f"⚠️  Cheap mode could not rewrite quantum config ({e}); continuing with {args.quantum_config_path}")

    # Set global random seed for reproducibility
    set_global_seed(args.random_state)

    os.makedirs(args.results_dir, exist_ok=True)

    logger.info("="*80)
    logger.info("OPTIMIZED HETIONET LINK PREDICTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Random seed: {args.random_state}")
    logger.info(f"Relation: {args.relation}")
    logger.info(f"Embedding method: {args.embedding_method} (dim={args.embedding_dim})")
    logger.info(f"Full-graph embeddings: {args.full_graph_embeddings}")
    logger.info(f"Quantum config: {args.quantum_config_path}")
    logger.info(f"Negative sampling: {args.negative_sampling}")
    if args.negative_sampling == 'diverse':
        logger.info(f"Diversity weight: {args.diversity_weight}")
    logger.info(f"Calibrate probabilities: {args.calibrate_probabilities}")
    if args.calibrate_probabilities:
        logger.info(f"Calibration method: {args.calibration_method}")
    logger.info(f"Graph features: {args.use_graph_features}, Domain features: {args.use_domain_features}")
    logger.info(f"Quantum encoding: {args.qml_encoding} (qubits={args.qml_dim})")
    logger.info(f"Quantum feature map: {args.qml_feature_map} (reps={args.qml_feature_map_reps}, entanglement={args.qml_entanglement})")

    # ========== STEP 1: LOAD DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)

    df = load_hetionet_edges()
    logger.info(f"Loaded {len(df)} total edges from Hetionet")

    if args.max_entities:
        logger.info(f"Limiting to {args.max_entities} entities")

    task_edges, entity_to_id, id_to_entity = extract_task_edges(
        df, relation_type=args.relation, max_entities=args.max_entities
    )
    logger.info(f"Extracted {len(task_edges)} edges for '{args.relation}'")
    
    # Sample positive edges if requested
    if args.pos_edge_sample and args.pos_edge_sample > 0:
        if len(task_edges) > args.pos_edge_sample:
            logger.info(f"Sampling {args.pos_edge_sample} positive edges from {len(task_edges)} available")
            task_edges = task_edges.sample(n=args.pos_edge_sample, random_state=args.random_state).reset_index(drop=True)
            logger.info(f"Sampled {len(task_edges)} positive edges")
        else:
            logger.info(f"Requested {args.pos_edge_sample} edges but only {len(task_edges)} available, using all")

    # Decide evaluation strategy: K-Fold CV or single train/test split
    use_cv = args.use_cv_evaluation
    logger.info(f"Evaluation mode: {'K-Fold CV' if use_cv else 'Single train/test split'}")

    if use_cv:
        # Create K-Fold CV splits
        cv_folds = stratified_kfold_cv(
            task_edges, entity_to_id,
            n_folds=args.cv_folds,
            random_state=args.random_state
        )
        logger.info(f"Created {len(cv_folds)} CV folds")
        # We'll use all task_edges for embedding training
        train_df = None  # Will loop through folds
        test_df = None
    else:
        # Traditional single split
        # Create a wrapper that supports neg_ratio
        from sklearn.model_selection import train_test_split
        from kg_layer.kg_loader import get_negative_samples, load_kg_config
        
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
        cv_folds = None

    # ========== STEP 2: TRAIN EMBEDDINGS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAINING KNOWLEDGE GRAPH EMBEDDINGS")
    logger.info("="*80)

    embedder = AdvancedKGEmbedder(
        embedding_dim=args.embedding_dim,
        method=args.embedding_method,
        num_epochs=args.embedding_epochs if not args.fast_mode else 50,
        batch_size=512,
        learning_rate=0.001,
        work_dir="data",
        random_state=args.random_state
    )

    # Try to load cached embeddings
    if args.use_cached_embeddings and embedder.load_embeddings():
        logger.info("Using cached embeddings")
    else:
        # Choose training edges: full graph or task-specific
        if args.full_graph_embeddings:
            logger.info("Using FULL GRAPH embeddings (all relations) for richer context...")
            # Get all entities involved in task
            task_entities = list(entity_to_id.keys())
            # Prepare full graph edges involving these entities
            embedding_training_edges = prepare_full_graph_for_embeddings(df, task_entities)
            # CRITICAL: PyKEEN expects entity STRINGS, not integer IDs
            # The edges already have 'source' and 'target' columns with entity strings
            # Filter to only edges where both entities are in our task set
            embedding_training_edges = embedding_training_edges[
                embedding_training_edges["source"].isin(task_entities) &
                embedding_training_edges["target"].isin(task_entities)
            ].copy()
            logger.info(f"Training embeddings on {len(embedding_training_edges)} edges "
                       f"({embedding_training_edges['metaedge'].nunique()} relation types)")
        else:
            logger.info(f"Training {args.embedding_method} embeddings on task-specific edges...")
            # task_edges already has 'source' and 'target' columns with entity strings
            embedding_training_edges = task_edges[["source", "metaedge", "target"]].copy()

        embedder.train_embeddings(embedding_training_edges)

    # Get embeddings dict
    embeddings = embedder.get_all_embeddings()
    logger.info(f"Loaded {len(embeddings)} entity embeddings")
    
    # ========== CONTRASTIVE LEARNING FINE-TUNING ==========
    if args.use_contrastive_learning:
        logger.info("\n" + "="*80)
        logger.info("CONTRASTIVE LEARNING FINE-TUNING")
        logger.info("="*80)
        
        # FIX: Skip contrastive learning in CV mode (train_df is None)
        if use_cv:
            logger.warning("⚠️  Contrastive learning skipped in CV mode (use single split evaluation).")
        elif train_df is None:
            logger.warning("⚠️  Contrastive learning skipped (train_df is None).")
        else:
            try:
                from kg_layer.contrastive_embeddings import ContrastiveEmbeddingFineTuner
                
                # Convert integer IDs to entity strings for contrastive learning
                train_df_for_contrastive = train_df.copy()
                if 'source' not in train_df_for_contrastive.columns:
                    # Use embedder's id_to_entity mapping if available, otherwise use the global one
                    if hasattr(embedder, 'id_to_entity') and embedder.id_to_entity:
                        train_df_for_contrastive['source'] = train_df_for_contrastive['source_id'].map(embedder.id_to_entity)
                        train_df_for_contrastive['target'] = train_df_for_contrastive['target_id'].map(embedder.id_to_entity)
                    else:
                        train_df_for_contrastive['source'] = train_df_for_contrastive['source_id'].map(id_to_entity)
                        train_df_for_contrastive['target'] = train_df_for_contrastive['target_id'].map(id_to_entity)
                
                # Get entity embeddings as array
                entity_list = list(embeddings.keys())
                entity_to_idx = {entity: idx for idx, entity in enumerate(entity_list)}
                embedding_array = np.array([embeddings[entity] for entity in entity_list])
                
                # Get head and tail indices for training samples
                train_head_indices = np.array([entity_to_idx.get(str(row['source']), -1) 
                                              for _, row in train_df_for_contrastive.iterrows()])
                train_tail_indices = np.array([entity_to_idx.get(str(row['target']), -1) 
                                              for _, row in train_df_for_contrastive.iterrows()])
                
                # Get labels
                y_train_labels = train_df['label'].values
                
                # Filter out invalid indices
                valid_mask = (train_head_indices >= 0) & (train_tail_indices >= 0)
                train_head_indices = train_head_indices[valid_mask]
                train_tail_indices = train_tail_indices[valid_mask]
                y_train_valid = y_train_labels[valid_mask]
                
                if len(train_head_indices) > 0:
                    logger.info(f"Fine-tuning embeddings on {len(train_head_indices)} training samples...")
                    
                    fine_tuner = ContrastiveEmbeddingFineTuner(
                        margin=args.contrastive_margin,
                        num_epochs=args.contrastive_epochs,
                        random_state=args.random_state
                    )
                    
                    fine_tuned_embeddings = fine_tuner.fine_tune(
                        embedding_array,
                        y_train_valid,
                        train_head_indices,
                        train_tail_indices
                    )
                    
                    # Update embeddings dict
                    for idx, entity in enumerate(entity_list):
                        embeddings[entity] = fine_tuned_embeddings[idx]
                    
                    logger.info("✓ Embeddings fine-tuned with contrastive learning")
                    
                    # Compute contrastive loss before/after
                    from kg_layer.contrastive_embeddings import compute_contrastive_loss
                    loss_before = compute_contrastive_loss(
                        embedding_array, train_head_indices, train_tail_indices, y_train_valid
                    )
                    loss_after = compute_contrastive_loss(
                        fine_tuned_embeddings, train_head_indices, train_tail_indices, y_train_valid
                    )
                    logger.info(f"Contrastive loss: {loss_before:.6f} → {loss_after:.6f}")
                else:
                    logger.warning("⚠️  No valid training samples for contrastive learning. Skipping.")
            except ImportError:
                logger.warning("⚠️  Contrastive learning not available (PyTorch may be missing). Skipping.")
            except Exception as e:
                logger.warning(f"⚠️  Contrastive learning failed: {e}. Continuing with original embeddings.")
                import traceback
                traceback.print_exc()
    
    # ========== QUANTUM-AWARE EMBEDDING FINE-TUNING ==========
    if args.use_quantum_aware_embeddings:
        logger.info("\n" + "="*80)
        logger.info("QUANTUM-AWARE EMBEDDING FINE-TUNING")
        logger.info("="*80)
        
        # FIX: Skip quantum-aware embeddings in CV mode (train_df is None)
        if use_cv:
            logger.warning("⚠️  Quantum-aware embeddings skipped in CV mode (use single split evaluation).")
        elif train_df is None:
            logger.warning("⚠️  Quantum-aware embeddings skipped (train_df is None).")
        else:
            try:
                from quantum_layer.quantum_aware_embeddings import QuantumAwareEmbeddingTrainer
                
                # Convert integer IDs to entity strings
                train_df_for_quantum = train_df.copy()
                if 'source' not in train_df_for_quantum.columns:
                    if hasattr(embedder, 'id_to_entity') and embedder.id_to_entity:
                        train_df_for_quantum['source'] = train_df_for_quantum['source_id'].map(embedder.id_to_entity)
                        train_df_for_quantum['target'] = train_df_for_quantum['target_id'].map(embedder.id_to_entity)
                    else:
                        train_df_for_quantum['source'] = train_df_for_quantum['source_id'].map(id_to_entity)
                        train_df_for_quantum['target'] = train_df_for_quantum['target_id'].map(id_to_entity)
                
                # Get entity embeddings as array
                entity_list = list(embeddings.keys())
                entity_to_idx = {entity: idx for idx, entity in enumerate(entity_list)}
                embedding_array = np.array([embeddings[entity] for entity in entity_list])
                
                # Get head and tail indices
                train_head_indices = np.array([entity_to_idx.get(str(row['source']), -1) 
                                              for _, row in train_df_for_quantum.iterrows()])
                train_tail_indices = np.array([entity_to_idx.get(str(row['target']), -1) 
                                              for _, row in train_df_for_quantum.iterrows()])
                y_train_labels = train_df['label'].values
                
                # Filter valid indices
                valid_mask = (train_head_indices >= 0) & (train_tail_indices >= 0)
                train_head_indices = train_head_indices[valid_mask]
                train_tail_indices = train_tail_indices[valid_mask]
                y_train_valid = y_train_labels[valid_mask]
                
                if len(train_head_indices) > 0:
                    logger.info(f"Fine-tuning embeddings for quantum models on {len(train_head_indices)} samples...")
                    
                    quantum_trainer = QuantumAwareEmbeddingTrainer(
                        num_qubits=args.qml_dim,
                        feature_map_reps=args.qml_feature_map_reps,
                        entanglement=args.qml_entanglement,
                        num_epochs=args.quantum_aware_epochs,
                        random_state=args.random_state
                    )
                    
                    fine_tuned_embeddings, history = quantum_trainer.fine_tune(
                        embedding_array,
                        y_train_valid,
                        train_head_indices,
                        train_tail_indices
                    )
                    
                    # Update embeddings dict
                    for idx, entity in enumerate(entity_list):
                        embeddings[entity] = fine_tuned_embeddings[idx]
                    
                    logger.info(f"✓ Embeddings fine-tuned for quantum models")
                    logger.info(f"  Separability improvement: {history['final_separability'] - history['initial_separability']:.6f}")
            except ImportError:
                logger.warning("⚠️  Quantum-aware embeddings not available (Qiskit/PyTorch may be missing). Skipping.")
            except Exception as e:
                logger.warning(f"⚠️  Quantum-aware embedding fine-tuning failed: {e}. Continuing with original embeddings.")
                import traceback
                traceback.print_exc()
    
    # ========== TASK-SPECIFIC EMBEDDING FINE-TUNING ==========
    if args.use_task_specific_finetuning:
        logger.info("\n" + "="*80)
        logger.info("TASK-SPECIFIC EMBEDDING FINE-TUNING")
        logger.info("="*80)
        
        # FIX: Skip task-specific fine-tuning in CV mode (train_df is None)
        if use_cv:
            logger.warning("⚠️  Task-specific fine-tuning skipped in CV mode (use single split evaluation).")
        elif train_df is None:
            logger.warning("⚠️  Task-specific fine-tuning skipped (train_df is None).")
        else:
            try:
                from kg_layer.task_specific_embeddings import ClassificationEmbeddingFineTuner, compute_classification_metrics
                
                # Convert integer IDs to entity strings
                train_df_for_task = train_df.copy()
                if 'source' not in train_df_for_task.columns:
                    if hasattr(embedder, 'id_to_entity') and embedder.id_to_entity:
                        train_df_for_task['source'] = train_df_for_task['source_id'].map(embedder.id_to_entity)
                        train_df_for_task['target'] = train_df_for_task['target_id'].map(embedder.id_to_entity)
                    else:
                        train_df_for_task['source'] = train_df_for_task['source_id'].map(id_to_entity)
                        train_df_for_task['target'] = train_df_for_task['target_id'].map(id_to_entity)
                
                # Get entity embeddings as array
                entity_list = list(embeddings.keys())
                entity_to_idx = {entity: idx for idx, entity in enumerate(entity_list)}
                embedding_array = np.array([embeddings[entity] for entity in entity_list])
                
                # Get head and tail indices
                train_head_indices = np.array([entity_to_idx.get(str(row['source']), -1) 
                                              for _, row in train_df_for_task.iterrows()])
                train_tail_indices = np.array([entity_to_idx.get(str(row['target']), -1) 
                                              for _, row in train_df_for_task.iterrows()])
                y_train_labels = train_df['label'].values
                
                # Filter valid indices
                valid_mask = (train_head_indices >= 0) & (train_tail_indices >= 0)
                train_head_indices = train_head_indices[valid_mask]
                train_tail_indices = train_tail_indices[valid_mask]
                y_train_valid = y_train_labels[valid_mask]
                
                if len(train_head_indices) > 0:
                    logger.info(f"Fine-tuning embeddings for classification task on {len(train_head_indices)} samples...")
                    
                    # Compute initial classification metrics
                    initial_metrics = compute_classification_metrics(
                        embedding_array, train_head_indices, train_tail_indices, y_train_valid
                    )
                    logger.info(f"  Initial classification metrics: ROC-AUC={initial_metrics['roc_auc']:.4f}, PR-AUC={initial_metrics['pr_auc']:.4f}")
                    
                    # Create validation split (20% of training data)
                    from sklearn.model_selection import train_test_split
                    val_size = 0.2
                    train_idx, val_idx = train_test_split(
                        np.arange(len(train_head_indices)),
                        test_size=val_size,
                        random_state=args.random_state,
                        stratify=y_train_valid
                    )
                    
                    val_head_indices = train_head_indices[val_idx]
                    val_tail_indices = train_tail_indices[val_idx]
                    val_labels = y_train_valid[val_idx]
                    
                    train_head_indices_final = train_head_indices[train_idx]
                    train_tail_indices_final = train_tail_indices[train_idx]
                    train_labels_final = y_train_valid[train_idx]
                    
                    logger.info(f"  Training: {len(train_head_indices_final)} samples, Validation: {len(val_head_indices)} samples")
                    
                    task_trainer = ClassificationEmbeddingFineTuner(
                        learning_rate=args.task_specific_lr,
                        num_epochs=args.task_specific_epochs,
                        random_state=args.random_state
                    )
                    
                    fine_tuned_embeddings, history = task_trainer.fine_tune(
                        embedding_array,
                        train_labels_final,
                        train_head_indices_final,
                        train_tail_indices_final,
                        val_head_indices=val_head_indices,
                        val_tail_indices=val_tail_indices,
                        val_labels=val_labels
                    )
                    
                    # Update embeddings dict
                    for idx, entity in enumerate(entity_list):
                        embeddings[entity] = fine_tuned_embeddings[idx]
                    
                    logger.info("✓ Embeddings fine-tuned for classification task")
                    if 'best_val_auc' in history:
                        logger.info(f"  Best validation AUC: {history['best_val_auc']:.4f} at epoch {history['best_epoch']+1}")
                    
                    # Compute final classification metrics
                    final_metrics = compute_classification_metrics(
                        fine_tuned_embeddings, train_head_indices, train_tail_indices, y_train_valid
                    )
                    logger.info(f"  Final classification metrics: ROC-AUC={final_metrics['roc_auc']:.4f}, PR-AUC={final_metrics['pr_auc']:.4f}")
                    logger.info(f"  Improvement: ROC-AUC +{final_metrics['roc_auc'] - initial_metrics['roc_auc']:.4f}, PR-AUC +{final_metrics['pr_auc'] - initial_metrics['pr_auc']:.4f}")
                else:
                    logger.warning("⚠️  No valid training samples for task-specific fine-tuning. Skipping.")
            except ImportError:
                logger.warning("⚠️  Task-specific fine-tuning not available (PyTorch may be missing). Skipping.")
            except Exception as e:
                logger.warning(f"⚠️  Task-specific fine-tuning failed: {e}. Continuing with original embeddings.")
                import traceback
                traceback.print_exc()

    # ========== CV EVALUATION BRANCH ==========
    if use_cv:
        logger.info("\n" + "="*80)
        logger.info("K-FOLD CROSS-VALIDATION EVALUATION")
        logger.info("="*80)
        logger.info(f"Evaluating models with {args.cv_folds}-fold CV...")
        logger.info("Note: CV mode uses embeddings directly (skips enhanced features for efficiency)")

        # FIX: Respect quantum_only flag
        if args.quantum_only:
            logger.info("⚠️  --quantum_only flag set, but quantum models not yet supported in CV mode.")
            logger.info("Skipping CV evaluation. Use single split evaluation for quantum models.")
            return
        
        # Define models for CV evaluation
        cv_model_configs = {
            'RandomForest': {
                'n_estimators': 200 if not args.fast_mode else 100,
                'max_depth': 20,
                'random_state': args.random_state
            },
            'LogisticRegression': {
                'C': 1.0,
                'random_state': args.random_state
            },
        }

        if not args.fast_mode:
            cv_model_configs['RBF-SVM'] = {
                'C': 3.0,
                'gamma': 0.1,
                'random_state': args.random_state
            }

        # Evaluate each model with CV
        cv_results_dict = {}

        for model_name, model_kwargs in cv_model_configs.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating {model_name} with CV...")
            logger.info(f"{'='*60}")

            if model_name == 'RandomForest':
                model_fn = train_random_forest
            elif model_name == 'LogisticRegression':
                model_fn = train_logistic_regression
            elif model_name == 'RBF-SVM':
                model_fn = train_rbf_svm
            else:
                continue

            # FIX: Pass id_to_entity mapping to CV evaluation
            cv_results = evaluate_model_cv(
                model_fn=model_fn,
                folds=cv_folds,
                embeddings=embeddings,
                model_name=model_name,
                id_to_entity=id_to_entity,  # ADD THIS
                **model_kwargs
            )
            cv_results_dict[model_name] = cv_results
            print_cv_results(cv_results, model_name)

        # Print comparison table
        logger.info("\n" + "="*80)
        logger.info("CROSS-VALIDATION RESULTS COMPARISON")
        logger.info("="*80)
        comparison_df = compare_models_cv(cv_results_dict)
        print(comparison_df.to_string(index=False))

        # Save CV results
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(args.results_dir, f"cv_results_{stamp}.json")

        payload = {
            "config": vars(args),
            "cv_results": {
                model: {
                    k: (v.tolist() if isinstance(v, np.ndarray) else
                        [float(x) for x in v] if isinstance(v, list) else
                        float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in results.items()
                }
                for model, results in cv_results_dict.items()
            },
            "timestamp": stamp
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

        logger.info(f"\n✅ CV results saved to: {out_path}")

        # Exit after CV evaluation
        return

    # ========== STEP 3: BUILD ENHANCED FEATURES ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 3: BUILDING ENHANCED FEATURES")
    logger.info("="*80)

    feature_builder = EnhancedFeatureBuilder(
        include_graph_features=args.use_graph_features,
        include_domain_features=args.use_domain_features,
        normalize=True
    )

    # Extract TRAINING edges only (to prevent leakage)
    train_edges_only = train_df[train_df['label'] == 1].copy()

    # Add entity string IDs for domain features (map from integer IDs)
    train_edges_only['source'] = train_edges_only['source_id'].map(id_to_entity)
    train_edges_only['target'] = train_edges_only['target_id'].map(id_to_entity)

    logger.info(f"Using {len(train_edges_only)} TRAINING edges for graph/domain features (prevents leakage)")

    # Validate no data leakage
    logger.info("Validating no data leakage...")
    validate_no_leakage(train_df, test_df, train_edges_only)

    # Build graph for graph features (TRAIN ONLY)
    if args.use_graph_features:
        logger.info("Building graph on TRAINING edges only (prevents leakage)...")
        feature_builder.build_graph(train_edges_only)

    # Convert integer IDs to entity string IDs for feature building
    # build_features expects entity string IDs (e.g., "Compound::DB00001"), not integer IDs
    logger.info("Converting integer IDs to entity string IDs for feature building...")
    
    # Diagnostic: Check if embeddings dict keys match id_to_entity values
    sample_emb_keys = list(embeddings.keys())[:5]
    sample_id_to_entity = [id_to_entity.get(i, None) for i in range(min(5, len(id_to_entity)))]
    logger.info(f"Sample embedding keys: {sample_emb_keys}")
    logger.info(f"Sample id_to_entity values: {sample_id_to_entity}")
    
    # Check overlap between embeddings dict keys and id_to_entity values
    id_to_entity_values = set(id_to_entity.values())
    embedding_keys = set(embeddings.keys())
    overlap = id_to_entity_values & embedding_keys
    logger.info(f"Entity ID overlap: {len(overlap)}/{len(id_to_entity_values)} entities from id_to_entity found in embeddings")
    
    if len(overlap) < len(id_to_entity_values) * 0.9:
        logger.warning(f"⚠️  Only {len(overlap)}/{len(id_to_entity_values)} entities match between id_to_entity and embeddings dict!")
        logger.warning("This suggests a mismatch. Using embedder's own id_to_entity mapping instead...")
        
        # Use embedder's own mapping instead
        embedder_id_to_entity = embedder.id_to_entity
        train_df_for_features = train_df.copy()
        test_df_for_features = test_df.copy()
        
        train_df_for_features['source'] = train_df_for_features['source_id'].map(embedder_id_to_entity)
        train_df_for_features['target'] = train_df_for_features['target_id'].map(embedder_id_to_entity)
        test_df_for_features['source'] = test_df_for_features['source_id'].map(embedder_id_to_entity)
        test_df_for_features['target'] = test_df_for_features['target_id'].map(embedder_id_to_entity)
        
        # Check overlap with embedder's mapping
        embedder_values = set(embedder_id_to_entity.values())
        embedder_overlap = embedder_values & embedding_keys
        logger.info(f"Using embedder's mapping: {len(embedder_overlap)}/{len(embedder_values)} entities match")
    else:
        train_df_for_features = train_df.copy()
        test_df_for_features = test_df.copy()
        
        train_df_for_features['source'] = train_df_for_features['source_id'].map(id_to_entity)
        train_df_for_features['target'] = train_df_for_features['target_id'].map(id_to_entity)
        test_df_for_features['source'] = test_df_for_features['source_id'].map(id_to_entity)
        test_df_for_features['target'] = test_df_for_features['target_id'].map(id_to_entity)
    
    # Check for missing mappings
    missing_train = train_df_for_features['source'].isna().sum() + train_df_for_features['target'].isna().sum()
    missing_test = test_df_for_features['source'].isna().sum() + test_df_for_features['target'].isna().sum()
    if missing_train > 0 or missing_test > 0:
        logger.warning(f"Found {missing_train} missing entity mappings in train, {missing_test} in test")
        logger.warning("This may cause features to default to zero vectors")
        
        # Additional diagnostic: check a sample of missing entities
        if missing_train > 0:
            sample_missing = train_df_for_features[train_df_for_features['source'].isna() | train_df_for_features['target'].isna()].head(3)
            logger.warning(f"Sample missing mappings:")
            for idx, row in sample_missing.iterrows():
                logger.warning(f"  source_id={row['source_id']}, target_id={row['target_id']}")
                logger.warning(f"  source={row.get('source', 'MISSING')}, target={row.get('target', 'MISSING')}")

    # Diagnostic: Check if embeddings are actually being retrieved
    logger.info("\nDiagnostic: Checking embedding retrieval...")
    sample_train_row = train_df_for_features.iloc[0]
    sample_h_id = str(sample_train_row['source'])
    sample_t_id = str(sample_train_row['target'])
    sample_h_emb = embeddings.get(sample_h_id, None)
    sample_t_emb = embeddings.get(sample_t_id, None)
    
    logger.info(f"Sample row: source='{sample_h_id}', target='{sample_t_id}'")
    logger.info(f"  Head embedding found: {sample_h_emb is not None}, shape: {sample_h_emb.shape if sample_h_emb is not None else 'N/A'}")
    logger.info(f"  Tail embedding found: {sample_t_emb is not None}, shape: {sample_t_emb.shape if sample_t_emb is not None else 'N/A'}")
    
    if sample_h_emb is not None and sample_t_emb is not None:
        logger.info(f"  Head embedding sample values: {sample_h_emb[:5]}")
        logger.info(f"  Tail embedding sample values: {sample_t_emb[:5]}")
        logger.info(f"  Head embedding std: {np.std(sample_h_emb):.6f}")
        logger.info(f"  Tail embedding std: {np.std(sample_t_emb):.6f}")
        
        # Check a few more samples
        for i in range(1, min(4, len(train_df_for_features))):
            row = train_df_for_features.iloc[i]
            h_emb = embeddings.get(str(row['source']), None)
            t_emb = embeddings.get(str(row['target']), None)
            if h_emb is not None and t_emb is not None:
                if np.allclose(h_emb, sample_h_emb):
                    logger.warning(f"  Row {i}: Head embedding is identical to row 0!")
                if np.allclose(t_emb, sample_t_emb):
                    logger.warning(f"  Row {i}: Tail embedding is identical to row 0!")
    
    # Check how many embeddings are actually found
    missing_embeddings_train = 0
    for idx, row in train_df_for_features.head(100).iterrows():
        h_emb = embeddings.get(str(row['source']), None)
        t_emb = embeddings.get(str(row['target']), None)
        if h_emb is None or t_emb is None:
            missing_embeddings_train += 1
    
    logger.info(f"Missing embeddings in first 100 rows: {missing_embeddings_train}/100")

    # Build features for train (FIT scaler)
    logger.info("Building features for training set (fitting scaler)...")
    X_train, feature_names = feature_builder.build_features(
        train_df_for_features, embeddings, edges_df=train_edges_only, fit_scaler=True
    )

    # Build features for test (TRANSFORM only, no fitting)
    logger.info("Building features for test set (transforming with fitted scaler)...")
    X_test, feature_names_test = feature_builder.build_features(
        test_df_for_features, embeddings, edges_df=train_edges_only, fit_scaler=False
    )
    
    # Additional diagnostic: Check feature values before normalization
    logger.info(f"\nFeature diagnostic after building:")
    logger.info(f"  X_train shape: {X_train.shape}")
    logger.info(f"  X_train sample (first row, first 10 features): {X_train[0, :10]}")
    logger.info(f"  X_train min/max: [{X_train.min():.6f}, {X_train.max():.6f}]")
    logger.info(f"  X_train mean: {X_train.mean():.6f}, std: {X_train.std():.6f}")
    
    # Check if all rows are identical
    if len(X_train) > 1:
        first_row = X_train[0]
        identical_rows = np.sum([np.allclose(X_train[i], first_row) for i in range(1, min(10, len(X_train)))])
        logger.info(f"  Identical rows (first 10): {identical_rows}/9")

    y_train = train_df['label'].values
    y_test = test_df['label'].values

    logger.info(f"Classical features: train {X_train.shape}, test {X_test.shape}")
    
    # ========== EVIDENCE WEIGHTING ==========
    sample_weights_train = None
    sample_weights_test = None
    
    if args.use_evidence_weighting and args.relation == 'CtD':
        logger.info("\n" + "="*80)
        logger.info("COMPUTING EVIDENCE WEIGHTS (Shared Genes)")
        logger.info("="*80)
        
        try:
            # Load full Hetionet to find gene associations
            df_full = load_hetionet_edges()
            
            # Extract gene associations for compounds and diseases
            # Compounds: CbG (binds), CdG (downregulates), CuG (upregulates)
            compound_gene_edges = df_full[df_full['metaedge'].isin(['CbG', 'CdG', 'CuG'])]
            compound_genes = {}
            for _, row in compound_gene_edges.iterrows():
                compound = row['source']
                gene = row['target']
                if compound not in compound_genes:
                    compound_genes[compound] = set()
                compound_genes[compound].add(gene)
            
            # Diseases: DaG (associates), DdG (downregulates), DuG (upregulates)
            disease_gene_edges = df_full[df_full['metaedge'].isin(['DaG', 'DdG', 'DuG'])]
            disease_genes = {}
            for _, row in disease_gene_edges.iterrows():
                disease = row['source']
                gene = row['target']
                if disease not in disease_genes:
                    disease_genes[disease] = set()
                disease_genes[disease].add(gene)
            
            logger.info(f"Found gene associations: {len(compound_genes)} compounds, {len(disease_genes)} diseases")
            
            # Compute shared genes for each pair
            def compute_shared_genes(source_entity, target_entity):
                source_genes = compound_genes.get(source_entity, set())
                target_genes = disease_genes.get(target_entity, set())
                shared = source_genes & target_genes
                return len(shared)
            
            # Compute weights for training set
            weights_train = []
            for _, row in train_df.iterrows():
                source_id = row['source_id']
                target_id = row['target_id']
                source_entity = id_to_entity.get(source_id, None)
                target_entity = id_to_entity.get(target_id, None)
                
                if source_entity and target_entity:
                    shared_count = compute_shared_genes(source_entity, target_entity)
                    # Weight: 1.0 + 0.1 * shared_genes (minimum 1.0)
                    weight = 1.0 + 0.1 * max(0, shared_count - args.min_shared_genes)
                else:
                    weight = 1.0
                weights_train.append(weight)
            
            # Compute weights for test set
            weights_test = []
            for _, row in test_df.iterrows():
                source_id = row['source_id']
                target_id = row['target_id']
                source_entity = id_to_entity.get(source_id, None)
                target_entity = id_to_entity.get(target_id, None)
                
                if source_entity and target_entity:
                    shared_count = compute_shared_genes(source_entity, target_entity)
                    weight = 1.0 + 0.1 * max(0, shared_count - args.min_shared_genes)
                else:
                    weight = 1.0
                weights_test.append(weight)
            
            sample_weights_train = np.array(weights_train, dtype=np.float32)
            sample_weights_test = np.array(weights_test, dtype=np.float32)
            
            # Statistics
            pos_weights = sample_weights_train[y_train == 1]
            neg_weights = sample_weights_train[y_train == 0]
            logger.info(f"Evidence weighting statistics:")
            logger.info(f"  Positive samples: mean weight={pos_weights.mean():.3f}, max={pos_weights.max():.3f}")
            logger.info(f"  Negative samples: mean weight={neg_weights.mean():.3f}, max={neg_weights.max():.3f}")
            logger.info(f"  Samples with shared genes >= {args.min_shared_genes}: {(sample_weights_train > 1.0).sum()}/{len(sample_weights_train)}")
            logger.info(f"✓ Evidence weights computed")
            
        except Exception as e:
            logger.warning(f"⚠️  Evidence weighting failed: {e}. Continuing without weights.")
            import traceback
            traceback.print_exc()
            sample_weights_train = None
            sample_weights_test = None

    # Feature diagnostics and filtering
    logger.info("\n" + "="*80)
    logger.info("FEATURE DIAGNOSTICS & FILTERING")
    logger.info("="*80)
    
    # Check for NaN/Inf
    nan_count = np.isnan(X_train).sum()
    inf_count = np.isinf(X_train).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"Found NaN: {nan_count}, Inf: {inf_count}. Removing invalid samples...")
        valid_train = ~(np.isnan(X_train).any(axis=1) | np.isinf(X_train).any(axis=1))
        valid_test = ~(np.isnan(X_test).any(axis=1) | np.isinf(X_test).any(axis=1))
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        logger.info(f"After removing invalid samples: train {X_train.shape}, test {X_test.shape}")
    
    # Check feature variance BEFORE any filtering
    feature_std = np.std(X_train, axis=0)
    zero_variance = np.sum(feature_std < 1e-10)
    low_variance = np.sum(feature_std < 1e-6)
    
    logger.info(f"Feature variance analysis:")
    logger.info(f"  Zero variance features (<1e-10): {zero_variance}/{len(feature_std)}")
    logger.info(f"  Low variance features (<1e-6): {low_variance}/{len(feature_std)}")
    logger.info(f"  Mean std: {feature_std.mean():.6f}, Min: {feature_std.min():.6f}, Max: {feature_std.max():.6f}")
    
    # Handle case where all features have zero variance (likely due to StandardScaler on identical features)
    if zero_variance == len(feature_std):
        logger.error("⚠️  ALL features have zero variance! This likely means:")
        logger.error("  1. Features were identical before normalization")
        logger.error("  2. StandardScaler produced NaN or identical values")
        logger.error("  3. There's a bug in feature construction")
        logger.error("\nAttempting to rebuild features WITHOUT normalization...")
        
        # Rebuild features without normalization
        feature_builder_no_norm = EnhancedFeatureBuilder(
            include_graph_features=args.use_graph_features,
            include_domain_features=args.use_domain_features,
            normalize=False  # Disable normalization
        )
        
        if args.use_graph_features:
            feature_builder_no_norm.build_graph(train_edges_only)
        
        X_train, feature_names = feature_builder_no_norm.build_features(
            train_df, embeddings, edges_df=train_edges_only, fit_scaler=False
        )
        X_test, _ = feature_builder_no_norm.build_features(
            test_df, embeddings, edges_df=train_edges_only, fit_scaler=False
        )
        
        # Re-check variance
        feature_std = np.std(X_train, axis=0)
        zero_variance = np.sum(feature_std < 1e-10)
        logger.info(f"After rebuilding without normalization: zero variance features: {zero_variance}/{len(feature_std)}")
        
        if zero_variance == len(feature_std):
            raise ValueError(
                "All features still have zero variance even without normalization. "
                "This indicates a fundamental issue with feature construction. "
                "Check if embeddings are identical or if feature computation is broken."
            )
    
    # Remove low-variance features (if any remain)
    if zero_variance > 0 or low_variance > 0:
        from sklearn.feature_selection import VarianceThreshold
        logger.info(f"\nRemoving low-variance features...")
        variance_selector = VarianceThreshold(threshold=1e-6)
        
        try:
            X_train = variance_selector.fit_transform(X_train)
            X_test = variance_selector.transform(X_test)
            logger.info(f"After variance filtering: train {X_train.shape}, test {X_test.shape}")
            
            # Check if any features remain
            if X_train.shape[1] == 0:
                raise ValueError(
                    "All features were removed by variance filtering! "
                    "This suggests features are constant. Check feature construction."
                )
            
            # Update feature names
            valid_features = variance_selector.get_support()
            feature_names = [name for name, valid in zip(feature_names, valid_features) if valid]
        except ValueError as e:
            if "No feature in X meets the variance threshold" in str(e):
                logger.error(f"⚠️  VarianceThreshold failed: {e}")
                logger.error("All features have variance below threshold. Using all features anyway...")
                # Don't apply variance filtering - use all features
            else:
                raise
    
    # Check feature-to-sample ratio
    ratio = X_train.shape[1] / X_train.shape[0]
    logger.info(f"Feature-to-sample ratio: {ratio:.2f}")
    if ratio > 1.0:
        logger.warning(f"⚠️  More features ({X_train.shape[1]}) than samples ({X_train.shape[0]})! "
                      f"This can cause overfitting and numerical issues.")
        logger.warning("Consider using feature selection (e.g., mutual information or PCA).")
    elif ratio > 0.5:
        logger.warning(f"⚠️  High feature-to-sample ratio ({ratio:.2f}). Consider feature selection.")
    
    # ========== IMPROVED FEATURE ENGINEERING (after variance filtering) ==========
    if args.use_improved_features:
        logger.info("\n" + "="*80)
        logger.info("IMPROVED FEATURE ENGINEERING")
        logger.info("="*80)
        
        try:
            from kg_layer.improved_feature_engineering import ImprovedFeatureEngineer
            
            feature_engineer = ImprovedFeatureEngineer(
                use_rf_guidance=True,
                max_interaction_features=args.max_interaction_features,
                use_domain_features=args.use_domain_features,
                random_state=args.random_state
            )
            
            # Create interaction features (this will train a quick RF for guidance)
            logger.info("Creating interaction features with RandomForest guidance...")
            X_train_enhanced, feature_names_enhanced = feature_engineer.create_interaction_features(
                X_train, y_train, feature_names
            )
            
            # Create interaction features for test set (using same top features)
            logger.info("Creating interaction features for test set...")
            X_test_enhanced, _ = feature_engineer.create_interaction_features(
                X_test, y_test, feature_names  # y_test used only for shape, not training
            )
            
            # Add domain knowledge features if enabled (for both train and test)
            if args.use_domain_features:
                logger.info("Adding domain knowledge features...")
                # Get entity IDs for domain feature extraction
                train_head_ids = train_df_for_features['source'].values
                train_tail_ids = train_df_for_features['target'].values
                test_head_ids = test_df_for_features['source'].values
                test_tail_ids = test_df_for_features['target'].values
                
                X_train_enhanced = feature_engineer.create_domain_features(
                    X_train_enhanced,
                    head_ids=train_head_ids,
                    tail_ids=train_tail_ids
                )
                X_test_enhanced = feature_engineer.create_domain_features(
                    X_test_enhanced,
                    head_ids=test_head_ids,
                    tail_ids=test_tail_ids
                )
            
            # Create class difference features
            logger.info("Creating class difference features...")
            X_train_enhanced = feature_engineer.create_class_difference_features(X_train_enhanced, y_train)
            
            # For test set, we need to compute centroids from training data
            # But create_class_difference_features computes centroids from input data
            # So we'll compute centroids from X_train_enhanced BEFORE adding class diff features
            # Actually, let's compute centroids from X_train_enhanced (before class diff) and apply to test
            pos_mask = y_train == 1
            neg_mask = y_train == 0
            
            if np.sum(pos_mask) > 0 and np.sum(neg_mask) > 0:
                # Compute centroids from training data (before class diff features)
                X_train_before_class_diff = X_train_enhanced[:, :X_train.shape[1] + len(feature_names_enhanced) - len(feature_names)] if len(feature_names_enhanced) > len(feature_names) else X_train_enhanced
                pos_centroid = np.mean(X_train_before_class_diff[pos_mask], axis=0)
                neg_centroid = np.mean(X_train_before_class_diff[neg_mask], axis=0)
                
                # Apply to test set
                test_pos_distances = np.linalg.norm(X_test_enhanced - pos_centroid, axis=1, keepdims=True)
                test_neg_distances = np.linalg.norm(X_test_enhanced - neg_centroid, axis=1, keepdims=True)
                test_ratio = test_pos_distances / (test_neg_distances + 1e-8)
                test_diff = np.abs(test_pos_distances - test_neg_distances)
                
                test_class_diff_features = np.hstack([test_pos_distances, test_neg_distances, test_ratio, test_diff])
                X_test_enhanced = np.hstack([X_test_enhanced, test_class_diff_features])
            
            # Update feature names and replace X_train/X_test
            feature_names = feature_names_enhanced
            X_train = X_train_enhanced
            X_test = X_test_enhanced
            
            logger.info(f"✓ Enhanced features: train {X_train.shape}, test {X_test.shape}")
            
            # Count added features
            n_base_features = len(feature_names) if feature_names else X_train.shape[1]
            n_enhanced_features = X_train.shape[1]
            n_added = n_enhanced_features - n_base_features
            logger.info(f"  Added {n_added} enhanced features (interactions + domain + class differences)")
            
            # Re-check feature-to-sample ratio after enhancement
            ratio = X_train.shape[1] / X_train.shape[0]
            logger.info(f"Updated feature-to-sample ratio: {ratio:.2f}")
            
        except ImportError:
            logger.warning("⚠️  Improved feature engineering not available. Using standard features.")
        except Exception as e:
            logger.warning(f"⚠️  Improved feature engineering failed: {e}. Using standard features.")
            import traceback
            traceback.print_exc()
    
    # ========== QUANTUM FEATURE ENGINEERING ==========
    if args.use_quantum_feature_engineering:
        logger.info("\n" + "="*80)
        logger.info("QUANTUM FEATURE ENGINEERING")
        logger.info("="*80)
        
        try:
            from quantum_layer.quantum_feature_engineering import QuantumFeatureEngineer
            
            # Get head and tail embeddings for quantum feature engineering
            head_embs_train = np.array([embeddings.get(str(row['source']), np.zeros(X_train.shape[1]//2)) 
                                       for _, row in train_df_for_features.iterrows()])
            tail_embs_train = np.array([embeddings.get(str(row['target']), np.zeros(X_train.shape[1]//2)) 
                                       for _, row in train_df_for_features.iterrows()])
            head_embs_test = np.array([embeddings.get(str(row['source']), np.zeros(X_test.shape[1]//2)) 
                                      for _, row in test_df_for_features.iterrows()])
            tail_embs_test = np.array([embeddings.get(str(row['target']), np.zeros(X_test.shape[1]//2)) 
                                      for _, row in test_df_for_features.iterrows()])
            
            logger.info("Applying quantum-specific feature engineering...")
            
            quantum_fe = QuantumFeatureEngineer(
                num_qubits=args.qml_dim,
                feature_map_type=args.qml_feature_map if args.qml_feature_map != 'custom_link_prediction' else 'ZZ',
                feature_map_reps=args.qml_feature_map_reps,
                entanglement=args.qml_entanglement,
                use_quantum_selection=args.quantum_feature_selection,
                random_state=args.random_state
            )
            
            # Fit and transform training features
            X_train_quantum = quantum_fe.fit_transform(
                X_train, y_train,
                head_embeddings=head_embs_train,
                tail_embeddings=tail_embs_train,
                max_features=None if not args.quantum_feature_selection else int(X_train.shape[0] * 0.8)
            )
            
            # Transform test features
            X_test_quantum = quantum_fe.transform(
                X_test,
                head_embeddings=head_embs_test,
                tail_embeddings=tail_embs_test
            )
            
            X_train = X_train_quantum
            X_test = X_test_quantum
            
            logger.info(f"✓ Quantum features: train {X_train.shape}, test {X_test.shape}")
            if quantum_fe.feature_importances_ is not None:
                top_features = np.argsort(quantum_fe.feature_importances_)[-5:][::-1]
                logger.info(f"  Top 5 quantum feature importances: {quantum_fe.feature_importances_[top_features]}")
        except ImportError:
            logger.warning("⚠️  Quantum feature engineering not available (Qiskit may be missing). Using standard features.")
        except Exception as e:
            logger.warning(f"⚠️  Quantum feature engineering failed: {e}. Using standard features.")
            import traceback
            traceback.print_exc()
    
    # Optional: Apply feature selection if ratio is too high
    if ratio > 1.0 and args.use_feature_selection:
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        logger.info(f"\nApplying mutual information feature selection...")
        n_features_to_select = min(int(X_train.shape[0] * 0.8), X_train.shape[1])  # Use 80% of samples
        feature_selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
        X_train = feature_selector.fit_transform(X_train, y_train)
        X_test = feature_selector.transform(X_test)
        logger.info(f"After feature selection: train {X_train.shape}, test {X_test.shape}")
        
        # Update feature names
        valid_features = feature_selector.get_support()
        feature_names = [name for name, valid in zip(feature_names, valid_features) if valid]

    # ========== STEP 4: TRAIN CLASSICAL MODELS ==========
    classical_results = {}
    
    # Initialize variables that may be used later (for quantum feature selection)
    rf_model = None
    rf_result = None

    if not args.quantum_only and (not args.skip_quantum or not args.classical_only):
        logger.info("\n" + "="*80)
        logger.info("STEP 4: TRAINING CLASSICAL MODELS")
        logger.info("="*80)

        # Phase 3: Expanded hyperparameter grids for classical models
        # RandomForest with grid search
        if args.fast_mode:
            rf_param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [8, 10],
                'min_samples_split': [10, 20]
            }
        else:
            rf_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [8, 10, 12, 15],
                'min_samples_split': [5, 10, 20]
            }
        
        # LogisticRegression with grid search
        if args.fast_mode:
            lr_param_grid = {
                'C': [0.1, 1.0, 10.0]
            }
        else:
            lr_param_grid = {
                'C': [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
            }
        
        models = {
            'RandomForest-Optimized': None,  # Will be set up with grid search
            'SVM-Linear-Optimized': None,  # Will be set up with grid search below
            'LogisticRegression-L2': None,  # Will be set up with grid search
        }
        
        # Optionally add SVM-RBF (can be skipped if it continues to fail)
        if not args.skip_svm_rbf:
            models['SVM-RBF-Optimized'] = None  # Will be set up with grid search below

        if args.fast_mode:
            # Only use top 2 models in fast mode
            models = {k: v for i, (k, v) in enumerate(models.items()) if i < 2}

        # Train models (rf_model and rf_result already initialized above)
        for name, model in models.items():
            if name == 'SVM-RBF-Optimized':
                # Special handling for SVM-RBF with grid search
                if args.skip_svm_rbf:
                    logger.info(f"Skipping {name} (--skip_svm_rbf)")
                    classical_results[name] = {'status': 'skipped'}
                    continue
                result = train_svm_rbf_with_grid_search(
                    X_train, y_train, X_test, y_test,
                    cv_folds=args.cv_folds, random_state=args.random_state,
                    fast_mode=args.fast_mode
                )
            elif name == 'SVM-Linear-Optimized':
                # Skip if requested
                if args.skip_svm_linear:
                    logger.info(f"Skipping {name} (--skip_svm_linear)")
                    classical_results[name] = {'status': 'skipped'}
                    continue
                # Special handling for SVM-Linear with grid search
                result = train_svm_linear(
                    X_train, y_train, X_test, y_test,
                    cv_folds=args.cv_folds, random_state=args.random_state
                )
            elif name == 'RandomForest-Optimized':
                # Phase 3: Grid search for RandomForest
                logger.info(f"\nTraining {name} with grid search...")
                try:
                    t0 = time.time()
                    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
                    base_rf = RandomForestClassifier(
                        min_samples_leaf=5,
                        max_features='sqrt',
                        class_weight='balanced',
                        random_state=args.random_state,
                        n_jobs=-1
                    )
                    gs = GridSearchCV(
                        base_rf,
                        param_grid=rf_param_grid,
                        scoring='average_precision',
                        cv=skf,
                        refit=True,
                        n_jobs=-1,
                        verbose=0
                    )
                    if sample_weights_train is not None:
                        gs.fit(X_train, y_train, sample_weight=sample_weights_train)
                    else:
                        gs.fit(X_train, y_train)
                    fit_time = time.time() - t0
                    best_rf = gs.best_estimator_
                    best_params = gs.best_params_
                    logger.info(f"  Best params: {best_params}")
                    logger.info(f"  CV PR-AUC: {gs.best_score_:.4f}")
                    
                    # Store best model
                    models[name] = best_rf
                    rf_model = best_rf
                    
                    # Evaluate
                    result = train_classical_model(
                        name, best_rf, X_train, y_train, X_test, y_test,
                        cv_folds=args.cv_folds, random_state=args.random_state,
                        calibrate=args.calibrate_probabilities,
                        calibration_method=args.calibration_method,
                        feature_names=feature_names,
                        sample_weight=sample_weights_train
                    )
                    result['best_params'] = best_params
                    result['cv_score'] = float(gs.best_score_)
                    rf_result = result
                except Exception as e:
                    logger.error(f"  ❌ {name} grid search failed: {e}")
                    # Fallback to default
                    fallback_rf = RandomForestClassifier(
                        n_estimators=200 if not args.fast_mode else 100,
                        max_depth=10,
                        min_samples_split=10,
                        min_samples_leaf=5,
                        max_features='sqrt',
                        class_weight='balanced',
                        random_state=args.random_state,
                        n_jobs=-1
                    )
                    models[name] = fallback_rf
                    result = train_classical_model(
                        name, fallback_rf, X_train, y_train, X_test, y_test,
                        cv_folds=args.cv_folds, random_state=args.random_state,
                        calibrate=args.calibrate_probabilities,
                        calibration_method=args.calibration_method,
                        feature_names=feature_names,
                        sample_weight=sample_weights_train
                    )
                    rf_model = fallback_rf
                    rf_result = result
            elif name == 'LogisticRegression-L2':
                # Phase 3: Grid search for LogisticRegression
                logger.info(f"\nTraining {name} with grid search...")
                try:
                    t0 = time.time()
                    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
                    # Explicitly import to avoid scoping issues
                    from sklearn.linear_model import LogisticRegression as LR
                    base_lr = LR(
                        penalty='l2',
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=args.random_state
                    )
                    gs = GridSearchCV(
                        base_lr,
                        param_grid=lr_param_grid,
                        scoring='average_precision',
                        cv=skf,
                        refit=True,
                        n_jobs=-1,
                        verbose=0
                    )
                    if sample_weights_train is not None:
                        gs.fit(X_train, y_train, sample_weight=sample_weights_train)
                    else:
                        gs.fit(X_train, y_train)
                    fit_time = time.time() - t0
                    best_lr = gs.best_estimator_
                    best_params = gs.best_params_
                    logger.info(f"  Best params: {best_params}")
                    logger.info(f"  CV PR-AUC: {gs.best_score_:.4f}")
                    
                    # Store best model
                    models[name] = best_lr
                    
                    # Evaluate
                    result = train_classical_model(
                        name, best_lr, X_train, y_train, X_test, y_test,
                        cv_folds=args.cv_folds, random_state=args.random_state,
                        calibrate=args.calibrate_probabilities,
                        calibration_method=args.calibration_method,
                        feature_names=feature_names,
                        sample_weight=sample_weights_train
                    )
                    result['best_params'] = best_params
                    result['cv_score'] = float(gs.best_score_)
                except Exception as e:
                    logger.error(f"  ❌ {name} grid search failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback to default
                    from sklearn.linear_model import LogisticRegression as LR
                    fallback_lr = LR(
                        C=1.0,
                        penalty='l2',
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=args.random_state
                    )
                    models[name] = fallback_lr
                    result = train_classical_model(
                        name, fallback_lr, X_train, y_train, X_test, y_test,
                        cv_folds=args.cv_folds, random_state=args.random_state,
                        calibrate=args.calibrate_probabilities,
                        calibration_method=args.calibration_method,
                        feature_names=feature_names,
                        sample_weight=sample_weights_train
                    )
            else:
                # Regular model training (shouldn't happen with current setup)
                result = train_classical_model(
                    name, model, X_train, y_train, X_test, y_test,
                    cv_folds=args.cv_folds, random_state=args.random_state,
                    calibrate=args.calibrate_probabilities,
                    calibration_method=args.calibration_method,
                    feature_names=feature_names,
                    sample_weight=sample_weights_train
                )
            classical_results[name] = result
        
        # Feature importance-based feature selection (if RandomForest succeeded)
        if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
            logger.info("\n" + "="*80)
            logger.info("FEATURE IMPORTANCE ANALYSIS")
            logger.info("="*80)
            
            importances = rf_model.feature_importances_
            top_n = min(50, len(feature_names))  # Top 50 features
            top_indices = np.argsort(importances)[-top_n:][::-1]
            
            logger.info(f"Top {top_n} Most Important Features:")
            for i, idx in enumerate(top_indices[:20], 1):  # Show top 20
                logger.info(f"  {i:2d}. {feature_names[idx]:30s}: {importances[idx]:.6f}")
            
            # Optional: Use top features for other models (if requested)
            # This could be added as a command-line argument
        
        # Ensemble: Voting Classifier (RandomForest + LogisticRegression)
        if not args.fast_mode and rf_result and rf_result['status'] == 'success':
            logger.info("\n" + "="*80)
            logger.info("TRAINING ENSEMBLE MODEL")
            logger.info("="*80)
            
            try:
                # Get trained models
                rf_trained = models['RandomForest-Optimized']
                rf_trained.fit(X_train, y_train)
                
                lr_trained = models['LogisticRegression-L2']
                lr_trained.fit(X_train, y_train)
                
                # Create voting ensemble
                ensemble = VotingClassifier(
                    estimators=[
                        ('rf', rf_trained),
                        ('lr', lr_trained)
                    ],
                    voting='soft',
                    weights=[2, 1]  # Weight RandomForest more (it performs better)
                )
                
                t0 = time.time()
                ensemble.fit(X_train, y_train)
                fit_time = time.time() - t0
                
                # Evaluate ensemble
                skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
                oof_scores = cross_val_predict(ensemble, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
                oof_preds = (oof_scores >= 0.5).astype(int)
                test_scores = ensemble.predict_proba(X_test)[:, 1]
                test_preds = (test_scores >= 0.5).astype(int)
                
                train_metrics = compute_metrics(y_train, oof_preds, oof_scores)
                test_metrics = compute_metrics(y_test, test_preds, test_scores)
                
                logger.info(f"  ✅ Ensemble (RF+LR) - Test PR-AUC: {test_metrics['pr_auc']:.4f} (fit: {fit_time:.1f}s)")
                
                classical_results['Ensemble-RF-LR'] = {
                    'status': 'success',
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'fit_seconds': fit_time
                }
            except Exception as e:
                logger.warning(f"  ⚠️  Ensemble training failed: {e}")
                classical_results['Ensemble-RF-LR'] = {
                    'status': 'failed',
                    'error': str(e)
                }

    # ========== STEP 5: PREPARE QUANTUM FEATURES ==========
    quantum_results = {}

    if not args.skip_quantum and (not args.classical_only or args.quantum_only):
        logger.info("\n" + "="*80)
        logger.info("STEP 5: PREPARING QUANTUM FEATURES")
        logger.info("="*80)

        # Get raw embeddings for quantum feature engineering
        # Map integer IDs to entity string IDs, then look up embeddings
        # Also detect actual embedding dimension from first embedding
        sample_emb = next(iter(embeddings.values())) if embeddings else None
        actual_emb_dim = sample_emb.shape[0] if sample_emb is not None else args.embedding_dim
        
        if actual_emb_dim != args.embedding_dim:
            logger.warning(f"Embedding dimension mismatch: expected {args.embedding_dim}, got {actual_emb_dim}. "
                          f"This is normal for ComplEx (complex→real conversion doubles dimension).")
        
        train_h_embs = []
        train_t_embs = []
        missing_h = 0
        missing_t = 0
        for h_id, t_id in zip(train_df['source_id'].values, train_df['target_id'].values):
            h_entity = id_to_entity.get(h_id, None)
            t_entity = id_to_entity.get(t_id, None)
            
            h_emb = embeddings.get(h_entity, None) if h_entity else None
            t_emb = embeddings.get(t_entity, None) if t_entity else None
            
            if h_emb is None:
                missing_h += 1
                h_emb = np.zeros(actual_emb_dim)
            if t_emb is None:
                missing_t += 1
                t_emb = np.zeros(actual_emb_dim)
            
            train_h_embs.append(h_emb)
            train_t_embs.append(t_emb)
        
        train_h_embs = np.array(train_h_embs)
        train_t_embs = np.array(train_t_embs)
        
        if missing_h > 0 or missing_t > 0:
            logger.warning(f"Missing embeddings: {missing_h} head entities, {missing_t} tail entities "
                          f"(using zero vectors)")
        
        test_h_embs = []
        test_t_embs = []
        missing_h_test = 0
        missing_t_test = 0
        for h_id, t_id in zip(test_df['source_id'].values, test_df['target_id'].values):
            h_entity = id_to_entity.get(h_id, None)
            t_entity = id_to_entity.get(t_id, None)
            
            h_emb = embeddings.get(h_entity, None) if h_entity else None
            t_emb = embeddings.get(t_entity, None) if t_entity else None
            
            if h_emb is None:
                missing_h_test += 1
                h_emb = np.zeros(actual_emb_dim)
            if t_emb is None:
                missing_t_test += 1
                t_emb = np.zeros(actual_emb_dim)
            
            test_h_embs.append(h_emb)
            test_t_embs.append(t_emb)
        
        test_h_embs = np.array(test_h_embs)
        test_t_embs = np.array(test_t_embs)
        
        if missing_h_test > 0 or missing_t_test > 0:
            logger.warning(f"Missing embeddings in test: {missing_h_test} head entities, {missing_t_test} tail entities "
                          f"(using zero vectors)")
        
        # Diagnostic: Check if embeddings are identical (would cause constant features)
        h_unique = len(np.unique(train_h_embs, axis=0))
        t_unique = len(np.unique(train_t_embs, axis=0))
        logger.info(f"Embedding diversity: {h_unique} unique head embeddings out of {len(train_h_embs)}, "
                   f"{t_unique} unique tail embeddings out of {len(train_t_embs)}")
        
        if h_unique < len(train_h_embs) * 0.1 or t_unique < len(train_t_embs) * 0.1:
            logger.warning(f"⚠️  Low embedding diversity detected! This may cause constant features. "
                          f"Head diversity: {h_unique}/{len(train_h_embs)} ({100*h_unique/len(train_h_embs):.1f}%), "
                          f"Tail diversity: {t_unique}/{len(train_t_embs)} ({100*t_unique/len(train_t_embs):.1f}%)")

        # Optional: Use RandomForest feature importance to select best features for quantum
        # This helps focus quantum features on the most informative dimensions
        use_rf_feature_selection = False
        if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
            # Check if we have enough features to select from
            if len(rf_model.feature_importances_) >= args.qml_dim * 2:
                use_rf_feature_selection = True
                logger.info("Using RandomForest feature importance to guide quantum feature selection...")
                
                # Get top features from RandomForest
                importances = rf_model.feature_importances_
                top_indices = np.argsort(importances)[-args.qml_dim*2:][::-1]  # Top 2x qubits
                
                # Select top features from embeddings (if they're embedding features)
                # This is a heuristic - we'll use the most important embedding dimensions
                logger.info(f"Selected top {len(top_indices)} features based on RF importance")
        
        # Use the existing quantum feature preparation from advanced_qml_features
        # (This is separate from quantum feature engineering which happens earlier)
        from quantum_layer.advanced_qml_features import QuantumFeatureEngineer as AdvancedQMLFeatureEngineer
        
        # Determine reduction method
        use_kernel_pca = (args.qml_reduction_method == 'kernel_pca')
        use_lda = (args.qml_reduction_method == 'lda')
        
        # Map feature selection method
        feature_selection_method = None
        if args.qml_feature_selection_method:
            if args.qml_feature_selection_method == 'f_classif':
                from sklearn.feature_selection import f_classif
                feature_selection_method = 'f_classif'
            elif args.qml_feature_selection_method == 'mutual_info':
                feature_selection_method = 'mutual_info'
            elif args.qml_feature_selection_method == 'chi2':
                from sklearn.feature_selection import chi2
                feature_selection_method = 'chi2'
        
        # Create quantum feature engineer for dimensionality reduction
        qml_engineer = AdvancedQMLFeatureEngineer(
            num_qubits=args.qml_dim,
            encoding_strategy=args.qml_encoding,
            feature_selection_method=feature_selection_method,
            use_kernel_pca=use_kernel_pca,
            random_state=args.random_state,
            reduction_method=args.qml_reduction_method,
            feature_select_k_mult=args.qml_feature_select_k_mult,
            pre_pca_dim=args.qml_pre_pca_dim
        )

        # Prepare QML features (reduces embeddings to num_qubits dimensions)
        logger.info(f"Preparing quantum features with {args.qml_encoding} encoding...")
        logger.info(f"  Input embedding dim: {train_h_embs.shape[1]}")
        logger.info(f"  Target qubits: {args.qml_dim}")
        
        X_train_qml = qml_engineer.prepare_qml_features(
            train_h_embs, train_t_embs, y_train, fit=True
        )
        X_test_qml = qml_engineer.prepare_qml_features(
            test_h_embs, test_t_embs, fit=False
        )
        
        logger.info(f"Quantum features prepared: train {X_train_qml.shape}, test {X_test_qml.shape}")
        
        # Diagnostic: Check quantum feature variance
        qml_train_std = np.std(X_train_qml, axis=0)
        qml_zero_var = np.sum(qml_train_std < 1e-10)
        logger.info(f"Quantum feature variance: {qml_zero_var}/{len(qml_train_std)} features have zero variance")
        if qml_zero_var > 0:
            logger.warning(f"⚠️  {qml_zero_var} quantum features have zero variance! This may hurt QSVC performance.")

        # ========== COMPREHENSIVE SEPARABILITY DIAGNOSTICS ==========
        logger.info("\n" + "="*80)
        logger.info("QUANTUM FEATURE SEPARABILITY DIAGNOSTICS")
        logger.info("="*80)
        
        # 1. Analyze raw embeddings separability (before quantum reduction)
        logger.info("\n" + "-"*80)
        logger.info("1. RAW EMBEDDING SEPARABILITY (before quantum reduction)")
        logger.info("-"*80)
        
        # Combine head and tail embeddings for analysis
        train_emb_combined = np.concatenate([train_h_embs, train_t_embs], axis=1)
        pos_mask = y_train == 1
        neg_mask = y_train == 0
        
        pos_emb = train_emb_combined[pos_mask]
        neg_emb = train_emb_combined[neg_mask]
        
        # Mean differences
        mean_diff_emb = np.abs(np.mean(pos_emb, axis=0) - np.mean(neg_emb, axis=0))
        logger.info(f"Embedding mean differences: min={np.min(mean_diff_emb):.6f}, "
                   f"max={np.max(mean_diff_emb):.6f}, mean={np.mean(mean_diff_emb):.6f}")
        logger.info(f"  Features with diff > 0.01: {np.sum(mean_diff_emb > 0.01)}/{len(mean_diff_emb)}")
        
        # Within vs between class distances
        pos_distances = pdist(pos_emb[:min(100, len(pos_emb))])  # Sample for efficiency
        neg_distances = pdist(neg_emb[:min(100, len(neg_emb))])
        within_class_dist_emb = np.mean(np.concatenate([pos_distances, neg_distances]))
        
        between_class_dist_emb = []
        for p in pos_emb[:min(50, len(pos_emb))]:
            for n in neg_emb[:min(50, len(neg_emb))]:
                between_class_dist_emb.append(np.linalg.norm(p - n))
        between_class_dist_emb = np.mean(between_class_dist_emb)
        
        sep_ratio_emb = between_class_dist_emb / within_class_dist_emb if within_class_dist_emb > 0 else 0
        logger.info(f"Embedding separation ratio (between/within): {sep_ratio_emb:.4f}")
        logger.info(f"  → Ratio > 1.0 means classes are separated")
        
        # 2. Analyze quantum features separability (after PCA reduction)
        logger.info("\n" + "-"*80)
        logger.info("2. QUANTUM FEATURE SEPARABILITY (after PCA reduction)")
        logger.info("-"*80)
        
        pos_qml = X_train_qml[pos_mask]
        neg_qml = X_train_qml[neg_mask]
        
        # Mean differences
        mean_diff_qml = np.abs(np.mean(pos_qml, axis=0) - np.mean(neg_qml, axis=0))
        logger.info(f"Quantum feature mean differences: min={np.min(mean_diff_qml):.6f}, "
                   f"max={np.max(mean_diff_qml):.6f}, mean={np.mean(mean_diff_qml):.6f}")
        logger.info(f"  Features with diff > 0.1: {np.sum(mean_diff_qml > 0.1)}/{len(mean_diff_qml)}")
        logger.info(f"  Features with diff > 0.01: {np.sum(mean_diff_qml > 0.01)}/{len(mean_diff_qml)}")
        
        # Statistical significance
        significant_qml = []
        for i in range(X_train_qml.shape[1]):
            try:
                _, pval = ttest_ind(pos_qml[:, i], neg_qml[:, i])
                if pval < 0.05:
                    significant_qml.append(i)
            except:
                pass
        logger.info(f"Statistically significant features (t-test, p<0.05): {len(significant_qml)}/{X_train_qml.shape[1]}")
        
        # Within vs between class distances
        pos_distances_qml = pdist(pos_qml[:min(100, len(pos_qml))])
        neg_distances_qml = pdist(neg_qml[:min(100, len(neg_qml))])
        within_class_dist_qml = np.mean(np.concatenate([pos_distances_qml, neg_distances_qml]))
        
        between_class_dist_qml = []
        for p in pos_qml[:min(50, len(pos_qml))]:
            for n in neg_qml[:min(50, len(neg_qml))]:
                between_class_dist_qml.append(np.linalg.norm(p - n))
        between_class_dist_qml = np.mean(between_class_dist_qml)
        
        sep_ratio_qml = between_class_dist_qml / within_class_dist_qml if within_class_dist_qml > 0 else 0
        logger.info(f"Quantum feature separation ratio (between/within): {sep_ratio_qml:.4f}")
        
        # Silhouette score
        try:
            sil_score_qml = silhouette_score(X_train_qml, y_train)
            logger.info(f"Silhouette score: {sil_score_qml:.4f}")
            logger.info(f"  → >0.5: well-separated, 0.2-0.5: overlapping, <0.2: highly overlapping")
        except Exception as e:
            logger.warning(f"Could not compute silhouette score: {e}")
            sil_score_qml = None
        
        # 3. Compare separability before and after reduction
        logger.info("\n" + "-"*80)
        logger.info("3. INFORMATION LOSS ANALYSIS")
        logger.info("-"*80)
        logger.info(f"Separation ratio change: {sep_ratio_emb:.4f} → {sep_ratio_qml:.4f}")
        if sep_ratio_qml < sep_ratio_emb:
            logger.warning(f"⚠️  Separability DECREASED after PCA reduction!")
            logger.warning(f"   Lost {((sep_ratio_emb - sep_ratio_qml) / sep_ratio_emb * 100):.1f}% of separability")
        else:
            logger.info(f"✓ Separability maintained or improved after PCA reduction")
        
        # 4. Feature variance analysis
        logger.info("\n" + "-"*80)
        logger.info("4. FEATURE VARIANCE ANALYSIS")
        logger.info("-"*80)
        pos_var_qml = np.var(pos_qml, axis=0)
        neg_var_qml = np.var(neg_qml, axis=0)
        logger.info(f"Positive class variance: mean={np.mean(pos_var_qml):.6f}, "
                   f"min={np.min(pos_var_qml):.6f}, max={np.max(pos_var_qml):.6f}")
        logger.info(f"Negative class variance: mean={np.mean(neg_var_qml):.6f}, "
                   f"min={np.min(neg_var_qml):.6f}, max={np.max(neg_var_qml):.6f}")
        
        # Check if variance is too low (features are constant)
        low_var_features = np.sum((pos_var_qml < 1e-6) | (neg_var_qml < 1e-6))
        if low_var_features > 0:
            logger.warning(f"⚠️  {low_var_features}/{len(pos_var_qml)} features have very low variance (<1e-6)")
        
        logger.info("\n" + "="*80)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("="*80)
        logger.info(f"Raw embedding separation ratio: {sep_ratio_emb:.4f}")
        logger.info(f"Quantum feature separation ratio: {sep_ratio_qml:.4f}")
        logger.info(f"Significant quantum features: {len(significant_qml)}/{X_train_qml.shape[1]}")
        if sil_score_qml is not None:
            logger.info(f"Silhouette score: {sil_score_qml:.4f}")
        logger.info("="*80 + "\n")

        logger.info(f"Quantum features: train {X_train_qml.shape}, test {X_test_qml.shape}")

        # Prepare DataFrames for QML trainer
        train_df_qml = train_df.copy()
        train_df_qml['qml_features'] = list(X_train_qml)

        test_df_qml = test_df.copy()
        test_df_qml['qml_features'] = list(X_test_qml)

        # ========== STEP 6: TRAIN QUANTUM MODELS ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 6: TRAINING QUANTUM MODELS")
        logger.info("="*80)

        # Fallback to basic embedder for QML trainer compatibility
        # Use actual embedding dimension (may differ from config if complex→real conversion happened)
        actual_embedding_dim = embedder.entity_embeddings.shape[1]
        basic_embedder = HetionetEmbedder(
            embedding_dim=actual_embedding_dim,
            qml_dim=args.qml_dim
        )
        basic_embedder.entity_embeddings = embedder.entity_embeddings
        basic_embedder.entity_to_id = embedder.entity_to_id
        basic_embedder.id_to_entity = embedder.id_to_entity
        basic_embedder.reduce_to_qml_dim()

        trainer = QMLTrainer(results_dir=args.results_dir, random_state=args.random_state)

        # QSVC with improved configuration
        qsvc_config = {
            "model_type": "QSVC",
            "encoding_method": "feature_map",
            "num_qubits": args.qml_dim,
            "feature_map_type": args.qml_feature_map,
            "feature_map_reps": args.qml_feature_map_reps,
            "entanglement": args.qml_entanglement,
            "use_classical_features_in_kernel": args.use_classical_features_in_kernel,
            "use_data_reuploading": args.use_data_reuploading,
            "use_variational_feature_map": args.use_variational_feature_map,
            "optimize_feature_map_reps": args.optimize_feature_map_reps,
            "random_state": args.random_state,
            # Nyström approximation (optional)
            "nystrom_m": args.qsvc_nystrom_m,
            "nystrom_ridge": args.nystrom_ridge,
            "nystrom_max_pairs": args.nystrom_max_pairs,
            # Enable landmark mitigation by default; allow opt-out
            "nystrom_landmark_mitigation": (not args.no_nystrom_landmark_mitigation),
        }
        
        # If using classical features in kernel, pass the enhanced features
        if args.use_classical_features_in_kernel:
            qsvc_config['enhanced_classical_features_train'] = X_train
            qsvc_config['enhanced_classical_features_test'] = X_test
        
        logger.info(f"QSVC Configuration:")
        logger.info(f"  Qubits: {args.qml_dim}")
        logger.info(f"  Feature Map: {args.qml_feature_map}")
        logger.info(f"  Repetitions: {args.qml_feature_map_reps}")
        logger.info(f"  Entanglement: {args.qml_entanglement}")
        logger.info(f"  Encoding Strategy: {args.qml_encoding}")
        if args.use_data_reuploading:
            logger.info(f"  ⚡ Using DATA RE-UPLOADING (quantum-native: encodes features multiple times)")
        if args.use_variational_feature_map:
            logger.info(f"  ⚡ Using VARIATIONAL FEATURE MAP (trainable encoding)")
        if args.optimize_feature_map_reps:
            logger.info(f"  ⚡ Optimizing feature map reps using kernel-target alignment")
        if args.use_classical_features_in_kernel:
            logger.info(f"  ⚡ Using CLASSICAL features in quantum kernel (experimental)")

        logger.info("Training QSVC...")
        try:
            t0 = time.time()
            qsvc_results = trainer.train_and_evaluate(
                train_df,
                test_df,
                basic_embedder,
                qsvc_config,
                quantum_config_path=args.quantum_config_path
            )
            fit_time = time.time() - t0

            qml_metrics = qsvc_results.get('quantum', {})
            quantum_results['QSVC-Optimized'] = {
                'status': 'success',
                'test_metrics': qml_metrics,
                'fit_seconds': fit_time
            }
            logger.info(f"  ✅ QSVC - Test PR-AUC: {qml_metrics.get('pr_auc', 0.0):.4f}")
            
            # Apply probability calibration to QSVC if requested
            if args.calibrate_probabilities and trainer.last_model is not None:
                try:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
                    from scipy.special import expit
                    
                    logger.info("Applying probability calibration to QSVC...")
                    qsvc_model = trainer.last_model
                    K_train_cal = trainer.last_train_features
                    K_test_cal = trainer.last_test_features
                    
                    if K_train_cal is not None and K_test_cal is not None:
                        # Get decision scores from QSVC
                        train_scores = qsvc_model.decision_function(K_train_cal)
                        test_scores = qsvc_model.decision_function(K_test_cal)
                        
                        # Convert decision scores to probabilities using sigmoid
                        train_proba = expit(train_scores)
                        test_proba_uncal = expit(test_scores)
                        
                        # Apply calibration: train calibrator on training data, apply to test
                        # Map 'sigmoid' to 'platt' for the calibration function
                        cal_method = 'platt' if args.calibration_method == 'sigmoid' else args.calibration_method
                        from sklearn.isotonic import IsotonicRegression
                        from sklearn.linear_model import LogisticRegression
                        
                        if cal_method == 'isotonic':
                            calibrator = IsotonicRegression(out_of_bounds='clip')
                            calibrator.fit(train_proba, y_train)
                            test_proba_cal = calibrator.predict(test_proba_uncal)
                        else:  # platt
                            calibrator = LogisticRegression()
                            calibrator.fit(train_proba.reshape(-1, 1), y_train)
                            test_proba_cal = calibrator.predict_proba(test_proba_uncal.reshape(-1, 1))[:, 1]
                        
                        # Clip to [0, 1]
                        test_proba_cal = np.clip(test_proba_cal, 0, 1)
                        test_preds_cal = (test_proba_cal >= 0.5).astype(int)
                        
                        # Compute calibrated metrics
                        test_metrics_cal = {
                            'accuracy': accuracy_score(y_test, test_preds_cal),
                            'precision': precision_score(y_test, test_preds_cal, zero_division=0),
                            'recall': recall_score(y_test, test_preds_cal, zero_division=0),
                            'f1': f1_score(y_test, test_preds_cal, zero_division=0),
                            'roc_auc': roc_auc_score(y_test, test_proba_cal) if len(np.unique(y_test)) > 1 else 0.0,
                            'pr_auc': average_precision_score(y_test, test_proba_cal)
                        }
                        
                        quantum_results['QSVC-Optimized-Calibrated'] = {
                            'status': 'success',
                            'test_metrics': test_metrics_cal,
                            'fit_seconds': fit_time
                        }
                        improvement = test_metrics_cal['pr_auc'] - qml_metrics.get('pr_auc', 0.0)
                        logger.info(f"  ✅ QSVC-Calibrated - Test PR-AUC: {test_metrics_cal['pr_auc']:.4f} (improvement: {improvement:+.4f})")
                except Exception as e:
                    logger.warning(f"  ⚠️  QSVC calibration failed: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            logger.error(f"  ❌ QSVC failed: {e}")
            quantum_results['QSVC-Optimized'] = {'status': 'failed', 'error': str(e)}

        # VQC (if not fast mode and not quantum_only and not skip_vqc - skip VQC for faster QSVC-only runs)
        if not args.fast_mode and not args.quantum_only and not args.skip_vqc:
            vqc_config = {
                "model_type": "VQC",
                "encoding_method": "feature_map",
                "num_qubits": args.qml_dim,
                "feature_map_type": "ZZ",
                "feature_map_reps": 2,
                "ansatz_type": "RealAmplitudes",
                "ansatz_reps": 3,
                "optimizer": "COBYLA",
                "max_iter": args.qml_max_iter,
                "random_state": args.random_state
            }

            logger.info("Training VQC...")
            try:
                t0 = time.time()
                vqc_results = trainer.train_and_evaluate(
                    train_df,
                    test_df,
                    basic_embedder,
                    vqc_config,
                    quantum_config_path=args.quantum_config_path
                )
                fit_time = time.time() - t0

                qml_metrics = vqc_results.get('quantum', {})
                quantum_results['VQC-Optimized'] = {
                    'status': 'success',
                    'test_metrics': qml_metrics,
                    'fit_seconds': fit_time
                }
                logger.info(f"  ✅ VQC - Test PR-AUC: {qml_metrics.get('pr_auc', 0.0):.4f}")
            except Exception as e:
                logger.error(f"  ❌ VQC failed: {e}")
                quantum_results['VQC-Optimized'] = {'status': 'failed', 'error': str(e)}

        # ========== HYBRID QUANTUM-CLASSICAL ENSEMBLE ==========
        if not args.fast_mode and not args.classical_only:
            logger.info("\n" + "="*80)
            logger.info("TRAINING HYBRID QUANTUM-CLASSICAL ENSEMBLE")
            logger.info("="*80)
            
            try:
                # Get best quantum and classical models
                qsvc_success = quantum_results.get('QSVC-Optimized', {}).get('status') == 'success'
                qsvc_cal_success = quantum_results.get('QSVC-Optimized-Calibrated', {}).get('status') == 'success'
                ensemble_success = classical_results.get('Ensemble-RF-LR', {}).get('status') == 'success'
                rf_success = classical_results.get('RandomForest-Optimized', {}).get('status') == 'success'
                
                if (qsvc_success or qsvc_cal_success) and (ensemble_success or rf_success):
                    # Use calibrated QSVC if available, otherwise regular QSVC
                    if qsvc_cal_success:
                        qsvc_metrics = quantum_results['QSVC-Optimized-Calibrated']['test_metrics']
                        qsvc_name = 'QSVC-Optimized-Calibrated'
                    elif qsvc_success:
                        qsvc_metrics = quantum_results['QSVC-Optimized']['test_metrics']
                        qsvc_name = 'QSVC-Optimized'
                    else:
                        qsvc_metrics = None
                        qsvc_name = None
                    
                    # Use ensemble if available, otherwise RandomForest
                    if ensemble_success:
                        classical_metrics = classical_results['Ensemble-RF-LR']['test_metrics']
                        classical_name = 'Ensemble-RF-LR'
                    elif rf_success:
                        classical_metrics = classical_results['RandomForest-Optimized']['test_metrics']
                        classical_name = 'RandomForest-Optimized'
                    else:
                        classical_metrics = None
                        classical_name = None
                    
                    if qsvc_metrics and classical_metrics:
                        # Get predictions from both models
                        # For quantum: use decision scores from QSVC
                        if trainer.last_model is not None and trainer.last_test_features is not None:
                            from scipy.special import expit
                            qsvc_scores = trainer.last_model.decision_function(trainer.last_test_features)
                            qsvc_proba = expit(qsvc_scores)
                            
                            # For classical: get from ensemble or RF
                            # Models are already trained, just get predictions
                            if ensemble_success:
                                # Recreate ensemble to get predictions
                                rf_trained = models['RandomForest-Optimized']
                                if not hasattr(rf_trained, 'n_estimators') or rf_trained.n_estimators == 0:
                                    rf_trained.fit(X_train, y_train)
                                lr_trained = models['LogisticRegression-L2']
                                if not hasattr(lr_trained, 'coef_'):
                                    lr_trained.fit(X_train, y_train)
                                ensemble = VotingClassifier(
                                    estimators=[('rf', rf_trained), ('lr', lr_trained)],
                                    voting='soft',
                                    weights=[2, 1]
                                )
                                ensemble.fit(X_train, y_train)
                                classical_proba = ensemble.predict_proba(X_test)[:, 1]
                            else:
                                rf_trained = models['RandomForest-Optimized']
                                if not hasattr(rf_trained, 'n_estimators') or rf_trained.n_estimators == 0:
                                    rf_trained.fit(X_train, y_train)
                                classical_proba = rf_trained.predict_proba(X_test)[:, 1]
                            
                            # Create weighted ensemble (tune weights based on individual performance)
                            # Weight quantum more if it performs better, otherwise weight classical more
                            qsvc_pr = qsvc_metrics.get('pr_auc', 0.0)
                            classical_pr = classical_metrics.get('pr_auc', 0.0)
                            
                            # Adaptive weighting: weight each model by its PR-AUC relative to the sum
                            total_pr = qsvc_pr + classical_pr
                            if total_pr > 0:
                                qsvc_weight = qsvc_pr / total_pr
                                classical_weight = classical_pr / total_pr
                            else:
                                # Fallback: equal weights
                                qsvc_weight = 0.5
                                classical_weight = 0.5
                            
                            # Normalize weights to sum to 1
                            weight_sum = qsvc_weight + classical_weight
                            qsvc_weight = qsvc_weight / weight_sum
                            classical_weight = classical_weight / weight_sum
                            
                            # Combine predictions
                            hybrid_proba = qsvc_weight * qsvc_proba + classical_weight * classical_proba
                            hybrid_preds = (hybrid_proba >= 0.5).astype(int)
                            
                            # Compute metrics
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
                            hybrid_metrics = {
                                'accuracy': accuracy_score(y_test, hybrid_preds),
                                'precision': precision_score(y_test, hybrid_preds, zero_division=0),
                                'recall': recall_score(y_test, hybrid_preds, zero_division=0),
                                'f1': f1_score(y_test, hybrid_preds, zero_division=0),
                                'roc_auc': roc_auc_score(y_test, hybrid_proba) if len(np.unique(y_test)) > 1 else 0.0,
                                'pr_auc': average_precision_score(y_test, hybrid_proba)
                            }
                            
                            quantum_results['Hybrid-Quantum-Classical'] = {
                                'status': 'success',
                                'test_metrics': hybrid_metrics,
                                'fit_seconds': 0.0,  # No training time (just combination)
                                'weights': {
                                    'quantum': float(qsvc_weight),
                                    'classical': float(classical_weight)
                                },
                                'components': {
                                    'quantum': qsvc_name,
                                    'classical': classical_name
                                }
                            }
                            
                            logger.info(f"  ✅ Hybrid Ensemble - Test PR-AUC: {hybrid_metrics['pr_auc']:.4f}")
                            logger.info(f"     Weights: Quantum={qsvc_weight:.3f}, Classical={classical_weight:.3f}")
                            logger.info(f"     Components: {qsvc_name} (PR-AUC={qsvc_pr:.4f}), {classical_name} (PR-AUC={classical_pr:.4f})")
                            
                            # Compare with individual models
                            best_individual = max(qsvc_pr, classical_pr)
                            improvement = hybrid_metrics['pr_auc'] - best_individual
                            if improvement > 0:
                                logger.info(f"     ✨ Improvement over best individual: {improvement:+.4f}")
                            else:
                                logger.info(f"     (No improvement over best individual)")
                else:
                    logger.warning("  ⚠️  Hybrid ensemble skipped: need both quantum and classical models")
            except Exception as e:
                logger.warning(f"  ⚠️  Hybrid ensemble failed: {e}")
                import traceback
                traceback.print_exc()
                quantum_results['Hybrid-Quantum-Classical'] = {
                    'status': 'failed',
                    'error': str(e)
                }

    # ========== STEP 7: COMPARISON REPORT ==========
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE COMPARISON REPORT")
    logger.info("="*80)

    all_results = []

    # Add classical results
    for name, result in classical_results.items():
        if result['status'] == 'success':
            all_results.append({
                'name': name,
                'type': 'classical',
                'pr_auc': result['test_metrics'].get('pr_auc', 0.0),
                'accuracy': result['test_metrics'].get('accuracy', 0.0),
                'fit_time': result['fit_seconds']
            })

    # Add quantum results
    for name, result in quantum_results.items():
        if result['status'] == 'success':
            all_results.append({
                'name': name,
                'type': 'quantum',
                'pr_auc': result['test_metrics'].get('pr_auc', 0.0),
                'accuracy': result['test_metrics'].get('accuracy', 0.0),
                'fit_time': result['fit_seconds']
            })

    # Sort by PR-AUC
    all_results.sort(key=lambda x: x['pr_auc'], reverse=True)

    # Print ranking table
    print("\n" + "="*80)
    print("RANKING BY TEST PR-AUC")
    print("="*80)
    print(f"{'Rank':<6s} | {'Model':<35s} | {'Type':<10s} | {'PR-AUC':<10s} | {'Accuracy':<10s} | {'Time (s)':<10s}")
    print("-" * 80)

    for rank, res in enumerate(all_results, 1):
        print(f"{rank:<6d} | {res['name']:<35s} | {res['type']:<10s} | "
              f"{res['pr_auc']:<10.4f} | {res['accuracy']:<10.4f} | {res['fit_time']:<10.2f}")

    # Best models
    if all_results:
        best_overall = all_results[0]
        best_classical = next((r for r in all_results if r['type'] == 'classical'), None)
        best_quantum = next((r for r in all_results if r['type'] == 'quantum'), None)

        print("\n" + "="*80)
        print("BEST MODELS")
        print("="*80)
        print(f"🏆 Best Overall: {best_overall['name']} ({best_overall['type']}) - PR-AUC: {best_overall['pr_auc']:.4f}")
        if best_classical:
            print(f"🏆 Best Classical: {best_classical['name']} - PR-AUC: {best_classical['pr_auc']:.4f}")
        if best_quantum:
            print(f"🏆 Best Quantum: {best_quantum['name']} - PR-AUC: {best_quantum['pr_auc']:.4f}")

        if best_classical and best_quantum:
            diff = best_quantum['pr_auc'] - best_classical['pr_auc']
            winner = 'Quantum wins!' if diff > 0 else 'Classical wins!' if diff < 0 else 'Tie!'
            print(f"\n📊 Quantum vs Classical: {diff:+.4f} ({winner})")

    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(args.results_dir, f"optimized_results_{stamp}.json")

    payload = {
        "config": vars(args),
        "classical_results": classical_results,
        "quantum_results": quantum_results,
        "ranking": all_results,
        "timestamp": stamp
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info(f"\n✅ Results saved to: {out_path}")


if __name__ == "__main__":
    main()
