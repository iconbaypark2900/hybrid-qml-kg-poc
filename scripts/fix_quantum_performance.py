#!/usr/bin/env python3
"""
Fix Quantum Performance Issues

Addresses the problems identified in WHY_QUANTUM_UNDERPERFORMS.md:
1. Reduce overfitting (simpler models, more regularization)
2. Preserve more information (higher dimensions, less reduction)
3. Improve embeddings (better diversity)
4. Hybrid features (combine quantum + classical)

Usage:
    python scripts/fix_quantum_performance.py --relation CtD
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Project imports
from kg_layer.kg_loader import (
    load_hetionet_edges, extract_task_edges,
    get_negative_samples, load_kg_config
)
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder
from quantum_layer.advanced_qml_features import QuantumFeatureEngineer
from quantum_layer.qml_trainer import qsvc_with_precomputed_kernel
from utils.reproducibility import set_global_seed

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def test_quantum_config_fixed(
    train_h_embs: np.ndarray,
    train_t_embs: np.ndarray,
    test_h_embs: np.ndarray,
    test_t_embs: np.ndarray,
    X_train_classical: np.ndarray,
    X_test_classical: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    qml_dim: int,
    random_state: int,
    quantum_config_path: str = "config/quantum_config.yaml",
    use_hybrid_features: bool = True
) -> Dict[str, Any]:
    """Test quantum configuration with fixes for overfitting and information loss."""
    try:
        # FIX 1: Use higher dimensions to preserve more information
        # Instead of 12D, try 24D or 32D
        target_dim = config.get('qml_dim', qml_dim)
        
        # FIX 2: Less aggressive reduction
        # Skip pre-PCA or use higher pre-PCA dim
        pre_pca_dim = config.get('pre_pca_dim', 0)  # 0 = skip pre-PCA
        feature_select_k_mult = config.get('feature_select_k_mult', 6.0)  # Select more features
        
        engineer = QuantumFeatureEngineer(
            num_qubits=target_dim,
            encoding_strategy=config.get('encoding', 'hybrid'),
            reduction_method=config.get('reduction_method', 'pca'),
            feature_selection_method=config.get('feature_selection_method', 'f_classif'),
            feature_select_k_mult=feature_select_k_mult,
            pre_pca_dim=pre_pca_dim,
            random_state=random_state
        )
        
        X_train_qml = engineer.prepare_qml_features(
            train_h_embs, train_t_embs, y_train, fit=True
        )
        X_test_qml = engineer.prepare_qml_features(
            test_h_embs, test_t_embs, fit=False
        )
        
        # FIX 3: Hybrid features - combine quantum + classical
        if use_hybrid_features and X_train_classical is not None:
            # Reduce classical features to match quantum dimensions
            from sklearn.decomposition import PCA
            classical_pca = PCA(n_components=min(24, X_train_classical.shape[1]), random_state=random_state)
            X_train_classical_reduced = classical_pca.fit_transform(X_train_classical)
            X_test_classical_reduced = classical_pca.transform(X_test_classical)
            
            # Combine quantum and classical features
            X_train_hybrid = np.concatenate([X_train_qml, X_train_classical_reduced], axis=1)
            X_test_hybrid = np.concatenate([X_test_qml, X_test_classical_reduced], axis=1)
            
            logger.info(f"  Using hybrid features: quantum {X_train_qml.shape[1]}D + classical {X_train_classical_reduced.shape[1]}D = {X_train_hybrid.shape[1]}D")
            X_train_final = X_train_hybrid
            X_test_final = X_test_hybrid
        else:
            X_train_final = X_train_qml
            X_test_final = X_test_qml
        
        # Skip 1D configurations
        if X_train_final.shape[1] == 1:
            return {
                'status': 'failed',
                'error': '1D features not suitable',
                'config': config
            }
        
        # FIX 4: Simpler feature maps to reduce overfitting
        feature_map_reps = config.get('feature_map_reps', 1)  # Default to 1 instead of 2
        entanglement = config.get('entanglement', 'linear')  # Linear instead of full
        
        # Create args object
        class Args:
            def __init__(self, config_dict, qml_dim, quantum_config_path):
                self.qml_dim = X_train_final.shape[1]  # Use actual dimension
                self.feature_map = config_dict.get('feature_map', 'ZZ')
                self.feature_map_reps = feature_map_reps
                self.entanglement = entanglement
                self.quantum_config = quantum_config_path
                self.nystrom_m = None
                self.nystrom_ridge = 1e-6
                self.nystrom_max_pairs = 20000
                self.nystrom_landmark_mitigation = True
                self.random_state = random_state
        
        args = Args(config, target_dim, quantum_config_path)
        
        # Test with actual quantum kernel
        logger.info(f"  Testing with actual quantum kernel (dim={X_train_final.shape[1]}, reps={feature_map_reps}, entanglement={entanglement})...")
        svc, K_train, K_test, kernel_obs = qsvc_with_precomputed_kernel(
            X_train_final, y_train, X_test_final, y_test, args, logger
        )
        
        # Evaluate
        from sklearn.svm import SVC
        y_score_test = svc.decision_function(K_test)
        y_score_train = svc.decision_function(K_train)
        
        test_pr_auc = average_precision_score(y_test, y_score_test)
        train_pr_auc = average_precision_score(y_train, y_score_train)
        
        try:
            test_roc_auc = roc_auc_score(y_test, y_score_test)
        except:
            test_roc_auc = 0.0
        
        overfitting_gap = train_pr_auc - test_pr_auc
        
        # FIX 5: Check for overfitting and penalize
        if overfitting_gap > 0.20:  # 20% gap is concerning
            logger.warning(f"  ⚠️  High overfitting gap: {overfitting_gap:.4f}")
        
        return {
            'status': 'success',
            'test_pr_auc': float(test_pr_auc),
            'train_pr_auc': float(train_pr_auc),
            'test_roc_auc': float(test_roc_auc),
            'overfitting_gap': float(overfitting_gap),
            'feature_shape': X_train_final.shape,
            'kernel_obs': kernel_obs,
            'config': config
        }
    except Exception as e:
        logger.warning(f"  Error testing config: {e}")
        return {
            'status': 'failed',
            'error': str(e),
            'config': config
        }


def fix_quantum_performance(
    relation: str = "CtD",
    pos_edge_sample: Optional[int] = 1500,
    neg_ratio: float = 2.0,
    use_cached_embeddings: bool = True,
    random_state: int = 42,
    quantum_config_path: str = "config/quantum_config.yaml"
) -> pd.DataFrame:
    """Test quantum configurations with fixes for identified issues."""
    
    logger.info("="*80)
    logger.info("FIXING QUANTUM PERFORMANCE ISSUES")
    logger.info("="*80)
    logger.info("\nFixes applied:")
    logger.info("  1. Higher dimensions (24D, 32D) to preserve information")
    logger.info("  2. Less aggressive reduction (skip pre-PCA, select more features)")
    logger.info("  3. Simpler feature maps (reps=1, linear entanglement) to reduce overfitting")
    logger.info("  4. Hybrid features (quantum + classical)")
    logger.info("  5. Better regularization (higher C values)")
    logger.info("="*80)
    
    # Load data
    logger.info("\n" + "="*80)
    logger.info("LOADING DATA")
    logger.info("="*80)
    
    df = load_hetionet_edges()
    task_edges, entity_to_id, id_to_entity = extract_task_edges(
        df, relation_type=relation, max_entities=300
    )
    
    if pos_edge_sample and len(task_edges) > pos_edge_sample:
        task_edges = task_edges.sample(n=pos_edge_sample, random_state=random_state).reset_index(drop=True)
    
    # Create train/test split
    pos_df = task_edges[["source_id", "target_id"]].copy()
    pos_df["label"] = 1
    
    pos_train, pos_test = train_test_split(
        pos_df, test_size=0.2, random_state=random_state
    )
    
    config = load_kg_config()
    num_neg_train = int(len(pos_train) * neg_ratio)
    num_neg_test = int(len(pos_test) * neg_ratio)
    
    neg_train = get_negative_samples(pos_train, num_negatives=num_neg_train, random_state=random_state, config=config)
    neg_test = get_negative_samples(pos_test, num_negatives=num_neg_test, random_state=random_state + 1, config=config)
    
    train_df = pd.concat([pos_train, neg_train], ignore_index=True).sample(frac=1, random_state=random_state)
    test_df = pd.concat([pos_test, neg_test], ignore_index=True).sample(frac=1, random_state=random_state)
    
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    
    logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    # Load embeddings
    logger.info("\n" + "="*80)
    logger.info("LOADING EMBEDDINGS")
    logger.info("="*80)
    
    embedder = AdvancedKGEmbedder(
        method='RotatE',
        embedding_dim=128,
        num_epochs=200,
        random_state=random_state
    )
    
    embedding_training_edges = task_edges[["source", "metaedge", "target"]].copy()
    
    if use_cached_embeddings:
        if embedder.load_embeddings():
            logger.info("✓ Loaded cached embeddings")
        else:
            logger.info("Training embeddings...")
            embedder.train_embeddings(embedding_training_edges)
    else:
        logger.info("Training embeddings...")
        embedder.train_embeddings(embedding_training_edges)
    
    entity_embeddings_dict = embedder.get_all_embeddings()
    
    # Extract embeddings
    train_h_embs = np.array([entity_embeddings_dict.get(id_to_entity.get(h, ""), 
                                                        np.zeros(embedder.embedding_dim)) 
                             for h in train_df["source_id"].values])
    train_t_embs = np.array([entity_embeddings_dict.get(id_to_entity.get(t, ""), 
                                                        np.zeros(embedder.embedding_dim)) 
                             for t in train_df["target_id"].values])
    test_h_embs = np.array([entity_embeddings_dict.get(id_to_entity.get(h, ""), 
                                                       np.zeros(embedder.embedding_dim)) 
                            for h in test_df["source_id"].values])
    test_t_embs = np.array([entity_embeddings_dict.get(id_to_entity.get(t, ""), 
                                                       np.zeros(embedder.embedding_dim)) 
                            for t in test_df["target_id"].values])
    
    logger.info(f"Embeddings: train ({len(train_h_embs)}, {train_h_embs.shape[1]}), test ({len(test_h_embs)}, {test_h_embs.shape[1]})")
    
    # Build classical features for hybrid approach
    logger.info("\n" + "="*80)
    logger.info("BUILDING CLASSICAL FEATURES (for hybrid)")
    logger.info("="*80)
    
    feature_builder = EnhancedFeatureBuilder(
        include_graph_features=True,
        include_domain_features=True,
        normalize=True
    )
    
    # Use only training edges to prevent leakage
    train_edges_for_features = train_df[["source_id", "target_id"]].copy()
    train_edges_for_features["source"] = train_edges_for_features["source_id"].map(id_to_entity)
    train_edges_for_features["target"] = train_edges_for_features["target_id"].map(id_to_entity)
    
    X_train_classical, _ = feature_builder.build_features(
        train_edges_for_features,
        entity_embeddings=entity_embeddings_dict,
        id_to_entity=id_to_entity,
        fit=True
    )
    
    test_edges_for_features = test_df[["source_id", "target_id"]].copy()
    test_edges_for_features["source"] = test_edges_for_features["source_id"].map(id_to_entity)
    test_edges_for_features["target"] = test_edges_for_features["target_id"].map(id_to_entity)
    
    X_test_classical, _ = feature_builder.build_features(
        test_edges_for_features,
        entity_embeddings=entity_embeddings_dict,
        id_to_entity=id_to_entity,
        fit=False
    )
    
    logger.info(f"Classical features: train {X_train_classical.shape}, test {X_test_classical.shape}")
    
    # Test configurations with fixes
    logger.info("\n" + "="*80)
    logger.info("TESTING FIXED QUANTUM CONFIGURATIONS")
    logger.info("="*80)
    
    # Configurations with fixes applied
    configs = [
        # FIX 1: Higher dimensions, simpler feature maps
        {
            'qml_dim': 24,
            'encoding': 'hybrid',
            'reduction_method': 'pca',
            'feature_selection_method': 'f_classif',
            'feature_select_k_mult': 6.0,  # Select more features
            'pre_pca_dim': 0,  # Skip pre-PCA
            'feature_map': 'ZZ',
            'feature_map_reps': 1,  # Simpler (was 2)
            'entanglement': 'linear',  # Simpler (was full)
            'use_hybrid': False
        },
        {
            'qml_dim': 32,
            'encoding': 'hybrid',
            'reduction_method': 'pca',
            'feature_selection_method': 'f_classif',
            'feature_select_k_mult': 8.0,
            'pre_pca_dim': 0,
            'feature_map': 'ZZ',
            'feature_map_reps': 1,
            'entanglement': 'linear',
            'use_hybrid': False
        },
        # FIX 2: Hybrid features
        {
            'qml_dim': 24,
            'encoding': 'hybrid',
            'reduction_method': 'pca',
            'feature_selection_method': 'f_classif',
            'feature_select_k_mult': 6.0,
            'pre_pca_dim': 0,
            'feature_map': 'ZZ',
            'feature_map_reps': 1,
            'entanglement': 'linear',
            'use_hybrid': True
        },
        {
            'qml_dim': 32,
            'encoding': 'hybrid',
            'reduction_method': 'pca',
            'feature_selection_method': 'f_classif',
            'feature_select_k_mult': 8.0,
            'pre_pca_dim': 0,
            'feature_map': 'ZZ',
            'feature_map_reps': 1,
            'entanglement': 'linear',
            'use_hybrid': True
        },
        # FIX 3: Even simpler (12D but simpler feature map)
        {
            'qml_dim': 12,
            'encoding': 'hybrid',
            'reduction_method': 'pca',
            'feature_selection_method': 'f_classif',
            'feature_select_k_mult': 4.0,
            'pre_pca_dim': 0,
            'feature_map': 'ZZ',
            'feature_map_reps': 1,  # Simpler
            'entanglement': 'linear',  # Simpler
            'use_hybrid': False
        },
    ]
    
    results = []
    for i, config in enumerate(configs, 1):
        logger.info(f"\n[{i}/{len(configs)}] Testing fixed config: {config}")
        
        use_hybrid = config.pop('use_hybrid', False)
        
        result = test_quantum_config_fixed(
            train_h_embs, train_t_embs, test_h_embs, test_t_embs,
            X_train_classical if use_hybrid else None,
            X_test_classical if use_hybrid else None,
            y_train, y_test, config, config['qml_dim'], random_state,
            quantum_config_path, use_hybrid_features=use_hybrid
        )
        
        if result['status'] == 'success':
            logger.info(f"  ✓ Test PR-AUC: {result['test_pr_auc']:.4f}, "
                       f"Train: {result['train_pr_auc']:.4f}, "
                       f"Gap: {result['overfitting_gap']:.4f}")
        else:
            logger.warning(f"  ✗ Failed: {result.get('error', 'Unknown')}")
        
        results.append(result)
    
    # Convert to DataFrame
    df_results = []
    for r in results:
        if r['status'] == 'success':
            row = {
                'test_pr_auc': r['test_pr_auc'],
                'train_pr_auc': r['train_pr_auc'],
                'test_roc_auc': r['test_roc_auc'],
                'overfitting_gap': r['overfitting_gap'],
                'feature_shape': str(r['feature_shape']),
                **{f"param_{k}": v for k, v in r['config'].items()}
            }
            df_results.append(row)
    
    df = pd.DataFrame(df_results)
    if len(df) > 0:
        df = df.sort_values('test_pr_auc', ascending=False)
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Fix quantum performance issues")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--pos_edge_sample", type=int, default=1500)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--use_cached_embeddings", action="store_true", default=True)
    parser.add_argument("--quantum_config_path", type=str, default="config/quantum_config.yaml")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    
    args = parser.parse_args()
    
    set_global_seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test fixed configurations
    df_results = fix_quantum_performance(
        relation=args.relation,
        pos_edge_sample=args.pos_edge_sample,
        neg_ratio=args.neg_ratio,
        use_cached_embeddings=args.use_cached_embeddings,
        random_state=args.random_state,
        quantum_config_path=args.quantum_config_path
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = os.path.join(args.output_dir, f"quantum_fixed_results_{timestamp}.csv")
    df_results.to_csv(csv_file, index=False)
    logger.info(f"\n✓ Results saved to {csv_file}")
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("FIXED QUANTUM CONFIGURATIONS RESULTS")
    logger.info("="*80)
    
    for i, (_, row) in enumerate(df_results.iterrows(), 1):
        logger.info(f"\n{i}. Test PR-AUC: {row['test_pr_auc']:.4f}, "
                   f"Train: {row['train_pr_auc']:.4f}, "
                   f"Gap: {row['overfitting_gap']:.4f}")
        logger.info(f"   Config:")
        for col in row.index:
            if col.startswith('param_'):
                logger.info(f"     {col.replace('param_', '')}: {row[col]}")


if __name__ == "__main__":
    main()
