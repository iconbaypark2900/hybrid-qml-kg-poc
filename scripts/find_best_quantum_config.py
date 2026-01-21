#!/usr/bin/env python3
"""
Find Best Quantum Configuration with Actual Quantum Kernels

Uses best classical parameters from exploration and tests quantum configs
with actual quantum kernels (not RBF proxy) to find the best quantum setup.

Usage:
    python scripts/find_best_quantum_config.py --relation CtD --use_cached_embeddings
    python scripts/find_best_quantum_config.py --relation CtD --top_n 5  # Test top 5 from RBF screening
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from itertools import product
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

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


def load_best_classical_params(recommendations_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load best classical parameters from recommendations JSON (optional)."""
    if recommendations_file is None:
        return None
    
    try:
        with open(recommendations_file, 'r') as f:
            data = json.load(f)
        
        return {
            'RandomForest': data.get('RandomForest', {}),
            'LogisticRegression': data.get('LogisticRegression', {})
        }
    except FileNotFoundError:
        logger.warning(f"Recommendations file not found: {recommendations_file}")
        return None


def test_quantum_config_with_actual_kernel(
    train_h_embs: np.ndarray,
    train_t_embs: np.ndarray,
    test_h_embs: np.ndarray,
    test_t_embs: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    qml_dim: int,
    random_state: int,
    quantum_config_path: str = "config/quantum_config.yaml"
) -> Dict[str, Any]:
    """Test quantum configuration with actual quantum kernel (not RBF proxy)."""
    try:
        # Prepare quantum features
        engineer = QuantumFeatureEngineer(
            num_qubits=qml_dim,
            encoding_strategy=config.get('encoding', 'optimized_diff'),
            reduction_method=config.get('reduction_method', 'pca'),
            feature_selection_method=config.get('feature_selection_method', 'f_classif'),
            feature_select_k_mult=config.get('feature_select_k_mult', 4.0),
            pre_pca_dim=config.get('pre_pca_dim', 0),
            random_state=random_state
        )
        
        X_train_qml = engineer.prepare_qml_features(
            train_h_embs, train_t_embs, y_train, fit=True
        )
        X_test_qml = engineer.prepare_qml_features(
            test_h_embs, test_t_embs, fit=False
        )
        
        # Skip 1D configurations
        if X_train_qml.shape[1] == 1:
            return {
                'status': 'failed',
                'error': '1D features not suitable for quantum kernels',
                'config': config
            }
        
        # Create args object for qsvc_with_precomputed_kernel
        class Args:
            def __init__(self, config_dict, qml_dim, quantum_config_path):
                self.qml_dim = qml_dim
                self.feature_map = config_dict.get('feature_map', 'ZZ')
                self.feature_map_reps = config_dict.get('feature_map_reps', 2)
                self.entanglement = config_dict.get('entanglement', 'full')
                self.quantum_config = quantum_config_path
                self.nystrom_m = None
                self.nystrom_ridge = 1e-6
                self.nystrom_max_pairs = 20000
                self.nystrom_landmark_mitigation = True
        
        args = Args(config, qml_dim, quantum_config_path)
        
        # Test with actual quantum kernel
        logger.info(f"  Testing with actual quantum kernel...")
        svc, K_train, K_test, kernel_obs = qsvc_with_precomputed_kernel(
            X_train_qml, y_train, X_test_qml, y_test, args, logger
        )
        
        # Evaluate
        from sklearn.svm import SVC
        from sklearn.metrics import roc_auc_score
        
        y_score_test = svc.decision_function(K_test)
        y_score_train = svc.decision_function(K_train)
        
        test_pr_auc = average_precision_score(y_test, y_score_test)
        train_pr_auc = average_precision_score(y_train, y_score_train)
        
        try:
            test_roc_auc = roc_auc_score(y_test, y_score_test)
        except:
            test_roc_auc = 0.0
        
        overfitting_gap = train_pr_auc - test_pr_auc
        
        return {
            'status': 'success',
            'test_pr_auc': float(test_pr_auc),
            'train_pr_auc': float(train_pr_auc),
            'test_roc_auc': float(test_roc_auc),
            'overfitting_gap': float(overfitting_gap),
            'feature_shape': X_train_qml.shape,
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


def find_best_quantum_config(
    recommendations_file: Optional[str] = None,
    relation: str = "CtD",
    pos_edge_sample: Optional[int] = 1500,
    neg_ratio: float = 2.0,
    qml_dim: int = 12,
    top_n: Optional[int] = None,
    use_cached_embeddings: bool = True,
    random_state: int = 42,
    quantum_config_path: str = "config/quantum_config.yaml",
    quantum_only: bool = True
) -> pd.DataFrame:
    """Find best quantum configuration using actual quantum kernels."""
    
    logger.info("="*80)
    logger.info("FINDING BEST QUANTUM CONFIGURATION WITH ACTUAL QUANTUM KERNELS")
    logger.info("="*80)
    
    # Load best classical parameters (optional, only if not quantum_only)
    if not quantum_only and recommendations_file:
        logger.info(f"\nLoading best classical parameters from {recommendations_file}...")
        classical_params = load_best_classical_params(recommendations_file)
        if classical_params:
            logger.info(f"  RandomForest: {classical_params['RandomForest']}")
            logger.info(f"  LogisticRegression: {classical_params['LogisticRegression']}")
    else:
        logger.info("\n⚠️  Quantum-only mode: Skipping classical model training")
    
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
    
    # task_edges from extract_task_edges has both integer IDs (source_id, target_id) 
    # and entity strings (source, target) - use entity strings for embedding training
    
    # Create train/test split (using integer IDs for consistency)
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
    
    # Load or train embeddings
    logger.info("\n" + "="*80)
    logger.info("LOADING EMBEDDINGS")
    logger.info("="*80)
    
    embedder = AdvancedKGEmbedder(
        method='RotatE',
        embedding_dim=128,
        num_epochs=200,
        random_state=random_state
    )
    
    # Prepare triples for embedding training
    # Need columns: source, metaedge/relation, target (entity strings)
    # task_edges should already have these columns from extract_task_edges
    if 'metaedge' in task_edges.columns:
        embedding_training_edges = task_edges[["source", "metaedge", "target"]].copy()
    elif 'relation' in task_edges.columns:
        embedding_training_edges = task_edges[["source", "relation", "target"]].copy()
        embedding_training_edges.rename(columns={'relation': 'metaedge'}, inplace=True)
    else:
        # Fallback: create metaedge column
        embedding_training_edges = task_edges[["source", "target"]].copy()
        embedding_training_edges['metaedge'] = relation
    
    if use_cached_embeddings:
        if embedder.load_embeddings():
            logger.info("✓ Loaded cached embeddings")
        else:
            logger.info("Training embeddings (this may take a while)...")
            embedder.train_embeddings(embedding_training_edges)
    else:
        logger.info("Training embeddings...")
        embedder.train_embeddings(embedding_training_edges)
    
    # Get embeddings - use get_all_embeddings() which returns a dict
    # Maps entity_id (string like "Compound::DB00153") -> embedding array
    entity_embeddings_dict = embedder.get_all_embeddings()
    
    # Extract head and tail embeddings
    # id_to_entity maps integer id -> entity_id (string)
    # We need to convert integer IDs to entity strings
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
    
    # Define quantum parameter grid
    # Start with top configurations from exploration, or use full grid
    if top_n:
        # Load exploration results to get top N
        csv_files = list(Path("results").glob("quantum_exploration_*.csv"))
        if csv_files:
            latest = sorted(csv_files)[-1]
            logger.info(f"Loading exploration results from {latest}")
            df_explore = pd.read_csv(latest)
            
            # Handle both old format (no test_pr_auc) and new format
            if 'test_pr_auc' in df_explore.columns:
                df_explore = df_explore.sort_values('test_pr_auc', ascending=False).head(top_n)
            elif 'quality_score' in df_explore.columns:
                df_explore = df_explore.sort_values('quality_score', ascending=False).head(top_n)
            else:
                logger.warning("No ranking column found, using first N rows")
                df_explore = df_explore.head(top_n)
            
            configs = []
            for _, row in df_explore.iterrows():
                # Handle NaN values
                encoding = row.get('param_encoding', 'hybrid')
                if pd.isna(encoding):
                    encoding = 'hybrid'
                
                reduction = row.get('param_reduction_method', 'pca')
                if pd.isna(reduction):
                    reduction = 'pca'
                
                feat_sel = row.get('param_feature_selection_method', None)
                if pd.isna(feat_sel):
                    feat_sel = None
                
                config = {
                    'encoding': encoding,
                    'reduction_method': reduction,
                    'feature_selection_method': feat_sel,
                    'feature_select_k_mult': float(row.get('param_feature_select_k_mult', 4.0)),
                    'pre_pca_dim': int(row.get('param_pre_pca_dim', 0)),
                    'feature_map': 'ZZ',
                    'feature_map_reps': 2,
                    'entanglement': 'full'
                }
                configs.append(config)
            
            logger.info(f"Testing top {top_n} configurations from exploration...")
        else:
            logger.warning("No exploration CSV found, using default grid")
            configs = None
    else:
        configs = None
    
    if configs is None:
        # Use reduced grid for actual quantum kernel testing (it's slow!)
        param_grid = {
            'encoding': ['hybrid', 'optimized_diff'],
            'reduction_method': ['pca'],  # Skip LDA (gives 1D)
            'feature_selection_method': ['mutual_info', 'f_classif', None],
            'feature_select_k_mult': [2.0, 4.0],
            'pre_pca_dim': [0, 64, 128],
            'feature_map': ['ZZ'],
            'feature_map_reps': [2, 3],
            'entanglement': ['full']
        }
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        configs = [dict(zip(keys, combo)) for combo in combinations]
        
        # Filter invalid combinations
        configs = [c for c in configs if not (c['reduction_method'] == 'lda' and c['pre_pca_dim'] > 0)]
        logger.info(f"Testing {len(configs)} quantum configurations...")
    
    # Test each configuration
    logger.info("\n" + "="*80)
    logger.info("TESTING QUANTUM CONFIGURATIONS WITH ACTUAL KERNELS")
    logger.info("="*80)
    logger.info("⚠️  This will be slow - testing actual quantum kernels!")
    logger.info("="*80)
    
    results = []
    for i, config in enumerate(configs, 1):
        logger.info(f"\n[{i}/{len(configs)}] Testing: {config}")
        
        result = test_quantum_config_with_actual_kernel(
            train_h_embs, train_t_embs, test_h_embs, test_t_embs,
            y_train, y_test, config, qml_dim, random_state, quantum_config_path
        )
        
        if result['status'] == 'success':
            logger.info(f"  ✓ Test PR-AUC: {result['test_pr_auc']:.4f}, "
                       f"ROC-AUC: {result['test_roc_auc']:.4f}, "
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
    parser = argparse.ArgumentParser(description="Find best quantum config with actual quantum kernels")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--pos_edge_sample", type=int, default=1500)
    parser.add_argument("--neg_ratio", type=float, default=2.0)
    parser.add_argument("--qml_dim", type=int, default=12)
    parser.add_argument("--top_n", type=int, default=None,
                       help="Test only top N configs from exploration (faster)")
    parser.add_argument("--use_cached_embeddings", action="store_true", default=True)
    parser.add_argument("--recommendations_file", type=str, default=None,
                       help="Path to recommendations JSON file (optional)")
    parser.add_argument("--quantum_config_path", type=str,
                       default="config/quantum_config.yaml")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--quantum_only", action="store_true", default=True,
                       help="Test only quantum models (skip classical)")
    
    args = parser.parse_args()
    
    set_global_seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find best quantum config
    df_results = find_best_quantum_config(
        recommendations_file=args.recommendations_file,
        relation=args.relation,
        pos_edge_sample=args.pos_edge_sample,
        neg_ratio=args.neg_ratio,
        qml_dim=args.qml_dim,
        top_n=args.top_n,
        use_cached_embeddings=args.use_cached_embeddings,
        random_state=args.random_state,
        quantum_config_path=args.quantum_config_path,
        quantum_only=args.quantum_only
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_file = os.path.join(args.output_dir, f"quantum_actual_kernel_results_{timestamp}.csv")
    df_results.to_csv(csv_file, index=False)
    logger.info(f"\n✓ Results saved to {csv_file}")
    
    # Print top configurations
    logger.info("\n" + "="*80)
    logger.info("TOP 5 QUANTUM CONFIGURATIONS (ACTUAL KERNELS)")
    logger.info("="*80)
    
    for i, (_, row) in enumerate(df_results.head(5).iterrows(), 1):
        logger.info(f"\n{i}. Test PR-AUC: {row['test_pr_auc']:.4f}, ROC-AUC: {row['test_roc_auc']:.4f}")
        logger.info(f"   Overfitting Gap: {row['overfitting_gap']:.4f}")
        logger.info(f"   Config:")
        for col in row.index:
            if col.startswith('param_'):
                logger.info(f"     {col.replace('param_', '')}: {row[col]}")
    
    # Save best config
    if len(df_results) > 0:
        best = df_results.iloc[0]
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        best_config = {
            'test_pr_auc': convert_to_native(best['test_pr_auc']),
            'test_roc_auc': convert_to_native(best['test_roc_auc']),
            'overfitting_gap': convert_to_native(best['overfitting_gap']),
            'config': {col.replace('param_', ''): convert_to_native(best[col]) 
                      for col in best.index if col.startswith('param_')}
        }
        
        json_file = os.path.join(args.output_dir, f"best_quantum_config_actual_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        logger.info(f"\n✓ Best config saved to {json_file}")


if __name__ == "__main__":
    main()
