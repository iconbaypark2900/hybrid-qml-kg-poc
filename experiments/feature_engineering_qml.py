# experiments/feature_engineering_qml.py
"""
Quantum Feature Engineering Experiments - Issue 7 Deep Dive

This script addresses Issue 7: Quantum Feature Engineering by testing:
1. Different feature combinations (diff, hadamard, both, weighted, polynomial)
2. Feature normalization strategies
3. Impact on VQC vs QSVC performance

Addresses:
- Issue 7: Quantum Feature Engineering
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_layer.qml_model import QMLLinkPredictor
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineeringExperiments:
    """Test different feature engineering strategies for quantum models."""
    
    def __init__(self, results_dir: str = "results/feature_engineering", random_state: int = 42):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        np.random.seed(random_state)
    
    def create_feature_encoders(self, qml_dim: int):
        """Create different feature encoding functions."""
        
        def diff_only(h: np.ndarray, t: np.ndarray) -> np.ndarray:
            """|h - t|"""
            return np.abs(h - t)
        
        def hadamard_only(h: np.ndarray, t: np.ndarray) -> np.ndarray:
            """h ⊙ t"""
            return h * t
        
        def concat_both(h: np.ndarray, t: np.ndarray) -> np.ndarray:
            """[|h - t|, h ⊙ t] truncated to qml_dim"""
            diff = np.abs(h - t)
            had = h * t
            both = np.concatenate([diff, had])
            if len(both) > qml_dim:
                return both[:qml_dim]
            elif len(both) < qml_dim:
                return np.pad(both, (0, qml_dim - len(both)))
            return both
        
        def weighted_combo(h: np.ndarray, t: np.ndarray) -> np.ndarray:
            """Weighted combination"""
            diff = np.abs(h - t)
            had = h * t
            mean = (h + t) / 2
            return 0.5 * diff + 0.3 * had + 0.2 * mean
        
        def polynomial_features(h: np.ndarray, t: np.ndarray) -> np.ndarray:
            """Polynomial features (limited interaction terms)"""
            n = min(3, len(h))
            features = []
            for i in range(n):
                for j in range(min(3, len(t))):
                    features.append(h[i] * t[j])
                    if len(features) >= qml_dim:
                        break
                if len(features) >= qml_dim:
                    break
            features = np.array(features[:qml_dim])
            if len(features) < qml_dim:
                features = np.pad(features, (0, qml_dim - len(features)))
            return features
        
        return {
            'diff_only': diff_only,
            'hadamard_only': hadamard_only,
            'concat_both': concat_both,
            'weighted_combo': weighted_combo,
            'polynomial': polynomial_features
        }
    
    def apply_normalization(
        self,
        X: np.ndarray,
        strategy: str
    ) -> np.ndarray:
        """Apply different normalization strategies."""
        if strategy == 'l2':
            return normalize(X, norm='l2')
        elif strategy == 'minmax':
            scaler = MinMaxScaler()
            return scaler.fit_transform(X)
        elif strategy == 'zscore':
            scaler = StandardScaler()
            return scaler.fit_transform(X)
        elif strategy == 'tanh':
            return np.tanh(X)
        elif strategy == 'none':
            return X
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")
    
    def test_feature_strategies(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        y_train: np.ndarray,
        test_pairs: List[Tuple[np.ndarray, np.ndarray]],
        y_test: np.ndarray,
        qml_dim: int,
        qml_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Test different feature encoding strategies."""
        encoders = self.create_feature_encoders(qml_dim)
        results = {}
        
        for name, encoder_fn in encoders.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing feature strategy: {name}")
            logger.info(f"{'='*60}")
            
            # Generate features
            X_train = np.array([encoder_fn(h, t) for h, t in train_pairs])
            X_test = np.array([encoder_fn(h, t) for h, t in test_pairs])
            
            # Check dimensions
            if X_train.shape[1] != qml_dim:
                logger.warning(f"Feature dim mismatch: {X_train.shape[1]} != {qml_dim}, truncating/padding")
                if X_train.shape[1] > qml_dim:
                    X_train = X_train[:, :qml_dim]
                    X_test = X_test[:, :qml_dim]
                else:
                    pad_width = qml_dim - X_train.shape[1]
                    X_train = np.pad(X_train, ((0, 0), (0, pad_width)))
                    X_test = np.pad(X_test, ((0, 0), (0, pad_width)))
            
            # Train model
            try:
                model = QMLLinkPredictor(**qml_config, random_state=self.random_state)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                test_pr_auc = average_precision_score(y_test, y_pred_proba)
                
                y_train_proba = model.predict_proba(X_train)[:, 1]
                train_pr_auc = average_precision_score(y_train, y_train_proba)
                
                results[name] = {
                    'train_pr_auc': float(train_pr_auc),
                    'test_pr_auc': float(test_pr_auc),
                    'feature_dim': int(X_train.shape[1])
                }
                
                logger.info(f"Train PR-AUC: {train_pr_auc:.4f}")
                logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")
            except Exception as e:
                logger.error(f"Failed to train with {name}: {e}")
                results[name] = {
                    'train_pr_auc': None,
                    'test_pr_auc': None,
                    'error': str(e)
                }
        
        return results
    
    def test_normalization_strategies(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        y_train: np.ndarray,
        test_pairs: List[Tuple[np.ndarray, np.ndarray]],
        y_test: np.ndarray,
        qml_dim: int,
        qml_config: Dict[str, Any],
        base_encoder: str = 'diff_only'
    ) -> Dict[str, Dict[str, float]]:
        """Test different normalization strategies."""
        encoders = self.create_feature_encoders(qml_dim)
        encoder_fn = encoders[base_encoder]
        
        normalization_strategies = ['none', 'l2', 'minmax', 'zscore', 'tanh']
        results = {}
        
        # Generate base features
        X_train_base = np.array([encoder_fn(h, t) for h, t in train_pairs])
        X_test_base = np.array([encoder_fn(h, t) for h, t in test_pairs])
        
        for norm_strategy in normalization_strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing normalization: {norm_strategy}")
            logger.info(f"{'='*60}")
            
            # Apply normalization
            X_train = self.apply_normalization(X_train_base.copy(), norm_strategy)
            X_test = self.apply_normalization(X_test_base.copy(), norm_strategy)
            
            # Train model
            try:
                model = QMLLinkPredictor(**qml_config, random_state=self.random_state)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                test_pr_auc = average_precision_score(y_test, y_pred_proba)
                
                y_train_proba = model.predict_proba(X_train)[:, 1]
                train_pr_auc = average_precision_score(y_train, y_train_proba)
                
                results[norm_strategy] = {
                    'train_pr_auc': float(train_pr_auc),
                    'test_pr_auc': float(test_pr_auc)
                }
                
                logger.info(f"Train PR-AUC: {train_pr_auc:.4f}")
                logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")
            except Exception as e:
                logger.error(f"Failed with {norm_strategy}: {e}")
                results[norm_strategy] = {
                    'train_pr_auc': None,
                    'test_pr_auc': None,
                    'error': str(e)
                }
        
        return results
    
    def compare_vqc_vs_qsvc(
        self,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        y_train: np.ndarray,
        test_pairs: List[Tuple[np.ndarray, np.ndarray]],
        y_test: np.ndarray,
        qml_dim: int,
        feature_strategy: str = 'diff_only'
    ) -> Dict[str, Dict[str, float]]:
        """Compare VQC vs QSVC with different feature strategies."""
        encoders = self.create_feature_encoders(qml_dim)
        encoder_fn = encoders[feature_strategy]
        
        # Generate features
        X_train = np.array([encoder_fn(h, t) for h, t in train_pairs])
        X_test = np.array([encoder_fn(h, t) for h, t in test_pairs])
        
        # Adjust dimensions if needed
        if X_train.shape[1] != qml_dim:
            if X_train.shape[1] > qml_dim:
                X_train = X_train[:, :qml_dim]
                X_test = X_test[:, :qml_dim]
            else:
                pad_width = qml_dim - X_train.shape[1]
                X_train = np.pad(X_train, ((0, 0), (0, pad_width)))
                X_test = np.pad(X_test, ((0, 0), (0, pad_width)))
        
        results = {}
        
        # Test VQC
        logger.info(f"\n{'='*60}")
        logger.info("Testing VQC")
        logger.info(f"{'='*60}")
        vqc_config = {
            'model_type': 'VQC',
            'num_qubits': qml_dim,
            'ansatz_type': 'RealAmplitudes',
            'ansatz_reps': 3,
            'optimizer': 'COBYLA',
            'max_iter': 100
        }
        
        try:
            vqc_model = QMLLinkPredictor(**vqc_config, random_state=self.random_state)
            vqc_model.fit(X_train, y_train)
            
            y_pred_proba = vqc_model.predict_proba(X_test)[:, 1]
            test_pr_auc = average_precision_score(y_test, y_pred_proba)
            
            y_train_proba = vqc_model.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_proba)
            
            results['VQC'] = {
                'train_pr_auc': float(train_pr_auc),
                'test_pr_auc': float(test_pr_auc)
            }
            logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")
        except Exception as e:
            logger.error(f"VQC failed: {e}")
            results['VQC'] = {'error': str(e)}
        
        # Test QSVC
        logger.info(f"\n{'='*60}")
        logger.info("Testing QSVC")
        logger.info(f"{'='*60}")
        qsvc_config = {
            'model_type': 'QSVC',
            'num_qubits': qml_dim,
            'feature_map_type': 'ZZ',
            'feature_map_reps': 2
        }
        
        try:
            qsvc_model = QMLLinkPredictor(**qsvc_config, random_state=self.random_state)
            qsvc_model.fit(X_train, y_train)
            
            y_pred_proba = qsvc_model.predict_proba(X_test)[:, 1]
            test_pr_auc = average_precision_score(y_test, y_pred_proba)
            
            y_train_proba = qsvc_model.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_proba)
            
            results['QSVC'] = {
                'train_pr_auc': float(train_pr_auc),
                'test_pr_auc': float(test_pr_auc)
            }
            logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")
        except Exception as e:
            logger.error(f"QSVC failed: {e}")
            results['QSVC'] = {'error': str(e)}
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON."""
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantum Feature Engineering Experiments")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "feature_strategies", "normalization", "vqc_vs_qsvc"])
    parser.add_argument("--results_dir", type=str, default="results/feature_engineering")
    parser.add_argument("--random_state", type=int, default=42)
    
    args = parser.parse_args()
    
    # Load data
    logger.info("Loading Hetionet data...")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(
        df,
        relation_type=args.relation,
        max_entities=args.max_entities
    )
    train_df, test_df = prepare_link_prediction_dataset(task_edges)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedder = HetionetEmbedder(
        embedding_dim=args.embedding_dim,
        qml_dim=args.qml_dim
    )
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
    else:
        embedder.reduce_to_qml_dim()
    
    # Prepare embedding pairs
    # Extract head and tail entity names, then get their embeddings
    cols = {c.lower(): c for c in train_df.columns}
    head_aliases = ("source", "source_id", "head", "head_id", "h", "src", "src_id", "u")
    tail_aliases = ("target", "target_id", "tail", "tail_id", "t", "dst", "dst_id", "v")
    h_col = next((cols[a] for a in head_aliases if a in cols), None)
    t_col = next((cols[a] for a in tail_aliases if a in cols), None)
    
    if not h_col or not t_col:
        raise ValueError(f"Could not find head/tail columns in {list(train_df.columns)}")
    
    train_pairs = []
    for h, t in zip(train_df[h_col].astype(str).values, train_df[t_col].astype(str).values):
        # Access private method for embedding extraction (no public API for single entity)
        hv = embedder._get_vec(h, reduced=True)
        tv = embedder._get_vec(t, reduced=True)
        train_pairs.append((hv, tv))
    
    test_pairs = []
    for h, t in zip(test_df[h_col].astype(str).values, test_df[t_col].astype(str).values):
        hv = embedder._get_vec(h, reduced=True)
        tv = embedder._get_vec(t, reduced=True)
        test_pairs.append((hv, tv))
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Train pairs: {len(train_pairs)}")
    logger.info(f"Test pairs: {len(test_pairs)}")
    
    # Initialize experiments
    experiments = FeatureEngineeringExperiments(
        results_dir=args.results_dir,
        random_state=args.random_state
    )
    
    # QML config
    qml_config = {
        'model_type': 'VQC',
        'num_qubits': args.qml_dim,
        'ansatz_type': 'RealAmplitudes',
        'ansatz_reps': 3,
        'optimizer': 'COBYLA',
        'max_iter': 100
    }
    
    all_results = {}
    
    if args.experiment in ["all", "feature_strategies"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 1: Feature Strategy Comparison")
        logger.info("="*60)
        
        feature_results = experiments.test_feature_strategies(
            train_pairs, y_train, test_pairs, y_test,
            args.qml_dim, qml_config
        )
        all_results['feature_strategies'] = feature_results
        experiments.save_results(feature_results, "feature_strategies.json")
        
        # Print summary
        df = pd.DataFrame(feature_results).T
        df = df.sort_values('test_pr_auc', ascending=False, na_position='last')
        logger.info("\nFeature Strategy Results:")
        logger.info(df.to_string())
    
    if args.experiment in ["all", "normalization"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 2: Normalization Strategy Comparison")
        logger.info("="*60)
        
        norm_results = experiments.test_normalization_strategies(
            train_pairs, y_train, test_pairs, y_test,
            args.qml_dim, qml_config, base_encoder='diff_only'
        )
        all_results['normalization'] = norm_results
        experiments.save_results(norm_results, "normalization.json")
        
        # Print summary
        df = pd.DataFrame(norm_results).T
        df = df.sort_values('test_pr_auc', ascending=False, na_position='last')
        logger.info("\nNormalization Strategy Results:")
        logger.info(df.to_string())
    
    if args.experiment in ["all", "vqc_vs_qsvc"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 3: VQC vs QSVC Comparison")
        logger.info("="*60)
        
        comparison_results = experiments.compare_vqc_vs_qsvc(
            train_pairs, y_train, test_pairs, y_test,
            args.qml_dim, feature_strategy='diff_only'
        )
        all_results['vqc_vs_qsvc'] = comparison_results
        experiments.save_results(comparison_results, "vqc_vs_qsvc.json")
    
    # Save all results
    experiments.save_results(all_results, "all_feature_engineering_results.json")
    
    logger.info("\n" + "="*60)
    logger.info("FEATURE ENGINEERING ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {experiments.results_dir}")


if __name__ == "__main__":
    main()

