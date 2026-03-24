# experiments/vqc_optimization_analysis.py
"""
VQC Optimization Analysis - Issue 1 Deep Dive

This script addresses Issue 1: VQC Underperforming (PR-AUC: 0.49)
by systematically analyzing optimization behavior and comparing architectures.

Experiments:
1. Loss tracking during training
2. Optimizer comparison (COBYLA, SPSA, NFT, ADAM)
3. Ansatz architecture search
4. Hyperparameter grid search
"""

import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal, ZZFeatureMap, ZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, SPSA, NFT
from qiskit_algorithms.optimizers.optimizer import OptimizerResult
from qiskit_machine_learning.algorithms import VQC
from qiskit.primitives import Sampler

from quantum_layer.qml_model import QMLLinkPredictor
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LossTracker:
    """Callback to track loss during VQC training.

    Compatible with multiple optimizer callback signatures:
    - COBYLA (scipy): callback(x)
    - SPSA (Qiskit): callback(nfev, params, fval, stepsize, accepted)
    - Generic Qiskit: callback(nfev, parameters, loss, step)
    """

    def __init__(self):
        self.losses: List[float] = []
        self.iterations: List[int] = []
        self.start_time = None
        self._call_count = 0

    def callback(self, *args, **kwargs) -> None:
        """Called during optimizer iteration. Accepts variable signatures."""
        if self.start_time is None:
            self.start_time = time.time()

        self._call_count += 1
        nfev = self._call_count

        if len(args) >= 3:
            # Qiskit SPSA / generic: (nfev, params, fval, ...)
            nfev = int(args[0]) if not isinstance(args[0], np.ndarray) else self._call_count
            loss = float(args[2]) if len(args) > 2 and not isinstance(args[2], np.ndarray) else float("nan")
        elif len(args) == 1:
            # COBYLA: callback(x) -- no loss available directly
            loss = float("nan")
        else:
            loss = float("nan")

        self.iterations.append(nfev)
        self.losses.append(loss)

        if nfev % 10 == 0:
            elapsed = time.time() - self.start_time
            if np.isnan(loss):
                logger.info(f"  Iter {nfev}: (elapsed: {elapsed:.1f}s)")
            else:
                logger.info(f"  Iter {nfev}: loss={loss:.6f} (elapsed: {elapsed:.1f}s)")

    def reset(self):
        """Reset tracker for new run."""
        self.losses = []
        self.iterations = []
        self.start_time = None
        self._call_count = 0


class VQCOptimizationAnalyzer:
    """Analyze VQC optimization behavior and compare configurations."""
    
    def __init__(self, results_dir: str = "results/vqc_analysis", random_state: int = 42):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        np.random.seed(random_state)
        
    def track_loss_curve(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_map,
        ansatz,
        optimizer,
        max_iter: int = 200
    ) -> Tuple[VQC, LossTracker]:
        """Train VQC with loss tracking."""
        tracker = LossTracker()
        
        # Create optimizer with callback
        if isinstance(optimizer, COBYLA):
            opt = COBYLA(maxiter=max_iter, callback=tracker.callback)
        elif isinstance(optimizer, SPSA):
            opt = SPSA(maxiter=max_iter, callback=tracker.callback)
        else:
            # For other optimizers, wrap if needed
            opt = optimizer
            logger.warning(f"Loss tracking may not work for {type(optimizer)}")
        
        vqc = VQC(
            sampler=Sampler(),
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=opt
        )
        
        logger.info(f"Training VQC with {type(optimizer).__name__} ({max_iter} iterations)...")
        vqc.fit(X_train, y_train)
        
        return vqc, tracker
    
    def compare_optimizers(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_map,
        ansatz,
        max_iter: int = 200
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different optimizers."""
        optimizers = {
            'COBYLA': COBYLA(maxiter=max_iter),
            'SPSA': SPSA(maxiter=max_iter),
        }
        
        # Try to add NFT if available
        try:
            optimizers['NFT'] = NFT(maxiter=max_iter)
        except Exception as e:
            logger.warning(f"NFT optimizer not available: {e}")
        
        results = {}
        
        for name, opt in optimizers.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing optimizer: {name}")
            logger.info(f"{'='*60}")
            
            tracker = LossTracker()
            
            # Create optimizer with callback
            if name == 'COBYLA':
                opt_with_callback = COBYLA(maxiter=max_iter, callback=tracker.callback)
            elif name == 'SPSA':
                opt_with_callback = SPSA(maxiter=max_iter, callback=tracker.callback)
            else:
                opt_with_callback = opt
            
            vqc = VQC(
                sampler=Sampler(),
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=opt_with_callback
            )
            
            start_time = time.time()
            vqc.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            y_pred_proba = vqc.predict_proba(X_test)[:, 1]
            test_pr_auc = average_precision_score(y_test, y_pred_proba)
            
            y_train_proba = vqc.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_proba)
            
            results[name] = {
                'train_pr_auc': train_pr_auc,
                'test_pr_auc': test_pr_auc,
                'train_time': train_time,
                'loss_curve': {
                    'iterations': tracker.iterations,
                    'losses': tracker.losses
                },
                'final_loss': tracker.losses[-1] if tracker.losses else None
            }
            
            logger.info(f"Train PR-AUC: {train_pr_auc:.4f}")
            logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")
            logger.info(f"Training time: {train_time:.2f}s")
        
        return results
    
    def compare_ansatzes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_map,
        num_qubits: int,
        optimizer,
        max_iter: int = 200
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different ansatz architectures."""
        ansatzes = {
            'RealAmplitudes_rep2': RealAmplitudes(num_qubits=num_qubits, reps=2),
            'RealAmplitudes_rep3': RealAmplitudes(num_qubits=num_qubits, reps=3),
            'RealAmplitudes_rep4': RealAmplitudes(num_qubits=num_qubits, reps=4),
            'EfficientSU2_rep2': EfficientSU2(num_qubits=num_qubits, reps=2),
            'EfficientSU2_rep3': EfficientSU2(num_qubits=num_qubits, reps=3),
            'TwoLocal_rep2': TwoLocal(
                num_qubits=num_qubits,
                rotation_blocks='ry',
                entanglement_blocks='cz',
                reps=2
            ),
            'TwoLocal_rep3': TwoLocal(
                num_qubits=num_qubits,
                rotation_blocks='ry',
                entanglement_blocks='cz',
                reps=3
            ),
        }
        
        results = {}
        
        for name, ansatz in ansatzes.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing ansatz: {name}")
            logger.info(f"  Parameters: {ansatz.num_parameters}")
            logger.info(f"  Depth: {ansatz.depth()}")
            logger.info(f"{'='*60}")
            
            vqc = VQC(
                sampler=Sampler(),
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer
            )
            
            start_time = time.time()
            vqc.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            y_pred_proba = vqc.predict_proba(X_test)[:, 1]
            test_pr_auc = average_precision_score(y_test, y_pred_proba)
            
            y_train_proba = vqc.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_proba)
            
            results[name] = {
                'train_pr_auc': train_pr_auc,
                'test_pr_auc': test_pr_auc,
                'train_time': train_time,
                'num_parameters': ansatz.num_parameters,
                'depth': ansatz.depth()
            }
            
            logger.info(f"Train PR-AUC: {train_pr_auc:.4f}")
            logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")
            logger.info(f"Training time: {train_time:.2f}s")
        
        return results
    
    def hyperparameter_grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        num_qubits: int,
        max_iter: int = 100  # Smaller for grid search
    ) -> pd.DataFrame:
        """Grid search over hyperparameters."""
        param_grid = {
            'ansatz_type': ['RealAmplitudes', 'EfficientSU2'],
            'ansatz_reps': [2, 3, 4],
            'feature_map_reps': [1, 2, 3],
            'optimizer': ['COBYLA', 'SPSA'],
        }
        
        results = []
        
        for params in ParameterGrid(param_grid):
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {params}")
            logger.info(f"{'='*60}")
            
            # Build feature map
            feature_map = ZZFeatureMap(
                feature_dimension=num_qubits,
                reps=params['feature_map_reps'],
                entanglement='linear'
            )
            
            # Build ansatz
            if params['ansatz_type'] == 'RealAmplitudes':
                ansatz = RealAmplitudes(num_qubits=num_qubits, reps=params['ansatz_reps'])
            else:
                ansatz = EfficientSU2(num_qubits=num_qubits, reps=params['ansatz_reps'])
            
            # Build optimizer
            if params['optimizer'] == 'COBYLA':
                optimizer = COBYLA(maxiter=max_iter)
            else:
                optimizer = SPSA(maxiter=max_iter)
            
            # Train
            vqc = VQC(
                sampler=Sampler(),
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer
            )
            
            start_time = time.time()
            vqc.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Evaluate
            y_pred_proba = vqc.predict_proba(X_test)[:, 1]
            test_pr_auc = average_precision_score(y_test, y_pred_proba)
            
            y_train_proba = vqc.predict_proba(X_train)[:, 1]
            train_pr_auc = average_precision_score(y_train, y_train_proba)
            
            results.append({
                **params,
                'train_pr_auc': train_pr_auc,
                'test_pr_auc': test_pr_auc,
                'train_time': train_time,
                'num_parameters': ansatz.num_parameters
            })
            
            logger.info(f"Test PR-AUC: {test_pr_auc:.4f}")
        
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('test_pr_auc', ascending=False)
        
        return df_results
    
    def plot_loss_curves(self, results: Dict[str, Dict], save_path: Optional[str] = None):
        """Plot loss curves for different optimizers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name, data in results.items():
            if 'loss_curve' in data and data['loss_curve']['iterations']:
                iterations = data['loss_curve']['iterations']
                losses = data['loss_curve']['losses']
                ax.plot(iterations, losses, label=name, marker='o', markersize=3)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('VQC Training Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved loss curve plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = self._to_json_serializable(value)
            elif isinstance(value, (np.ndarray, np.generic)):
                json_results[key] = value.tolist() if hasattr(value, 'tolist') else float(value)
            else:
                json_results[key] = value
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
    
    def _to_json_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        else:
            return obj


def main():
    parser = argparse.ArgumentParser(description="VQC Optimization Analysis")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=300)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--qml_features", type=str, default="diff", choices=["diff", "hadamard", "both"])
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "optimizers", "ansatzes", "grid_search", "loss_curve"])
    parser.add_argument("--results_dir", type=str, default="results/vqc_analysis")
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
    if not embedder.load_saved_embeddings(expected_dim=args.embedding_dim):
        logger.info("Training new embeddings...")
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
    else:
        embedder.reduce_to_qml_dim()
    
    # Prepare features
    X_train_qml = embedder.prepare_link_features_qml(train_df, mode=args.qml_features)
    X_test_qml = embedder.prepare_link_features_qml(test_df, mode=args.qml_features)
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    logger.info(f"Train set: {X_train_qml.shape}")
    logger.info(f"Test set: {X_test_qml.shape}")
    
    # Initialize analyzer
    analyzer = VQCOptimizationAnalyzer(
        results_dir=args.results_dir,
        random_state=args.random_state
    )
    
    # Build default feature map and ansatz
    feature_map = ZZFeatureMap(
        feature_dimension=args.qml_dim,
        reps=2,
        entanglement='linear'
    )
    ansatz = RealAmplitudes(num_qubits=args.qml_dim, reps=3)
    optimizer = COBYLA(maxiter=args.max_iter)
    
    # Run experiments
    all_results = {}
    
    if args.experiment in ["all", "optimizers"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 1: Optimizer Comparison")
        logger.info("="*60)
        opt_results = analyzer.compare_optimizers(
            X_train_qml, y_train, X_test_qml, y_test,
            feature_map, ansatz, max_iter=args.max_iter
        )
        all_results['optimizer_comparison'] = opt_results
        analyzer.save_results(opt_results, "optimizer_comparison.json")
        
        # Plot loss curves
        analyzer.plot_loss_curves(
            opt_results,
            save_path=str(analyzer.results_dir / "loss_curves_optimizers.png")
        )
    
    if args.experiment in ["all", "ansatzes"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 2: Ansatz Architecture Comparison")
        logger.info("="*60)
        ansatz_results = analyzer.compare_ansatzes(
            X_train_qml, y_train, X_test_qml, y_test,
            feature_map, args.qml_dim, optimizer, max_iter=args.max_iter
        )
        all_results['ansatz_comparison'] = ansatz_results
        analyzer.save_results(ansatz_results, "ansatz_comparison.json")
    
    if args.experiment in ["all", "grid_search"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 3: Hyperparameter Grid Search")
        logger.info("="*60)
        grid_results = analyzer.hyperparameter_grid_search(
            X_train_qml, y_train, X_test_qml, y_test,
            args.qml_dim, max_iter=args.max_iter
        )
        all_results['grid_search'] = grid_results.to_dict('records')
        grid_results.to_csv(analyzer.results_dir / "grid_search_results.csv", index=False)
        logger.info("\nTop 5 configurations:")
        logger.info(grid_results.head().to_string())
    
    if args.experiment in ["all", "loss_curve"]:
        logger.info("\n" + "="*60)
        logger.info("EXPERIMENT 4: Detailed Loss Curve Tracking")
        logger.info("="*60)
        vqc, tracker = analyzer.track_loss_curve(
            X_train_qml, y_train, feature_map, ansatz, optimizer, max_iter=args.max_iter
        )
        
        # Evaluate
        y_pred_proba = vqc.predict_proba(X_test_qml)[:, 1]
        test_pr_auc = average_precision_score(y_test, y_pred_proba)
        
        loss_curve_data = {
            'iterations': tracker.iterations,
            'losses': tracker.losses,
            'test_pr_auc': test_pr_auc
        }
        all_results['loss_curve'] = loss_curve_data
        analyzer.save_results(loss_curve_data, "loss_curve.json")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(tracker.iterations, tracker.losses, 'b-', marker='o', markersize=3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title(f'VQC Training Loss Curve (Test PR-AUC: {test_pr_auc:.4f})')
        ax.grid(True, alpha=0.3)
        plt.savefig(analyzer.results_dir / "loss_curve_detailed.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Save summary
    analyzer.save_results(all_results, "all_results.json")
    
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {analyzer.results_dir}")


if __name__ == "__main__":
    main()

