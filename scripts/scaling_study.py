#!/usr/bin/env python3
"""Study how performance scales with dataset size"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np
from datetime import datetime

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor
from quantum_layer.qml_trainer import QMLTrainer

def run_single_size(
    N: int,
    relation: str,
    embedding_dim: int,
    qml_dim: int,
    random_state: int
) -> dict:
    """Run pipeline for a single entity count."""
    print(f"\n{'='*60}")
    print(f"Testing with N={N} entities")
    print(f"{'='*60}")
    
    # Load data
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(
        df, relation_type=relation, max_entities=N
    )
    train_df, test_df = prepare_link_prediction_dataset(
        task_edges, random_state=random_state
    )
    
    # Generate embeddings
    embedder = HetionetEmbedder(embedding_dim=embedding_dim, qml_dim=qml_dim)
    if not embedder.load_saved_embeddings():
        t0 = time.time()
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
        embedding_time = time.time() - t0
    else:
        embedding_time = 0.0
    
    result = {
        'N': N,
        'num_train': len(train_df),
        'num_test': len(test_df),
        'embedding_time': embedding_time
    }
    
    # Classical baseline
    try:
        print("  Training classical baseline...")
        t0 = time.time()
        classical_predictor = ClassicalLinkPredictor(random_state=random_state)
        classical_predictor.train(train_df, embedder, test_df)
        classical_time = time.time() - t0
        
        result['classical'] = {
            'train_pr_auc': classical_predictor.metrics.get('train_pr_auc', 0.0),
            'test_pr_auc': classical_predictor.metrics.get('test_pr_auc', 0.0),
            'train_time': classical_time
        }
        print(f"    Test PR-AUC: {result['classical']['test_pr_auc']:.4f}")
    except Exception as e:
        print(f"    ❌ Classical failed: {e}")
        result['classical'] = {'status': 'failed', 'error': str(e)}
    
    # Quantum (only for smaller sizes to save time)
    if N <= 2000:
        try:
            print("  Training quantum model...")
            qml_config = {
                "model_type": "QSVC",  # QSVC is faster than VQC
                "encoding_method": "feature_map",
                "num_qubits": qml_dim,
                "feature_map_type": "ZZ",
                "feature_map_reps": 2,
                "ansatz_type": "RealAmplitudes",
                "ansatz_reps": 2,  # Reduced for speed
                "optimizer": "COBYLA",
                "max_iter": 30,  # Reduced for speed
                "random_state": random_state
            }
            
            t0 = time.time()
            trainer = QMLTrainer(random_state=random_state)
            qml_results = trainer.train_and_evaluate(
                train_df, test_df, embedder, qml_config
            )
            quantum_time = time.time() - t0
            
            result['quantum'] = {
                'test_pr_auc': qml_results.get('quantum', {}).get('pr_auc', 0.0),
                'train_time': quantum_time
            }
            print(f"    Test PR-AUC: {result['quantum']['test_pr_auc']:.4f}")
        except Exception as e:
            print(f"    ❌ Quantum failed: {e}")
            result['quantum'] = {'status': 'failed', 'error': str(e)}
    else:
        result['quantum'] = {'status': 'skipped', 'reason': 'N > 2000'}
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Scaling study")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--entity_sizes", type=int, nargs='+',
                       default=[100, 300, 500, 1000, 2000],
                       help="Entity counts to test")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    results = []
    
    for N in args.entity_sizes:
        result = run_single_size(
            N, args.relation, args.embedding_dim, args.qml_dim, args.random_state
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SCALING STUDY SUMMARY")
    print(f"{'='*60}\n")
    
    print("Entity Count | Train Size | Classical PR-AUC | Quantum PR-AUC | Classical Time | Quantum Time")
    print("-" * 90)
    
    for r in results:
        classical_pr = r.get('classical', {}).get('test_pr_auc', 0.0)
        quantum_pr = r.get('quantum', {}).get('test_pr_auc', 0.0) if r.get('quantum', {}).get('status') != 'failed' else 0.0
        classical_time = r.get('classical', {}).get('train_time', 0.0)
        quantum_time = r.get('quantum', {}).get('train_time', 0.0) if r.get('quantum', {}).get('status') != 'failed' else 0.0
        
        print(f"{r['N']:12d} | {r['num_train']:10d} | {classical_pr:15.4f} | {quantum_pr:13.4f} | {classical_time:14.2f} | {quantum_time:11.2f}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"scaling_study_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")
    
    # Optional: Plot scaling curves
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        entity_counts = [r['N'] for r in results]
        classical_prs = [r.get('classical', {}).get('test_pr_auc', 0.0) for r in results]
        quantum_prs = [r.get('quantum', {}).get('test_pr_auc', 0.0) if r.get('quantum', {}).get('status') != 'failed' else None for r in results]
        quantum_prs = [p for p in quantum_prs if p is not None]
        quantum_counts = [r['N'] for r in results if r.get('quantum', {}).get('status') != 'failed']
        
        plt.figure(figsize=(10, 6))
        plt.plot(entity_counts, classical_prs, 'o-', label='Classical', linewidth=2)
        if quantum_prs:
            plt.plot(quantum_counts, quantum_prs, 's-', label='Quantum', linewidth=2)
        plt.xlabel('Number of Entities', fontsize=12)
        plt.ylabel('Test PR-AUC', fontsize=12)
        plt.title('Scaling Study: Performance vs Dataset Size', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(args.results_dir, f"scaling_study_{stamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved plot → {plot_path}")
    except ImportError:
        pass

if __name__ == "__main__":
    main()

