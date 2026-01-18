#!/usr/bin/env python3
"""Compare different optimizers for VQC training"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from quantum_layer.qml_model import QMLLinkPredictor
from sklearn.metrics import average_precision_score

def compare_optimizers(
    X_train, y_train, X_test, y_test,
    optimizers: list,
    qml_config_base: dict,
    results_dir: str
) -> dict:
    """Compare multiple optimizers."""
    results = {}
    
    for opt_name in optimizers:
        print(f"\n{'='*60}")
        print(f"Testing optimizer: {opt_name}")
        print(f"{'='*60}")
        
        # Track losses
        loss_history = []
        def loss_callback(nfev, parameters, loss, step):
            loss_val = float(loss) if hasattr(loss, '__float__') else float(loss[0]) if isinstance(loss, (list, tuple)) else loss
            loss_history.append({
                'nfev': int(nfev),
                'loss': loss_val,
                'step': int(step) if step is not None else nfev
            })
            if len(loss_history) % 10 == 0:
                print(f"  Iteration {nfev}: loss={loss_val:.4f}")
        
        qml_config = {**qml_config_base, 'optimizer': opt_name}
        
        try:
            predictor = QMLLinkPredictor(callback=loss_callback, **qml_config)
            predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = predictor.predict(X_test)
            y_proba = predictor.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, y_proba)
            
            results[opt_name] = {
                'pr_auc': float(pr_auc),
                'loss_history': loss_history,
                'final_loss': loss_history[-1]['loss'] if loss_history else None,
                'num_iterations': len(loss_history),
                'status': 'success'
            }
            print(f"✅ {opt_name}: PR-AUC = {pr_auc:.4f}")
        except Exception as e:
            print(f"❌ {opt_name} failed: {e}")
            results[opt_name] = {
                'status': 'failed',
                'error': str(e)
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare VQC optimizers")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--ansatz", type=str, default="RealAmplitudes", choices=["RealAmplitudes", "EfficientSU2", "TwoLocal"])
    parser.add_argument("--ansatz_reps", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=200)
    parser.add_argument("--optimizers", type=str, nargs='+', 
                       default=["COBYLA", "SPSA"],
                       help="List of optimizers to test")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(
        df, relation_type=args.relation, max_entities=args.max_entities
    )
    train_df, test_df = prepare_link_prediction_dataset(
        task_edges, random_state=args.random_state
    )
    
    # Generate embeddings
    print("Generating embeddings...")
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=args.qml_dim)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
    
    # Prepare features
    X_train = embedder.prepare_link_features_qml(train_df)
    y_train = train_df["label"].values
    X_test = embedder.prepare_link_features_qml(test_df)
    y_test = test_df["label"].values
    
    # Remove invalid samples
    valid_train = ~np.isnan(X_train).any(axis=1)
    valid_test = ~np.isnan(X_test).any(axis=1)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Base config
    qml_config_base = {
        "model_type": "VQC",
        "encoding_method": "feature_map",
        "num_qubits": args.qml_dim,
        "feature_map_type": "ZZ",
        "feature_map_reps": 2,
        "ansatz_type": args.ansatz,
        "ansatz_reps": args.ansatz_reps,
        "max_iter": args.max_iter,
        "random_state": args.random_state
    }
    
    # Compare optimizers
    results = compare_optimizers(
        X_train, y_train, X_test, y_test,
        args.optimizers, qml_config_base, args.results_dir
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("OPTIMIZER COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    successful = {k: v for k, v in results.items() if v.get('status') == 'success'}
    if successful:
        best = max(successful.items(), key=lambda x: x[1]['pr_auc'])
        print(f"\nBest optimizer: {best[0]} (PR-AUC: {best[1]['pr_auc']:.4f})")
        
        print("\nAll results:")
        for opt_name, res in sorted(successful.items(), key=lambda x: x[1]['pr_auc'], reverse=True):
            final_loss_str = f"{res['final_loss']:.4f}" if res['final_loss'] is not None else "N/A"
            print(f"  {opt_name:15s}: PR-AUC = {res['pr_auc']:.4f}, "
                  f"final_loss = {final_loss_str}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"optimizer_comparison_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

