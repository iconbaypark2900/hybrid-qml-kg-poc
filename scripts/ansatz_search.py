#!/usr/bin/env python3
"""Search over ansatz architectures for VQC"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, TwoLocal

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from quantum_layer.qml_model import QMLLinkPredictor
from sklearn.metrics import average_precision_score

def evaluate_ansatz(
    ansatz_name: str,
    reps: int,
    X_train, y_train, X_test, y_test,
    qml_config_base: dict,
    num_qubits: int
) -> dict:
    """Evaluate a single ansatz configuration."""
    print(f"\nTesting {ansatz_name} (reps={reps})...")
    
    # Build ansatz to get metrics
    if ansatz_name == "RealAmplitudes":
        ansatz = RealAmplitudes(num_qubits=num_qubits, reps=reps)
    elif ansatz_name == "EfficientSU2":
        ansatz = EfficientSU2(num_qubits=num_qubits, reps=reps)
    elif ansatz_name == "TwoLocal":
        ansatz = TwoLocal(
            num_qubits=num_qubits,
            rotation_blocks='ry',
            entanglement_blocks='cz',
            reps=reps,
            entanglement='linear'
        )
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_name}")
    
    num_params = ansatz.num_parameters
    depth = ansatz.depth()
    
    # Track losses
    loss_history = []
    def loss_callback(nfev, parameters, loss, step):
        loss_val = float(loss) if hasattr(loss, '__float__') else float(loss[0]) if isinstance(loss, (list, tuple)) else loss
        loss_history.append(loss_val)
    
    qml_config = {**qml_config_base, 'ansatz_type': ansatz_name, 'ansatz_reps': reps}
    
    try:
        predictor = QMLLinkPredictor(callback=loss_callback, **qml_config)
        predictor.fit(X_train, y_train)
        
        y_proba = predictor.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_proba)
        
        return {
            'ansatz': ansatz_name,
            'reps': reps,
            'num_parameters': num_params,
            'depth': depth,
            'pr_auc': float(pr_auc),
            'final_loss': loss_history[-1] if loss_history else None,
            'status': 'success'
        }
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return {
            'ansatz': ansatz_name,
            'reps': reps,
            'num_parameters': num_params,
            'depth': depth,
            'status': 'failed',
            'error': str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Search over ansatz architectures")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--ansatzes", type=str, nargs='+',
                       default=["RealAmplitudes", "EfficientSU2", "TwoLocal"])
    parser.add_argument("--reps_range", type=int, nargs=2, default=[1, 5],
                       help="Range of reps to test [min, max]")
    parser.add_argument("--optimizer", type=str, default="COBYLA")
    parser.add_argument("--max_iter", type=int, default=100)
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
        "optimizer": args.optimizer,
        "max_iter": args.max_iter,
        "random_state": args.random_state
    }
    
    # Search over ansatzes and reps
    results = []
    reps_range = range(args.reps_range[0], args.reps_range[1] + 1)
    
    for ansatz_name in args.ansatzes:
        for reps in reps_range:
            result = evaluate_ansatz(
                ansatz_name, reps, X_train, y_train, X_test, y_test,
                qml_config_base, args.qml_dim
            )
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("ANSATZ SEARCH SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        best = max(successful, key=lambda x: x['pr_auc'])
        print(f"\nBest: {best['ansatz']} (reps={best['reps']})")
        print(f"  PR-AUC: {best['pr_auc']:.4f}")
        print(f"  Parameters: {best['num_parameters']}")
        print(f"  Depth: {best['depth']}")
        
        print("\nTop 5 configurations:")
        for r in sorted(successful, key=lambda x: x['pr_auc'], reverse=True)[:5]:
            print(f"  {r['ansatz']:20s} reps={r['reps']:2d}: "
                  f"PR-AUC={r['pr_auc']:.4f}, params={r['num_parameters']:3d}, depth={r['depth']:3d}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"ansatz_search_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

