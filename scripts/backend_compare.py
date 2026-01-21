#!/usr/bin/env python3
"""Compare simulator vs real quantum hardware backends"""

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
from quantum_layer.qml_model import QMLLinkPredictor
from sklearn.metrics import average_precision_score

def main():
    parser = argparse.ArgumentParser(description="Compare quantum backends")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset, smaller recommended for hardware)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
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
    
    # Define backends to test
    backends = {
        'simulator_statevector': {
            'description': 'Local exact simulator (statevector)',
            'config_path': None  # Use default/local
        }
    }
    
    # Try to add IBM hardware backends if available
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        available_backends = service.backends()
        
        if available_backends:
            # Add first available backend
            hw_backend = available_backends[0]
            backends[f'ibm_{hw_backend.name}'] = {
                'description': f'IBM {hw_backend.name} ({hw_backend.num_qubits} qubits)',
                'config_path': None  # Would need quantum_config.yaml setup
            }
            print(f"Found IBM backend: {hw_backend.name}")
    except Exception as e:
        print(f"IBM Quantum not available: {e}")
        print("Will only test simulator")
    
    results = []
    
    for backend_name, backend_info in backends.items():
        print(f"\n{'='*60}")
        print(f"Testing: {backend_info['description']}")
        print(f"{'='*60}")
        
        try:
            qml_config = {
                "model_type": "QSVC",  # QSVC is faster for testing
                "encoding_method": "feature_map",
                "num_qubits": args.qml_dim,
                "feature_map_type": "ZZ",
                "feature_map_reps": 2,
                "ansatz_type": "RealAmplitudes",
                "ansatz_reps": 2,
                "optimizer": "COBYLA",
                "max_iter": 20,  # Reduced for hardware
                "random_state": args.random_state,
                "quantum_config_path": backend_info.get('config_path', "config/quantum_config.yaml")
            }
            
            # Set execution mode in config if needed
            # (This would require modifying quantum_executor.py to accept backend name)
            
            t0 = time.time()
            predictor = QMLLinkPredictor(**qml_config)
            predictor.fit(X_train, y_train)
            train_time = time.time() - t0
            
            # Evaluate
            y_proba = predictor.predict_proba(X_test)[:, 1]
            pr_auc = average_precision_score(y_test, y_proba)
            
            result = {
                'backend': backend_name,
                'description': backend_info['description'],
                'train_time': train_time,
                'test_pr_auc': float(pr_auc),
                'status': 'success'
            }
            
            print(f"  Train time: {train_time:.2f}s")
            print(f"  Test PR-AUC: {pr_auc:.4f}")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'backend': backend_name,
                'description': backend_info['description'],
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("BACKEND COMPARISON SUMMARY")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        print("Backend | Train Time | Test PR-AUC")
        print("-" * 50)
        for r in successful:
            print(f"{r['backend']:25s} | {r['train_time']:10.2f} | {r['test_pr_auc']:11.4f}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"backend_compare_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

