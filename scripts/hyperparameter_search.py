#!/usr/bin/env python3
"""Grid search over VQC hyperparameters with cross-validation"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import ParameterGrid, StratifiedKFold, cross_val_score
from sklearn.metrics import make_scorer, average_precision_score

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from quantum_layer.qml_model import QMLLinkPredictor

def main():
    parser = argparse.ArgumentParser(description="Grid search over VQC hyperparameters")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--n_splits", type=int, default=5, help="CV folds")
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
    
    # Parameter grid
    param_grid = {
        'ansatz_reps': [2, 3, 4],
        'feature_map_reps': [1, 2, 3],
        'optimizer': ['COBYLA', 'SPSA'],
        'max_iter': [50, 100, 200]
    }
    
    print(f"\nGrid search over {len(list(ParameterGrid(param_grid)))} configurations...")
    
    # PR-AUC scorer
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    results = []
    best_score = -np.inf
    best_params = None
    
    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f"\n[{i+1}/{len(list(ParameterGrid(param_grid)))}] Testing: {params}")
        
        qml_config = {
            "model_type": "VQC",
            "encoding_method": "feature_map",
            "num_qubits": args.qml_dim,
            "feature_map_type": "ZZ",
            "ansatz_type": "RealAmplitudes",
            "random_state": args.random_state,
            **params
        }
        
        try:
            predictor = QMLLinkPredictor(**qml_config)
            
            # Cross-validation
            scores = cross_val_score(
                predictor, X_train, y_train,
                cv=cv, scoring=pr_auc_scorer, n_jobs=1
            )
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            result = {
                **params,
                'cv_mean_pr_auc': float(mean_score),
                'cv_std_pr_auc': float(std_score),
                'cv_scores': scores.tolist(),
                'status': 'success'
            }
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
            
            print(f"  CV PR-AUC: {mean_score:.4f} ± {std_score:.4f}")
            results.append(result)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                **params,
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get('status') == 'success']
    if successful and best_params:
        print(f"\nBest parameters: {best_params}")
        print(f"Best CV PR-AUC: {best_score:.4f}")
        
        # Evaluate best on test set
        print("\nEvaluating best model on test set...")
        best_config = {
            "model_type": "VQC",
            "encoding_method": "feature_map",
            "num_qubits": args.qml_dim,
            "feature_map_type": "ZZ",
            "ansatz_type": "RealAmplitudes",
            "random_state": args.random_state,
            **best_params
        }
        best_predictor = QMLLinkPredictor(**best_config)
        best_predictor.fit(X_train, y_train)
        y_proba = best_predictor.predict_proba(X_test)[:, 1]
        test_pr_auc = average_precision_score(y_test, y_proba)
        print(f"Test PR-AUC: {test_pr_auc:.4f}")
        
        # Update best result
        for r in successful:
            if all(r.get(k) == v for k, v in best_params.items()):
                r['test_pr_auc'] = float(test_pr_auc)
                break
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"hyperparameter_search_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'param_grid': param_grid,
            'best_params': best_params,
            'best_cv_score': float(best_score) if best_score > -np.inf else None,
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

