#!/usr/bin/env python3
"""Compare different classical ML models via cross-validation"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder

def main():
    parser = argparse.ArgumentParser(description="Compare classical ML models")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
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
    
    # Combine for CV
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Generate embeddings
    print("Generating embeddings...")
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=5)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
    
    # Prepare features
    X = embedder.prepare_link_features(full_df, reduced=False)
    y = full_df["label"].values
    
    valid = ~np.isnan(X).any(axis=1)
    X, y = X[valid], y[valid]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Dataset: {X_scaled.shape}")
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', random_state=args.random_state, max_iter=1000
        ),
        'RidgeClassifier': RidgeClassifier(
            class_weight='balanced', random_state=args.random_state
        ),
        'SVM-Linear': SVC(
            kernel='linear', class_weight='balanced', probability=True,
            random_state=args.random_state
        ),
        'SVM-RBF': SVC(
            kernel='rbf', class_weight='balanced', probability=True,
            random_state=args.random_state
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=args.random_state, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100, random_state=args.random_state
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=args.random_state, early_stopping=True
        )
    }
    
    # PR-AUC scorer
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
    cv = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    results = []
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    
    for name, model in models.items():
        print(f"Testing {name}...")
        
        try:
            # Cross-validation
            scores = cross_val_score(
                model, X_scaled, y,
                cv=cv, scoring=pr_auc_scorer, n_jobs=-1
            )
            
            mean_score = scores.mean()
            std_score = scores.std()
            
            result = {
                'model': name,
                'cv_mean_pr_auc': float(mean_score),
                'cv_std_pr_auc': float(std_score),
                'cv_scores': scores.tolist(),
                'status': 'success'
            }
            
            print(f"  CV PR-AUC: {mean_score:.4f} ± {std_score:.4f}")
            results.append(result)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append({
                'model': name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        best = max(successful, key=lambda x: x['cv_mean_pr_auc'])
        print(f"Best model: {best['model']} (PR-AUC: {best['cv_mean_pr_auc']:.4f} ± {best['cv_std_pr_auc']:.4f})")
        
        print("\nAll models (sorted by PR-AUC):")
        for r in sorted(successful, key=lambda x: x['cv_mean_pr_auc'], reverse=True):
            print(f"  {r['model']:20s}: {r['cv_mean_pr_auc']:.4f} ± {r['cv_std_pr_auc']:.4f}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"model_comparison_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

