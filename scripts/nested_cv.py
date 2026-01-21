#!/usr/bin/env python3
"""Nested cross-validation for unbiased performance estimation"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.linear_model import LogisticRegression

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor

def nested_cv(
    X, y,
    model,
    param_grid: dict,
    outer_cv: int = 5,
    inner_cv: int = 3,
    random_state: int = 42
) -> dict:
    """
    Perform nested cross-validation.
    
    Outer loop: Unbiased performance estimation
    Inner loop: Hyperparameter tuning
    """
    outer_scores = []
    best_params_per_fold = []
    
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=random_state)
    inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y)):
        print(f"\n=== Outer Fold {fold+1}/{outer_cv} ===")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV: Hyperparameter tuning
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=inner_cv_splitter,
            scoring=pr_auc_scorer,
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_params_per_fold.append(grid_search.best_params_)
        
        # Outer loop: Unbiased evaluation
        best_model = grid_search.best_estimator_
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        score = average_precision_score(y_test, y_pred_proba)
        outer_scores.append(score)
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Test PR-AUC: {score:.4f}")
    
    # Report unbiased estimate
    print(f"\n{'='*60}")
    print("NESTED CV RESULTS")
    print(f"{'='*60}")
    print(f"Mean PR-AUC: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    print(f"95% CI: [{np.percentile(outer_scores, 2.5):.4f}, {np.percentile(outer_scores, 97.5):.4f}]")
    
    return {
        'outer_scores': outer_scores,
        'mean': float(np.mean(outer_scores)),
        'std': float(np.std(outer_scores)),
        'ci_low': float(np.percentile(outer_scores, 2.5)),
        'ci_high': float(np.percentile(outer_scores, 97.5)),
        'best_params_per_fold': best_params_per_fold
    }

def main():
    parser = argparse.ArgumentParser(description="Nested cross-validation")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--model", type=str, default="LogisticRegression",
                       choices=["LogisticRegression", "SVM", "RandomForest"])
    parser.add_argument("--outer_cv", type=int, default=5)
    parser.add_argument("--inner_cv", type=int, default=3)
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
    
    # Combine train and test for nested CV
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
    
    print(f"Full dataset: {X.shape}")
    
    # Create model and param grid
    if args.model == "LogisticRegression":
        model = LogisticRegression(class_weight='balanced', random_state=args.random_state, max_iter=1000)
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    else:
        # For other models, use ClassicalLinkPredictor interface
        print(f"Warning: Nested CV for {args.model} not fully implemented. Using LogisticRegression.")
        model = LogisticRegression(class_weight='balanced', random_state=args.random_state, max_iter=1000)
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    
    # Run nested CV
    results = nested_cv(
        X, y, model, param_grid,
        outer_cv=args.outer_cv,
        inner_cv=args.inner_cv,
        random_state=args.random_state
    )
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"nested_cv_{args.model}_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

