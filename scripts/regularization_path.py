#!/usr/bin/env python3
"""CLI wrapper for regularization path analysis"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import regularization_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run regularization path analysis")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--qml_dim", type=int, default=5)
    parser.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--C_min", type=float, default=-4, help="log10 of minimum C")
    parser.add_argument("--C_max", type=float, default=4, help="log10 of maximum C")
    parser.add_argument("--n_C", type=int, default=20, help="Number of C values")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load data
    print("Loading Hetionet edges...")
    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(df, relation_type=args.relation, max_entities=args.max_entities)
    train_df, test_df = prepare_link_prediction_dataset(task_edges, random_state=args.random_state)

    # Generate embeddings
    print("Generating embeddings...")
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=args.qml_dim)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()

    # Prepare features
    print("Preparing features...")
    X_train = embedder.prepare_link_features(train_df, reduced=False)
    y_train = train_df["label"].values
    X_test = embedder.prepare_link_features(test_df, reduced=False)
    y_test = test_df["label"].values

    # Remove invalid samples
    valid_train = ~np.isnan(X_train).any(axis=1)
    valid_test = ~np.isnan(X_test).any(axis=1)
    X_train, y_train = X_train[valid_train], y_train[valid_train]
    X_test, y_test = X_test[valid_test], y_test[valid_test]

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Compute regularization path
    C_values = np.logspace(args.C_min, args.C_max, args.n_C).tolist()
    results = regularization_path(
        X_train, y_train, X_test, y_test,
        C_values=C_values,
        penalty=args.penalty,
        results_dir=args.results_dir
    )

    print(f"\n✅ Regularization path complete!")
    print(f"Best C: {results['best_C']:.4f}")
    print(f"Best test PR-AUC: {results['best_test_score']:.4f}")

