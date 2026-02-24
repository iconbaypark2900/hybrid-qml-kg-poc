#!/usr/bin/env python3
"""Compare hard negative mining strategies"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
    get_negative_samples,
    get_hard_negatives_similarity,
    get_hard_negatives_adversarial,
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor
from sklearn.metrics import average_precision_score

def main():
    parser = argparse.ArgumentParser(description="Compare hard negative strategies")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
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
    
    # Split positives
    from sklearn.model_selection import train_test_split
    pos_train, pos_test = train_test_split(
        task_edges, test_size=0.2, random_state=args.random_state
    )
    
    # Generate embeddings
    print("Generating embeddings...")
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=5)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
    
    results = []
    
    # Strategy 1: Random negatives
    print("\n[1/3] Testing random negatives...")
    try:
        neg_train_random = get_negative_samples(pos_train, random_state=args.random_state)
        neg_test = get_negative_samples(pos_test, random_state=args.random_state + 1)
        
        train_df_random = pd.concat([pos_train, neg_train_random], ignore_index=True)
        train_df_random['label'] = [1] * len(pos_train) + [0] * len(neg_train_random)
        test_df = pd.concat([pos_test, neg_test], ignore_index=True)
        test_df['label'] = [1] * len(pos_test) + [0] * len(neg_test)
        
        predictor = ClassicalLinkPredictor(random_state=args.random_state)
        predictor.train(train_df_random, embedder, test_df)
        
        results.append({
            'strategy': 'random',
            'test_pr_auc': predictor.metrics.get('test_pr_auc', 0.0),
            'status': 'success'
        })
        print(f"  Test PR-AUC: {predictor.metrics.get('test_pr_auc', 0.0):.4f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results.append({'strategy': 'random', 'status': 'failed', 'error': str(e)})
    
    # Strategy 2: Hard negatives (similarity)
    print("\n[2/3] Testing hard negatives (similarity)...")
    try:
        neg_train_hard = get_hard_negatives_similarity(
            pos_train, embedder, random_state=args.random_state
        )
        train_df_hard = pd.concat([pos_train, neg_train_hard], ignore_index=True)
        train_df_hard['label'] = [1] * len(pos_train) + [0] * len(neg_train_hard)
        
        predictor = ClassicalLinkPredictor(random_state=args.random_state)
        predictor.train(train_df_hard, embedder, test_df)
        
        results.append({
            'strategy': 'hard_similarity',
            'test_pr_auc': predictor.metrics.get('test_pr_auc', 0.0),
            'status': 'success'
        })
        print(f"  Test PR-AUC: {predictor.metrics.get('test_pr_auc', 0.0):.4f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results.append({'strategy': 'hard_similarity', 'status': 'failed', 'error': str(e)})
    
    # Strategy 3: Hard negatives (adversarial) - requires trained model
    print("\n[3/3] Testing hard negatives (adversarial)...")
    try:
        # First train a model on random negatives
        neg_train_init = get_negative_samples(pos_train, random_state=args.random_state)
        train_df_init = pd.concat([pos_train, neg_train_init], ignore_index=True)
        train_df_init['label'] = [1] * len(pos_train) + [0] * len(neg_train_init)
        
        init_predictor = ClassicalLinkPredictor(random_state=args.random_state)
        init_predictor.train(train_df_init, embedder, None)
        
        # Generate adversarial hard negatives
        neg_train_adv = get_hard_negatives_adversarial(
            pos_train, init_predictor.model, embedder, random_state=args.random_state
        )
        train_df_adv = pd.concat([pos_train, neg_train_adv], ignore_index=True)
        train_df_adv['label'] = [1] * len(pos_train) + [0] * len(neg_train_adv)
        
        predictor = ClassicalLinkPredictor(random_state=args.random_state)
        predictor.train(train_df_adv, embedder, test_df)
        
        results.append({
            'strategy': 'hard_adversarial',
            'test_pr_auc': predictor.metrics.get('test_pr_auc', 0.0),
            'status': 'success'
        })
        print(f"  Test PR-AUC: {predictor.metrics.get('test_pr_auc', 0.0):.4f}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        results.append({'strategy': 'hard_adversarial', 'status': 'failed', 'error': str(e)})
    
    # Summary
    print(f"\n{'='*60}")
    print("HARD NEGATIVES COMPARISON")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        best = max(successful, key=lambda x: x['test_pr_auc'])
        print(f"Best strategy: {best['strategy']} (Test PR-AUC: {best['test_pr_auc']:.4f})")
        
        print("\nAll strategies:")
        for r in sorted(successful, key=lambda x: x['test_pr_auc'], reverse=True):
            print(f"  {r['strategy']:20s}: {r['test_pr_auc']:.4f}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"hard_negatives_experiment_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

