#!/usr/bin/env python3
"""Compare different embedding algorithms using PyKEEN"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime

# Check if PyKEEN is available
try:
    from pykeen.pipeline import pipeline
    PYKEEN_AVAILABLE = True
except ImportError:
    PYKEEN_AVAILABLE = False
    print("⚠️ PyKEEN not available. Install with: pip install pykeen")
    print("Skipping embedding comparison.")

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder
from classical_baseline.train_baseline import ClassicalLinkPredictor
from sklearn.metrics import average_precision_score

def main():
    if not PYKEEN_AVAILABLE:
        return
    
    parser = argparse.ArgumentParser(description="Compare embedding algorithms")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--algorithms", type=str, nargs='+',
                       default=["TransE", "DistMult", "ComplEx"],
                       help="Embedding algorithms to test")
    parser.add_argument("--epochs", type=int, default=50)
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
    
    results = []
    
    for algo_name in args.algorithms:
        print(f"\n{'='*60}")
        print(f"Testing {algo_name}")
        print(f"{'='*60}")
        
        try:
            # Train embeddings with PyKEEN
            print(f"  Training {algo_name} embeddings...")
            
            # Prepare triples for PyKEEN
            triples = []
            for _, row in task_edges.iterrows():
                h = row['source']
                r = row.get('metaedge', 'treats')  # Use relation if available
                t = row['target']
                triples.append((h, r, t))
            
            # Write to temp file for PyKEEN
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
                for h, r, t in triples:
                    f.write(f"{h}\t{r}\t{t}\n")
                temp_path = f.name
            
            from pykeen.datasets.base import PathDataset
            dataset = PathDataset(
                training_path=temp_path,
                testing_path=temp_path,
                validation_path=temp_path
            )
            
            # Run PyKEEN pipeline
            pykeen_result = pipeline(
                dataset=dataset,
                model=algo_name,
                model_kwargs=dict(embedding_dim=args.embedding_dim),
                training_kwargs=dict(num_epochs=args.epochs, batch_size=1024),
                optimizer="adam",
                random_seed=args.random_state
            )
            
            # Extract embeddings
            # PyKEEN stores embeddings in model.entity_representations[0]
            model = pykeen_result.model
            entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()
            entity_to_id = dataset.training.entity_to_id
            
            # Create embedder-like interface
            class PyKEEHEmbedder:
                def __init__(self, embeddings, entity_map):
                    self.embeddings = embeddings
                    self.entity_map = entity_map
                
                def get_embedding(self, entity_id):
                    if entity_id in self.entity_map:
                        idx = self.entity_map[entity_id]
                        return self.embeddings[idx]
                    return np.zeros(self.embeddings.shape[1])
                
                def prepare_link_features(self, df):
                    features = []
                    for _, row in df.iterrows():
                        h = str(row.get('source', row.get('source_id', '')))
                        t = str(row.get('target', row.get('target_id', '')))
                        h_emb = self.get_embedding(h)
                        t_emb = self.get_embedding(t)
                        diff = np.abs(h_emb - t_emb)
                        prod = h_emb * t_emb
                        features.append(np.concatenate([h_emb, t_emb, diff, prod]))
                    return np.array(features)
            
            embedder = PyKEEHEmbedder(entity_embeddings, entity_to_id)
            
            # Evaluate on link prediction
            print("  Evaluating on link prediction...")
            X_train = embedder.prepare_link_features(train_df)
            X_test = embedder.prepare_link_features(test_df)
            y_train = train_df["label"].values
            y_test = test_df["label"].values
            
            # Train classifier
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            classifier = ClassicalLinkPredictor(random_state=args.random_state)
            classifier.model.fit(X_train_scaled, y_train)
            
            y_proba_train = classifier.model.predict_proba(X_train_scaled)[:, 1]
            y_proba_test = classifier.model.predict_proba(X_test_scaled)[:, 1]
            
            train_pr_auc = average_precision_score(y_train, y_proba_train)
            test_pr_auc = average_precision_score(y_test, y_proba_test)
            
            # Get PyKEEN metrics
            hits_at_10 = pykeen_result.metric_results.get('hits@10', 0.0)
            
            result = {
                'algorithm': algo_name,
                'hits_at_10': float(hits_at_10),
                'train_pr_auc': float(train_pr_auc),
                'test_pr_auc': float(test_pr_auc),
                'num_entities': len(entity_to_id),
                'status': 'success'
            }
            
            print(f"    Hits@10: {hits_at_10:.4f}")
            print(f"    Test PR-AUC: {test_pr_auc:.4f}")
            
            results.append(result)
            
            # Cleanup
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'algorithm': algo_name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("EMBEDDING ALGORITHM COMPARISON")
    print(f"{'='*60}\n")
    
    successful = [r for r in results if r.get('status') == 'success']
    if successful:
        best = max(successful, key=lambda x: x['test_pr_auc'])
        print(f"Best algorithm: {best['algorithm']} (Test PR-AUC: {best['test_pr_auc']:.4f})")
        
        print("\nAll algorithms (sorted by Test PR-AUC):")
        for r in sorted(successful, key=lambda x: x['test_pr_auc'], reverse=True):
            print(f"  {r['algorithm']:15s}: PR-AUC={r['test_pr_auc']:.4f}, Hits@10={r['hits_at_10']:.4f}")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"embedding_comparison_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

