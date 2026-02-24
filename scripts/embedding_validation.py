#!/usr/bin/env python3
"""Validate embedding quality with similarity metrics and visualizations"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from scipy.stats import spearmanr

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder

def main():
    parser = argparse.ArgumentParser(description="Validate embedding quality")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--reduced", action="store_true", help="Use reduced embeddings")
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
    embedder = HetionetEmbedder(embedding_dim=args.embedding_dim, qml_dim=5)
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)
        embedder.reduce_to_qml_dim()
    
    results = {}
    
    # 1. Link prediction correlation
    print("\n[1/3] Computing embedding similarity vs label correlation...")
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    embedding_sims = []
    true_labels = []
    
    for _, row in all_df.iterrows():
        h = row['source'] if 'source' in row else row.get('source_id', None)
        t = row['target'] if 'target' in row else row.get('target_id', None)
        label = row['label']
        
        if h is None or t is None:
            continue
        
        try:
            h_emb = embedder.get_embedding(str(h), reduced=args.reduced)
            t_emb = embedder.get_embedding(str(t), reduced=args.reduced)
            
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(h_emb.reshape(1, -1), t_emb.reshape(1, -1))[0, 0]
            
            embedding_sims.append(sim)
            true_labels.append(label)
        except Exception as e:
            continue
    
    if embedding_sims:
        corr, pval = spearmanr(embedding_sims, true_labels)
        results['link_prediction_correlation'] = {
            'spearman_rho': float(corr),
            'p_value': float(pval),
            'n_pairs': len(embedding_sims)
        }
        print(f"  Spearman correlation: ρ={corr:.4f}, p={pval:.4f}")
    
    # 2. t-SNE visualization
    print("\n[2/3] Generating t-SNE visualization...")
    try:
        tsne_result = embedder.tsne_visualize(
            reduced=args.reduced,
            results_dir=args.results_dir
        )
        if tsne_result is not None:
            results['tsne'] = {'status': 'success', 'n_components': tsne_result.shape[1]}
            print("  ✓ t-SNE visualization saved")
        else:
            results['tsne'] = {'status': 'skipped'}
    except Exception as e:
        print(f"  ⚠️ t-SNE failed: {e}")
        results['tsne'] = {'status': 'failed', 'error': str(e)}
    
    # 3. Drug class similarity (if we can infer drug classes)
    print("\n[3/3] Computing drug class similarity...")
    # Try to group entities by type prefix
    all_embeddings = embedder.get_all_embeddings(reduced=args.reduced)
    
    # Group by entity type prefix (e.g., "Compound::", "Disease::")
    type_groups = {}
    for eid in all_embeddings.keys():
        if '::' in eid:
            etype = eid.split('::')[0]
            if etype not in type_groups:
                type_groups[etype] = []
            type_groups[etype].append(eid)
    
    if len(type_groups) >= 2:
        # Compute within-type vs between-type similarity
        from sklearn.metrics.pairwise import cosine_similarity
        
        within_sims = []
        between_sims = []
        
        type_list = list(type_groups.keys())
        for i, type1 in enumerate(type_list):
            embs1 = [all_embeddings[eid] for eid in type_groups[type1] if eid in all_embeddings]
            if len(embs1) < 2:
                continue
            
            # Within-type
            for j, emb1 in enumerate(embs1):
                for emb2 in embs1[j+1:]:
                    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
                    within_sims.append(sim)
            
            # Between-type
            for type2 in type_list[i+1:]:
                embs2 = [all_embeddings[eid] for eid in type_groups[type2] if eid in all_embeddings]
                for emb1 in embs1:
                    for emb2 in embs2:
                        sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0, 0]
                        between_sims.append(sim)
        
        if within_sims and between_sims:
            within_mean = float(np.mean(within_sims))
            between_mean = float(np.mean(between_sims))
            difference = within_mean - between_mean
            
            results['type_similarity'] = {
                'within_type_mean': within_mean,
                'between_type_mean': between_mean,
                'difference': difference,
                'within_type_std': float(np.std(within_sims)),
                'between_type_std': float(np.std(between_sims)),
                'num_within_pairs': len(within_sims),
                'num_between_pairs': len(between_sims)
            }
            print(f"  Within-type similarity: {within_mean:.4f} ± {np.std(within_sims):.4f}")
            print(f"  Between-type similarity: {between_mean:.4f} ± {np.std(between_sims):.4f}")
            print(f"  Difference: {difference:.4f}")
            
            if difference > 0.1:
                print("  ✅ Good separation between entity types")
            else:
                print("  ⚠️ Weak separation between entity types")
    
    # Save results
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"embedding_validation_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"\n✅ Saved results → {json_path}")

if __name__ == "__main__":
    main()

