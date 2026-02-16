#!/usr/bin/env python3
"""
Embedding diversity analysis script.

Computes diversity metrics for embeddings used in QML training:
- % unique embeddings in train heads/tails
- Average pairwise cosine similarity
- Embedding coverage (fraction of entities with embeddings)

Outputs a JSON/CSV that can be correlated with quantum PR-AUC in the dashboard.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def compute_diversity_metrics(
    embeddings: np.ndarray,
    entity_ids: list,
    head_ids: list,
    tail_ids: list,
) -> Dict[str, float]:
    """
    Compute diversity metrics for embeddings used in training.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix (n_entities x dim)
    entity_ids : list
        List of entity IDs corresponding to embedding rows
    head_ids : list
        List of head entity IDs from training data
    tail_ids : list
        List of tail entity IDs from training data
        
    Returns
    -------
    dict
        Diversity metrics including unique fraction, coverage, and similarity stats
    """
    entity_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    
    # Count unique heads and tails
    unique_heads = len(set(head_ids))
    unique_tails = len(set(tail_ids))
    total_heads = len(head_ids)
    total_tails = len(tail_ids)
    
    head_unique_frac = unique_heads / total_heads if total_heads > 0 else 0.0
    tail_unique_frac = unique_tails / total_tails if total_tails > 0 else 0.0
    
    # Coverage: fraction of training entities that have embeddings
    all_train_entities = set(head_ids) | set(tail_ids)
    covered = sum(1 for e in all_train_entities if e in entity_to_idx)
    coverage = covered / len(all_train_entities) if all_train_entities else 0.0
    
    # Pairwise similarity (sample if too large)
    train_indices = [entity_to_idx[e] for e in all_train_entities if e in entity_to_idx]
    if len(train_indices) > 0:
        train_embs = embeddings[train_indices]
        # Normalize for cosine similarity
        norms = np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8
        train_embs_norm = train_embs / norms
        
        # Sample pairs if too many
        n_samples = min(1000, len(train_indices))
        if len(train_indices) > n_samples:
            sample_idx = np.random.choice(len(train_indices), n_samples, replace=False)
            sample_embs = train_embs_norm[sample_idx]
        else:
            sample_embs = train_embs_norm
        
        # Pairwise cosine similarities
        sim_matrix = sample_embs @ sample_embs.T
        # Get upper triangle (exclude diagonal)
        upper_tri = np.triu_indices(len(sample_embs), k=1)
        sims = sim_matrix[upper_tri]
        
        mean_sim = float(np.mean(sims))
        std_sim = float(np.std(sims))
        min_sim = float(np.min(sims))
        max_sim = float(np.max(sims))
    else:
        mean_sim = std_sim = min_sim = max_sim = 0.0
    
    return {
        "unique_heads": unique_heads,
        "total_heads": total_heads,
        "head_unique_fraction": head_unique_frac,
        "unique_tails": unique_tails,
        "total_tails": total_tails,
        "tail_unique_fraction": tail_unique_frac,
        "entity_coverage": coverage,
        "mean_pairwise_similarity": mean_sim,
        "std_pairwise_similarity": std_sim,
        "min_pairwise_similarity": min_sim,
        "max_pairwise_similarity": max_sim,
    }


def load_train_data(results_dir: Path) -> Optional[pd.DataFrame]:
    """Load training data to get head/tail IDs."""
    # Try to find split data
    for name in ["train_df.csv", "train_split.csv"]:
        path = results_dir / name
        if path.exists():
            return pd.read_csv(path)
    
    # Try to load from predictions (has split column)
    pred_path = results_dir / "predictions_latest.csv"
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        if "split" in df.columns:
            return df[df["split"] == "train"]
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Compute embedding diversity metrics")
    parser.add_argument("--embeddings_path", type=str, default="",
                        help="Path to embeddings NPZ/NPY file. If empty, tries data/*.npy")
    parser.add_argument("--entity_ids_path", type=str, default="",
                        help="Path to entity IDs JSON/TXT file")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Results directory with training data")
    parser.add_argument("--output", type=str, default="",
                        help="Output path for diversity report JSON")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load embeddings
    embeddings = None
    entity_ids = None
    
    if args.embeddings_path:
        emb_path = Path(args.embeddings_path)
        if emb_path.suffix == ".npz":
            data = np.load(emb_path)
            embeddings = data["embeddings"] if "embeddings" in data else data[list(data.keys())[0]]
        else:
            embeddings = np.load(emb_path)
    else:
        # Try default locations
        for pattern in ["data/*embeddings*.npy", "data/*embedding*.npz"]:
            from glob import glob
            matches = list(glob(str(PROJECT_ROOT / pattern)))
            if matches:
                if matches[0].endswith(".npz"):
                    data = np.load(matches[0])
                    embeddings = data["embeddings"] if "embeddings" in data else data[list(data.keys())[0]]
                else:
                    embeddings = np.load(matches[0])
                print(f"Loaded embeddings from {matches[0]}")
                break
    
    if embeddings is None:
        print("No embeddings found. Provide --embeddings_path.")
        return 1
    
    # Load entity IDs
    if args.entity_ids_path:
        ids_path = Path(args.entity_ids_path)
        if ids_path.suffix == ".json":
            with open(ids_path) as f:
                entity_ids = json.load(f)
        else:
            entity_ids = ids_path.read_text().strip().split("\n")
    else:
        # Try default locations
        for pattern in ["data/*entity_ids*.json", "data/*ids*.txt"]:
            from glob import glob
            matches = list(glob(str(PROJECT_ROOT / pattern)))
            if matches:
                if matches[0].endswith(".json"):
                    with open(matches[0]) as f:
                        entity_ids = json.load(f)
                else:
                    entity_ids = Path(matches[0]).read_text().strip().split("\n")
                print(f"Loaded entity IDs from {matches[0]}")
                break
    
    if entity_ids is None:
        # Use indices as IDs
        entity_ids = [str(i) for i in range(len(embeddings))]
        print("No entity IDs found; using indices.")
    
    # Load training data
    train_df = load_train_data(results_dir)
    if train_df is None:
        print("No training data found. Cannot compute diversity metrics.")
        return 1
    
    # Get head/tail columns
    head_col = "source" if "source" in train_df.columns else "head"
    tail_col = "target" if "target" in train_df.columns else "tail"
    
    if head_col not in train_df.columns or tail_col not in train_df.columns:
        print(f"Training data missing head/tail columns. Found: {list(train_df.columns)}")
        return 1
    
    head_ids = train_df[head_col].astype(str).tolist()
    tail_ids = train_df[tail_col].astype(str).tolist()
    
    # Compute metrics
    metrics = compute_diversity_metrics(embeddings, entity_ids, head_ids, tail_ids)
    
    # Add metadata
    metrics["embeddings_shape"] = list(embeddings.shape)
    metrics["n_train_samples"] = len(train_df)
    
    print("\n=== Embedding Diversity Report ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # Write output
    output_path = Path(args.output) if args.output else results_dir / "diversity_report.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote report to {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
