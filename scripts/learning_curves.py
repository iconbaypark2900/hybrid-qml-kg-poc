#!/usr/bin/env python3
"""Generate learning curves to diagnose bias vs variance"""

import sys
import os
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
from datetime import datetime
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

from kg_layer.kg_loader import (
    load_hetionet_edges,
    extract_task_edges,
    prepare_link_prediction_dataset,
)
from kg_layer.kg_embedder import HetionetEmbedder

def main():
    parser = argparse.ArgumentParser(description="Generate learning curves")
    parser.add_argument("--relation", type=str, default="CtD")
    parser.add_argument("--max_entities", type=int, default=None,
                        help="Limit number of entities (None = use full dataset)")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--model", type=str, default="LogisticRegression",
                       choices=["LogisticRegression", "SVM", "RandomForest"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--train_sizes", type=int, nargs='+',
                       default=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                       help="Training set sizes (percentages or absolute)")
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
    
    # Combine for learning curves
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
    
    # Create model
    if args.model == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(class_weight='balanced', random_state=args.random_state, max_iter=1000)
    elif args.model == "SVM":
        from sklearn.svm import SVC
        model = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=args.random_state)
    elif args.model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=args.random_state)
    
    # Convert train_sizes to absolute numbers if they're percentages
    train_sizes = args.train_sizes
    if max(train_sizes) <= 100:
        # Assume percentages
        train_sizes = [int(len(X_scaled) * ts / 100) for ts in train_sizes]
    
    train_sizes = sorted(set(train_sizes))
    train_sizes = [ts for ts in train_sizes if 5 <= ts <= len(X_scaled)]
    
    print(f"\nGenerating learning curves with sizes: {train_sizes}")
    
    # Generate learning curves
    from sklearn.metrics import make_scorer, average_precision_score
    pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_scaled, y,
        train_sizes=train_sizes,
        cv=args.n_splits,
        scoring=pr_auc_scorer,
        n_jobs=-1,
        random_state=args.random_state
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Save results
    results = {
        'train_sizes': train_sizes_abs.tolist(),
        'train_mean': train_mean.tolist(),
        'train_std': train_std.tolist(),
        'val_mean': val_mean.tolist(),
        'val_std': val_std.tolist()
    }
    
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(args.results_dir, f"learning_curves_{args.model}_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({
            'args': vars(args),
            'results': results,
            'timestamp': stamp
        }, f, indent=2)
    print(f"✅ Saved results → {json_path}")
    
    # Plot if available
    if MATPLOTLIB_AVAILABLE:
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Train PR-AUC', linewidth=2)
            plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
            plt.plot(train_sizes_abs, val_mean, 's-', color='red', label='Validation PR-AUC', linewidth=2)
            plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
            plt.xlabel('Training Set Size', fontsize=12)
            plt.ylabel('PR-AUC', fontsize=12)
            plt.title(f'Learning Curves ({args.model})', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(args.results_dir, f"learning_curves_{args.model}_{stamp}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved plot → {plot_path}")
        except Exception as e:
            print(f"⚠️ Failed to create plot: {e}")
    
    # Diagnose bias vs variance
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")
    
    final_train = train_mean[-1]
    final_val = val_mean[-1]
    gap = final_train - final_val
    
    print(f"Final train PR-AUC: {final_train:.4f}")
    print(f"Final validation PR-AUC: {final_val:.4f}")
    print(f"Gap: {gap:.4f}")
    
    if gap > 0.15:
        print("⚠️ High variance (overfitting): Consider regularization or simpler model")
    elif final_val < 0.5:
        print("⚠️ High bias (underfitting): Consider more complex model or features")
    else:
        print("✅ Good balance between bias and variance")

if __name__ == "__main__":
    main()

