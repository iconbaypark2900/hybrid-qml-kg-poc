"""
Diagnostic script to investigate why quantum features aren't separating classes.

Analyzes:
1. Classical feature separability (before quantum reduction)
2. Quantum feature separability (after PCA reduction)
3. Quantum kernel separability
4. Feature distributions for positive vs negative classes
5. Information loss during dimensionality reduction
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind, mannwhitneyu
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


def analyze_feature_separability(X, y, feature_name="Features"):
    """Analyze how well features separate classes."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {feature_name} Separability")
    logger.info(f"{'='*60}")
    
    pos_mask = y == 1
    neg_mask = y == 0
    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    
    logger.info(f"Shape: {X.shape}")
    logger.info(f"Positive samples: {X_pos.shape[0]}, Negative samples: {X_neg.shape[0]}")
    
    # 1. Mean differences per feature
    mean_pos = np.mean(X_pos, axis=0)
    mean_neg = np.mean(X_neg, axis=0)
    mean_diff = np.abs(mean_pos - mean_neg)
    
    logger.info(f"\nMean differences (|pos - neg|):")
    logger.info(f"  Min: {np.min(mean_diff):.6f}")
    logger.info(f"  Max: {np.max(mean_diff):.6f}")
    logger.info(f"  Mean: {np.mean(mean_diff):.6f}")
    logger.info(f"  Median: {np.median(mean_diff):.6f}")
    logger.info(f"  Features with diff > 0.1: {np.sum(mean_diff > 0.1)}/{len(mean_diff)}")
    logger.info(f"  Features with diff > 0.01: {np.sum(mean_diff > 0.01)}/{len(mean_diff)}")
    
    # 2. Statistical significance tests
    significant_features = []
    for i in range(X.shape[1]):
        try:
            stat, pval = ttest_ind(X_pos[:, i], X_neg[:, i])
            if pval < 0.05:
                significant_features.append(i)
        except:
            pass
    
    logger.info(f"\nStatistically significant features (t-test, p<0.05): {len(significant_features)}/{X.shape[1]}")
    
    # 3. Within-class vs between-class distances
    # Within-class distances
    pos_distances = pdist(X_pos)
    neg_distances = pdist(X_neg)
    within_class_dist = np.mean(np.concatenate([pos_distances, neg_distances]))
    
    # Between-class distances
    between_class_dist = []
    for x_pos in X_pos[:min(100, len(X_pos))]:  # Sample for efficiency
        for x_neg in X_neg[:min(100, len(X_neg))]:
            between_class_dist.append(np.linalg.norm(x_pos - x_neg))
    between_class_dist = np.mean(between_class_dist)
    
    logger.info(f"\nDistance Analysis:")
    logger.info(f"  Within-class mean distance: {within_class_dist:.6f}")
    logger.info(f"  Between-class mean distance: {between_class_dist:.6f}")
    logger.info(f"  Separation ratio (between/within): {between_class_dist/within_class_dist:.4f}")
    logger.info(f"  → Ratio > 1.0 means classes are separated")
    
    # 4. Silhouette score (higher is better, range: -1 to 1)
    try:
        silhouette = silhouette_score(X, y)
        logger.info(f"\nSilhouette Score: {silhouette:.4f}")
        logger.info(f"  → >0.5: well-separated, 0.2-0.5: overlapping, <0.2: highly overlapping")
    except Exception as e:
        logger.warning(f"  Could not compute silhouette score: {e}")
    
    # 5. Davies-Bouldin index (lower is better)
    try:
        db_index = davies_bouldin_score(X, y)
        logger.info(f"Davies-Bouldin Index: {db_index:.4f}")
        logger.info(f"  → Lower is better (good: <1.0, poor: >2.0)")
    except Exception as e:
        logger.warning(f"  Could not compute DB index: {e}")
    
    # 6. Feature variance analysis
    pos_var = np.var(X_pos, axis=0)
    neg_var = np.var(X_neg, axis=0)
    logger.info(f"\nVariance Analysis:")
    logger.info(f"  Positive class - Mean var: {np.mean(pos_var):.6f}, Min: {np.min(pos_var):.6f}, Max: {np.max(pos_var):.6f}")
    logger.info(f"  Negative class - Mean var: {np.mean(neg_var):.6f}, Min: {np.min(neg_var):.6f}, Max: {np.max(neg_var):.6f}")
    
    return {
        'mean_diff': mean_diff,
        'significant_features': len(significant_features),
        'within_class_dist': within_class_dist,
        'between_class_dist': between_class_dist,
        'separation_ratio': between_class_dist / within_class_dist if within_class_dist > 0 else 0,
        'silhouette': silhouette if 'silhouette' in locals() else None,
        'db_index': db_index if 'db_index' in locals() else None
    }


def analyze_kernel_separability(K, y, kernel_name="Kernel"):
    """Analyze kernel matrix separability."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {kernel_name} Separability")
    logger.info(f"{'='*60}")
    
    pos_mask = y == 1
    neg_mask = y == 0
    
    # Extract kernel similarities
    pos_pos = K[np.ix_(pos_mask, pos_mask)]
    neg_neg = K[np.ix_(neg_mask, neg_mask)]
    pos_neg = K[np.ix_(pos_mask, neg_mask)]
    
    # Remove diagonal (self-similarity = 1.0)
    pos_pos_offdiag = pos_pos[~np.eye(pos_pos.shape[0], dtype=bool)]
    neg_neg_offdiag = neg_neg[~np.eye(neg_neg.shape[0], dtype=bool)]
    pos_neg_flat = pos_neg.flatten()
    
    logger.info(f"Kernel Similarity Statistics:")
    logger.info(f"  Positive-Positive: mean={np.mean(pos_pos_offdiag):.6f}, std={np.std(pos_pos_offdiag):.6f}")
    logger.info(f"  Negative-Negative: mean={np.mean(neg_neg_offdiag):.6f}, std={np.std(neg_neg_offdiag):.6f}")
    logger.info(f"  Positive-Negative: mean={np.mean(pos_neg_flat):.6f}, std={np.std(pos_neg_flat):.6f}")
    
    # Separation metric
    within_class = (np.mean(pos_pos_offdiag) + np.mean(neg_neg_offdiag)) / 2
    between_class = np.mean(pos_neg_flat)
    separation = within_class - between_class
    
    logger.info(f"\nSeparation Analysis:")
    logger.info(f"  Within-class similarity: {within_class:.6f}")
    logger.info(f"  Between-class similarity: {between_class:.6f}")
    logger.info(f"  Separation (within - between): {separation:.6f}")
    logger.info(f"  → Positive separation means classes are separated")
    
    # Statistical test
    try:
        stat, pval = mannwhitneyu(
            np.concatenate([pos_pos_offdiag, neg_neg_offdiag]),
            pos_neg_flat,
            alternative='greater'
        )
        logger.info(f"\nMann-Whitney U test (within > between):")
        logger.info(f"  p-value: {pval:.6f}")
        logger.info(f"  → p < 0.05 means classes are significantly separated")
    except Exception as e:
        logger.warning(f"  Could not perform statistical test: {e}")
    
    return {
        'within_class': within_class,
        'between_class': between_class,
        'separation': separation,
        'p_value': pval if 'pval' in locals() else None
    }


def analyze_pca_information_loss(X_original, X_reduced, y, n_components):
    """Analyze how much information is lost during PCA reduction."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing PCA Information Loss")
    logger.info(f"{'='*60}")
    
    # Fit PCA to get explained variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    logger.info(f"Original dimensions: {X_original.shape[1]} → Reduced: {n_components}")
    logger.info(f"Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
    logger.info(f"Lost variance: {1-explained_var:.4f} ({(1-explained_var)*100:.2f}%)")
    
    # Compare separability before and after
    logger.info(f"\nSeparability Comparison:")
    orig_sep = analyze_feature_separability(X_scaled, y, "Original Features (scaled)")
    red_sep = analyze_feature_separability(X_pca, y, "PCA-Reduced Features")
    
    logger.info(f"\nSeparability Change:")
    logger.info(f"  Separation ratio: {orig_sep['separation_ratio']:.4f} → {red_sep['separation_ratio']:.4f}")
    logger.info(f"  Silhouette score: {orig_sep.get('silhouette', 'N/A')} → {red_sep.get('silhouette', 'N/A')}")
    
    return {
        'explained_variance': explained_var,
        'lost_variance': 1 - explained_var,
        'original_separability': orig_sep,
        'reduced_separability': red_sep
    }


def main():
    """Main diagnostic function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose quantum feature separability")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing results files")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing data files")
    parser.add_argument("--output_dir", type=str, default="diagnostics",
                       help="Output directory for diagnostic plots")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("QUANTUM FEATURE SEPARABILITY DIAGNOSTICS")
    logger.info("="*60)
    
    # Load the most recent results to get file paths
    # For now, we'll need to load data from the pipeline
    # This is a simplified version - in practice, you'd load from saved files
    
    logger.info("\nNOTE: This script needs to be run after the pipeline")
    logger.info("or with access to the feature matrices.")
    logger.info("\nTo use this script, modify it to load:")
    logger.info("  1. X_train_classical (before quantum reduction)")
    logger.info("  2. X_train_qml (after quantum reduction)")
    logger.info("  3. y_train (labels)")
    logger.info("  4. K_train (quantum kernel matrix)")
    
    # Example analysis structure (would need actual data)
    logger.info("\n" + "="*60)
    logger.info("DIAGNOSTIC STRUCTURE:")
    logger.info("="*60)
    logger.info("1. Classical Feature Separability")
    logger.info("2. Quantum Feature Separability (after PCA)")
    logger.info("3. PCA Information Loss Analysis")
    logger.info("4. Quantum Kernel Separability")
    logger.info("5. Feature Distribution Comparison")
    
    logger.info("\nTo run full diagnostics, integrate this script with the pipeline")
    logger.info("or save feature matrices during pipeline execution.")


if __name__ == "__main__":
    main()

