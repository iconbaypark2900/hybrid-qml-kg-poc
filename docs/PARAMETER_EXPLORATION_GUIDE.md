# Parameter Exploration Guide

## Overview

The `explore_parameters.py` script helps you find optimal hyperparameters for both classical and quantum models **before** running the full pipeline. This saves time by testing many configurations quickly and recommending the best ones.

## Quick Start

### Explore Both Quantum and Classical Parameters

```bash
python scripts/explore_parameters.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --full_graph_embeddings \
  --embedding_method RotatE \
  --embedding_dim 128 \
  --embedding_epochs 200 \
  --use_evidence_weighting \
  --min_shared_genes 1 \
  --use_contrastive_learning \
  --contrastive_epochs 75 \
  --qml_dim 12 \
  --random_state 42
```

### Quantum Only (Faster)

```bash
python scripts/explore_parameters.py \
  --relation CtD \
  --full_graph_embeddings \
  --quantum_only \
  --qml_dim 12
```

### Classical Only

```bash
python scripts/explore_parameters.py \
  --relation CtD \
  --full_graph_embeddings \
  --classical_only
```

## What It Tests

### Quantum Parameters

The script tests combinations of:
- **Encoding strategies**: `optimized_diff`, `hybrid`
- **Reduction methods**: `pca`, `lda`, `None`
- **Feature selection**: `f_classif`, `mutual_info`, `None`
- **Feature selection multiplier**: `2.0`, `4.0`, `6.0`
- **Pre-PCA dimension**: `0`, `64`, `128`

**Metrics computed:**
- Separation ratio (between-class / within-class distance)
- Silhouette score
- Davies-Bouldin index
- Number of statistically significant features
- Overall quality score (weighted combination)

### Classical Parameters

**RandomForest:**
- `n_estimators`: [100, 200, 300]
- `max_depth`: [8, 10, 12, 15]
- `min_samples_split`: [5, 10, 20]
- `min_samples_leaf`: [3, 5, 10]
- `max_features`: ['sqrt', 0.7, 0.9]

**LogisticRegression:**
- `C`: [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

**Metrics computed:**
- Cross-validation PR-AUC (mean ± std)
- Test set PR-AUC

## Output Files

The script generates several output files in the `results/` directory:

1. **`parameter_recommendations_YYYYMMDD-HHMMSS.json`**
   - Top recommended configurations
   - Best quantum configuration with quality score
   - Best classical configurations with PR-AUC scores

2. **`quantum_exploration_YYYYMMDD-HHMMSS.csv`**
   - All tested quantum configurations
   - Sorted by quality score (best first)
   - Includes all metrics for each configuration

3. **`randomforest_exploration_YYYYMMDD-HHMMSS.csv`**
   - All tested RandomForest configurations
   - Sorted by test PR-AUC (best first)

4. **`logisticregression_exploration_YYYYMMDD-HHMMSS.csv`**
   - All tested LogisticRegression configurations
   - Sorted by test PR-AUC (best first)

## Interpreting Results

### Quantum Quality Score

The quality score is a weighted combination:
- **40%** - Separation ratio (higher = better class separation)
- **30%** - Silhouette score (scaled, higher = better)
- **20%** - Significant features ratio (more = better)
- **10%** - Negative Davies-Bouldin index (lower DB = better)

**Good scores:**
- Quality score > 5.0: Excellent separability
- Quality score 3.0-5.0: Good separability
- Quality score < 3.0: Poor separability (may need different parameters)

### Classical PR-AUC

- **PR-AUC > 0.80**: Excellent performance
- **PR-AUC 0.70-0.80**: Good performance
- **PR-AUC < 0.70**: May need better features or different models

## Using Recommendations

After running the script, use the recommended parameters in your full pipeline:

```bash
# Example: Using recommended quantum parameters
python scripts/run_optimized_pipeline.py \
  --relation CtD \
  --qml_reduction_method pca \
  --qml_feature_selection_method f_classif \
  --qml_feature_select_k_mult 4.0 \
  --qml_pre_pca_dim 0 \
  --qml_encoding optimized_diff \
  # ... other parameters from recommendations
```

## Tips

1. **Start with quantum-only** to quickly find good quantum parameters
2. **Use cached embeddings** if you've already trained them:
   ```bash
   --use_cached_embeddings
   ```
3. **Reduce parameter grid** if testing takes too long:
   - Edit the script to test fewer combinations
   - Use `--quantum_only` or `--classical_only` to focus
4. **Compare with historical results** - Check previous successful runs to see what worked

## Time Estimates

- **Quantum exploration**: ~2-5 minutes (tests ~50-100 combinations)
- **Classical exploration**: ~5-15 minutes (tests ~100-500 combinations)
- **Both**: ~10-20 minutes total

## Troubleshooting

**"All features were constant" error:**
- Try a different encoding strategy
- Check if embeddings are properly loaded
- Verify embedding dimension matches expected

**Low quality scores:**
- Try different reduction methods (LDA vs PCA)
- Adjust feature selection multiplier
- Check if embeddings have sufficient diversity

**Memory issues:**
- Reduce `--pos_edge_sample` size
- Test fewer parameter combinations
- Use `--quantum_only` or `--classical_only`
