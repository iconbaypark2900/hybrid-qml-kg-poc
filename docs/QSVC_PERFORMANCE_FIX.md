# QSVC Performance Issue: Root Cause and Fix

## Problem

QSVC was performing poorly (~0.37 PR-AUC) despite the parameter exploration script showing excellent separability metrics (separation ratio 14.39, silhouette 0.93) for LDA-reduced features.

## Root Cause

The `explore_parameters.py` script was **only measuring feature separability**, not actually testing QSVC performance. This led to misleading recommendations:

1. **LDA with 1D output** scored highly on separability metrics
   - LDA for binary classification always produces 1D output
   - This maximizes linear class separation in 1D space
   - Quality score: 10.73 (excellent!)

2. **But 1D is too restrictive for quantum kernels**
   - Quantum kernels need multiple dimensions to create useful feature spaces
   - 1D features can't express complex quantum patterns
   - Actual QSVC PR-AUC: ~0.37 (terrible!)

3. **The separability metrics were misleading**
   - They measure linear separability, not quantum kernel performance
   - High separability in 1D doesn't translate to good quantum performance

## Solution

Updated `explore_parameters.py` to:

1. **Actually test QSVC performance** using RBF kernel as a proxy (faster than full quantum kernel)
2. **Filter out 1D configurations** - Skip LDA when it produces 1D output
3. **Prioritize PR-AUC over separability** - Quality score now weights PR-AUC at 60% vs separability at 40%
4. **Show PR-AUC in recommendations** - Display actual performance metrics, not just separability

## Changes Made

### `test_quantum_config()` function:
- Added check to skip 1D configurations
- Added QSVC test using RBF kernel (fast proxy for quantum kernel)
- Returns `test_pr_auc` and `train_pr_auc` metrics
- Updated quality score to weight PR-AUC at 60%

### Output improvements:
- Recommendations now sorted by PR-AUC first, then quality score
- Display shows PR-AUC prominently
- CSV includes `test_pr_auc` and `train_pr_auc` columns

## Expected Impact

- **Better parameter recommendations** - Based on actual QSVC performance, not just separability
- **Avoid 1D configurations** - Automatically filters out LDA when it produces 1D
- **More accurate quality scores** - PR-AUC weighted more heavily than separability metrics

## Overfitting Detection (Added)

### Problem
- **2313 features vs 825 samples** (almost 3x more features than samples!)
- **Low embedding diversity**: Only 21.5% unique head embeddings, 5.3% tail embeddings
- Classic overfitting scenario: model memorizes training data

### Solution
Added comprehensive overfitting detection:

1. **Feature-to-sample ratio check**: Warns when features > 2x samples
2. **Cross-validation**: Uses 3-fold CV to detect generalization issues
3. **Train-Test gap monitoring**: Tracks overfitting gap (train PR-AUC - test PR-AUC)
4. **CV-Test gap monitoring**: Tracks CV-Test mismatch (CV PR-AUC - test PR-AUC)
5. **Quality score penalty**: Penalizes configurations with high overfitting gaps
6. **C parameter grid search**: Tests multiple regularization values (0.1, 0.3, 1.0, 3.0, 10.0)

### Metrics Added
- `test_pr_auc`: Test set performance (primary metric)
- `cv_pr_auc`: Cross-validation performance (generalization)
- `train_pr_auc`: Training set performance (overfitting check)
- `overfitting_gap`: Train - Test gap (warns if > 0.15)
- `cv_test_gap`: CV - Test gap (warns if > 0.10)
- `best_C`: Optimal regularization parameter
- `feature_ratio`: Features / Samples ratio

### Quality Score Formula
```
quality_score = 
  0.5 * test_pr_auc * 10 +           # Test performance (primary)
  0.2 * cv_pr_auc * 10 +             # CV performance (generalization)
  0.1 * separation_ratio +           # Feature separability
  0.1 * silhouette * 10 +            # Cluster quality
  0.05 * significant_features_ratio +  # Feature quality
  -0.05 * overfitting_penalty +       # Penalty for overfitting
  -0.05 * cv_mismatch_penalty         # Penalty for CV-Test mismatch
```

## Next Steps

1. Re-run `explore_parameters.py` to get updated recommendations with overfitting detection
2. Use PR-AUC-based recommendations for full pipeline runs
3. Consider using PCA instead of LDA for quantum features (PCA preserves multiple dimensions)
4. **Prioritize configurations with low overfitting gaps** (< 0.10)

## Key Insights

1. **Feature separability ≠ Quantum kernel performance**
   - High separability in low dimensions (especially 1D) doesn't guarantee good quantum kernel performance
   - Quantum kernels need sufficient dimensionality to express complex patterns

2. **More features ≠ Better performance**
   - 2313 features with 825 samples leads to overfitting
   - Feature selection and regularization are critical

3. **Cross-validation is essential**
   - CV PR-AUC helps detect generalization issues
   - CV-Test gap > 0.10 indicates the model won't generalize well
