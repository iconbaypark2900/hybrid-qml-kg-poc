# Feature Diagnostics & Filtering

## Overview

This document describes the feature diagnostics and filtering improvements added to address the issue where classical models were getting exactly 0.5000 PR-AUC (random performance).

## Problem

When using enhanced features (1177D), all classical models (RandomForest, SVM, LogisticRegression) were getting exactly 0.5000 PR-AUC, indicating they weren't learning. Meanwhile, simpler 20D features worked fine (PR-AUC: 0.6004).

## Root Causes

1. **Zero/low variance features**: After StandardScaler normalization, some features had zero or very low variance
2. **High feature-to-sample ratio**: 1177 features vs 1208 samples (ratio ~0.97) can cause numerical instability
3. **NaN/Inf values**: Some features might contain invalid values
4. **Perfect multicollinearity**: Highly correlated features can cause numerical issues

## Solutions Implemented

### 1. Diagnostic Script (`scripts/diagnose_features.py`)

A comprehensive diagnostic tool that analyzes features and identifies issues:

```bash
python scripts/diagnose_features.py --relation CtD --use_cached_embeddings --full_graph_embeddings
```

**Features:**
- Checks for NaN/Inf values
- Analyzes feature variance (zero, low, very low)
- Detects perfect/near-perfect correlations
- Tests model predictions to see if models are learning
- Analyzes feature importances
- Checks feature-to-sample ratio
- Provides detailed warnings and recommendations

### 2. Automatic Feature Filtering in Pipeline

The pipeline now automatically:

1. **Removes NaN/Inf values**: Filters out samples with invalid values
2. **Removes low-variance features**: Uses `VarianceThreshold(threshold=1e-6)` to remove features with variance < 1e-6
3. **Warns about high feature-to-sample ratio**: Alerts when ratio > 0.5 or > 1.0
4. **Optional feature selection**: When `--use_feature_selection` flag is used and ratio > 1.0, applies mutual information feature selection

### 3. Enhanced Logging

The pipeline now provides detailed diagnostics:

```
================================================================================
FEATURE DIAGNOSTICS & FILTERING
================================================================================
Feature variance analysis:
  Zero variance features (<1e-10): X/Y
  Low variance features (<1e-6): X/Y
  Mean std: X.XXXXXX, Min: X.XXXXXX, Max: X.XXXXXX

Removing low-variance features...
After variance filtering: train (1208, N), test (302, N)

Feature-to-sample ratio: X.XX
⚠️  Warning messages if ratio is high
```

## Usage

### Basic Usage (with automatic filtering)

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --fast_mode
```

### With Feature Selection (recommended for high feature-to-sample ratio)

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --use_feature_selection \
    --fast_mode
```

### Run Diagnostics Only

```bash
python scripts/diagnose_features.py \
    --relation CtD \
    --use_cached_embeddings \
    --full_graph_embeddings
```

## Expected Improvements

After implementing these fixes:

1. **Zero-variance features removed**: Models won't try to learn from constant features
2. **Better numerical stability**: Removing problematic features prevents numerical issues
3. **Improved model performance**: Models should now learn properly instead of predicting randomly
4. **Better diagnostics**: Clear visibility into what's happening with features

## Technical Details

### Variance Threshold

- **Threshold**: 1e-6 (features with std < 1e-6 are removed)
- **Applied to**: Both training and test sets (fitted on training, transformed on test)
- **Preserves**: Feature names for debugging

### Feature Selection (when enabled)

- **Method**: Mutual Information (`SelectKBest` with `mutual_info_classif`)
- **Number of features**: min(80% of samples, total features)
- **Applied when**: Feature-to-sample ratio > 1.0 AND `--use_feature_selection` flag is set
- **Preserves**: Feature names for debugging

### Feature-to-Sample Ratio Warnings

- **Ratio > 1.0**: Critical warning - more features than samples
- **Ratio > 0.5**: Warning - high ratio, consider feature selection
- **Ratio ≤ 0.5**: No warning

## Troubleshooting

### Models still getting 0.5000 PR-AUC?

1. Run diagnostics: `python scripts/diagnose_features.py --relation CtD`
2. Check if feature selection is needed: Look at feature-to-sample ratio
3. Enable feature selection: Add `--use_feature_selection` flag
4. Check embeddings: Ensure embeddings have sufficient diversity

### Too many features removed?

- Lower the variance threshold (currently 1e-6)
- Check if features are being normalized correctly
- Verify embeddings have sufficient variance

### Feature selection removing important features?

- Check feature importances in diagnostics output
- Adjust `n_features_to_select` ratio (currently 80% of samples)
- Consider using PCA instead of mutual information

## Future Improvements

1. **PCA-based feature reduction**: Alternative to mutual information selection
2. **Adaptive threshold**: Adjust variance threshold based on data characteristics
3. **Feature importance-based selection**: Use model-based feature importance
4. **Cross-validation for feature selection**: Prevent overfitting in feature selection

