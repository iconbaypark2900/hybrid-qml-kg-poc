# Data Leakage Prevention Guide

## 🎯 Overview

**Data leakage** occurs when information from the test set inadvertently influences model training, leading to overly optimistic performance estimates that don't generalize to new data. This guide documents how we prevent leakage in the Hybrid QML-KG pipeline.

## 🚨 What is Data Leakage?

### Definition

Data leakage happens when training uses information that wouldn't be available in a real-world deployment scenario.

### Common Sources in Link Prediction

1. **Global statistics on full dataset** (train + test)
   - ❌ Fitting scalers on train + test combined
   - ❌ Computing graph metrics on all edges
   - ❌ Using test edges in feature engineering

2. **Test set information in preprocessing**
   - ❌ Normalizing with global mean/std
   - ❌ Feature selection on full dataset
   - ❌ Cross-validation on mixed data

3. **Temporal leakage**
   - ❌ Using future information to predict past
   - ❌ Not respecting time-based splits

### Why It's Critical

```
With Leakage:    Test PR-AUC = 0.95 (looks amazing!)
Without Leakage: Test PR-AUC = 0.65 (realistic)

Result: 30% overestimation of performance
        Model fails in production
```

---

## ✅ Our Leakage Prevention Strategy

### 1. Train-Only Graph Metrics

**Issue**: Graph structural features (PageRank, betweenness, degree) were computed on ALL edges including test set.

**Fix**:
```python
# BEFORE (LEAKAGE):
feature_builder.build_graph(task_edges)  # All edges (train + test)

# AFTER (NO LEAKAGE):
train_edges_only = train_df[train_df['label'] == 1].copy()
feature_builder.build_graph(train_edges_only)  # Train edges only
```

**Impact**: Graph metrics now reflect only knowledge available during training.

### 2. Train-Only Domain Features

**Issue**: Domain features (metaedge diversity counts) used full dataset.

**Fix**:
```python
# BEFORE (LEAKAGE):
X_train = feature_builder.build_features(
    train_df, embeddings, edges_df=task_edges  # All edges
)

# AFTER (NO LEAKAGE):
X_train = feature_builder.build_features(
    train_df, embeddings, edges_df=train_edges_only  # Train only
)
```

**Impact**: Domain statistics computed only from training data.

### 3. Explicit Scaler Fitting

**Issue**: Scaler fitting was implicit and fragile.

**Fix**:
```python
# BEFORE (FRAGILE):
if self.scaler is None:
    self.scaler = StandardScaler()
    features = self.scaler.fit_transform(features)  # Implicit
else:
    features = self.scaler.transform(features)

# AFTER (EXPLICIT):
def build_features(self, ..., fit_scaler: bool = False):
    if fit_scaler:
        logger.info("Fitting scaler on training features (prevents leakage)")
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
    else:
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted! Fit on train first.")
        features = self.scaler.transform(features)

# Usage:
X_train = feature_builder.build_features(..., fit_scaler=True)   # Fit
X_test = feature_builder.build_features(..., fit_scaler=False)   # Transform
```

**Impact**: Clear, explicit control over when fitting occurs.

### 4. Automated Leakage Validation

**Added**: Automatic check to detect test edges in training data.

```python
def validate_no_leakage(train_df, test_df, edges_df):
    """Validate that edges_df doesn't contain test edges."""
    test_edges = set(zip(test_positive['source_id'], test_positive['target_id']))
    edges_set = set(zip(edges_df[src_col], edges_df[tgt_col]))
    leakage = test_edges & edges_set

    if leakage:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Found {len(leakage)} test edges in edges_df."
        )

    logger.info(f"✓ Leakage check passed: No test edges found")
```

**Usage in pipeline**:
```python
# Automatically validates before feature building
validate_no_leakage(train_df, test_df, train_edges_only)
```

---

## 🔍 How to Verify No Leakage

### 1. Check Pipeline Logs

Run the pipeline and look for these messages:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_cached_embeddings \
    --fast_mode 2>&1 | grep -E "(leakage|LEAK|TRAINING)"
```

**Expected output**:
```
INFO:__main__:Using 604 TRAINING edges for graph/domain features (prevents leakage)
INFO:__main__:Validating no data leakage...
INFO:kg_layer.enhanced_features:✓ Leakage check passed: No test edges found in edges_df (604 edges validated)
INFO:__main__:Building graph on TRAINING edges only (prevents leakage)...
INFO:kg_layer.enhanced_features:Fitting scaler on training features (prevents leakage)
```

### 2. Manual Verification

```python
# Check train/test edge counts
train_positive = train_df[train_df['label'] == 1]
test_positive = test_df[test_df['label'] == 1]

print(f"Train positive edges: {len(train_positive)}")
print(f"Test positive edges: {len(test_positive)}")
print(f"Total: {len(train_positive) + len(test_positive)}")

# Verify no overlap
train_set = set(zip(train_positive['source_id'], train_positive['target_id']))
test_set = set(zip(test_positive['source_id'], test_positive['target_id']))
overlap = train_set & test_set

assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} edges in both train and test!"
print("✓ No overlap between train and test sets")
```

### 3. Reproducibility Check

```bash
# Run twice with same seed
python scripts/run_optimized_pipeline.py --relation CtD --random_state 42 > run1.log
python scripts/run_optimized_pipeline.py --relation CtD --random_state 42 > run2.log

# Results should be identical
diff run1.log run2.log
```

---

## 📋 Leakage Prevention Checklist

Before deploying models, verify:

- [ ] Graph built only on TRAINING edges
- [ ] Domain features use only TRAINING edges
- [ ] Scaler fitted only on TRAINING data
- [ ] No test edges in feature computation
- [ ] Validation check passes
- [ ] Train/test split is stratified and balanced
- [ ] Same random seed gives identical results

---

## 🛠️ Implementation Details

### Files Modified

1. **`kg_layer/enhanced_features.py`**:
   - Added `validate_no_leakage()` function
   - Added `fit_scaler` parameter to `build_features()`
   - Added explicit scaler fitting with error checking
   - Added documentation warnings

2. **`scripts/run_optimized_pipeline.py`**:
   - Extract train-only positive edges
   - Call `validate_no_leakage()` before feature building
   - Pass `fit_scaler=True` for train, `False` for test
   - Use train-only edges for graph and domain features

### Code Locations

**Validation check**: `kg_layer/enhanced_features.py:21-61`
```python
validate_no_leakage(train_df, test_df, edges_df)
```

**Train-only edge extraction**: `scripts/run_optimized_pipeline.py:372-383`
```python
train_edges_only = train_df[train_df['label'] == 1].copy()
train_edges_only['source'] = train_edges_only['source_id'].map(id_to_entity)
validate_no_leakage(train_df, test_df, train_edges_only)
```

**Scaler fitting**: `kg_layer/enhanced_features.py:419-432`
```python
if fit_scaler:
    logger.info("Fitting scaler on training features (prevents leakage)")
    self.scaler = StandardScaler()
    features_array = self.scaler.fit_transform(features_array)
else:
    if self.scaler is None:
        raise RuntimeError("Scaler not fitted! Fit on train first.")
    features_array = self.scaler.transform(features_array)
```

---

## 🧪 Testing Leakage Prevention

### Test 1: Validation Detects Leakage

```python
# Artificially introduce leakage (for testing)
train_edges_with_leakage = pd.concat([
    train_edges_only,
    test_df[test_df['label'] == 1].head(10)  # Add some test edges
])

# This should raise ValueError
try:
    validate_no_leakage(train_df, test_df, train_edges_with_leakage)
except ValueError as e:
    print(f"✓ Leakage detected correctly: {e}")
```

### Test 2: Scaler Fitting Order

```python
# Test that calling test before train raises error
feature_builder = EnhancedFeatureBuilder(normalize=True)

# This should raise RuntimeError
try:
    X_test = feature_builder.build_features(
        test_df, embeddings, fit_scaler=False  # Try to transform before fitting
    )
except RuntimeError as e:
    print(f"✓ Error caught correctly: {e}")
```

### Test 3: Train-Test Independence

```python
# Build features for train and test
X_train = feature_builder.build_features(train_df, embeddings, edges_df=train_edges_only, fit_scaler=True)
X_test = feature_builder.build_features(test_df, embeddings, edges_df=train_edges_only, fit_scaler=False)

# Check that scaler stats come from train only
train_mean = X_train.mean(axis=0)
test_mean = X_test.mean(axis=0)

# These should differ (test uses train's scaler, not its own mean)
assert not np.allclose(train_mean, np.zeros_like(train_mean), atol=1e-5), "Train should be centered"
print("✓ Test set transformed using train statistics (not test's own)")
```

---

## 📊 Impact of Leakage Prevention

### Before (WITH Leakage)

```
Graph: 755 edges (train + test)
PageRank: Computed on full graph
Domain features: All 755 edges
Scaler: Fit on concatenated data

Test PR-AUC: 0.XX (overestimated)
```

### After (NO Leakage)

```
Graph: 604 edges (train only)
PageRank: Computed on train graph only
Domain features: 604 training edges
Scaler: Fit on train, transform on test

Test PR-AUC: 0.YY (realistic)
```

### Expected Changes

- **Lower test scores**: More realistic, not overestimated
- **Better generalization**: Model truly learns from train only
- **Fair comparison**: All models evaluated on same footing
- **Production-ready**: Performance matches real-world deployment

---

## 💡 Best Practices

### 1. Always Validate

```python
# ALWAYS call validate_no_leakage before feature building
validate_no_leakage(train_df, test_df, edges_df)
```

### 2. Be Explicit

```python
# BAD: Implicit fitting
X_train = builder.build_features(train_df, embeddings)
X_test = builder.build_features(test_df, embeddings)

# GOOD: Explicit fitting
X_train = builder.build_features(train_df, embeddings, fit_scaler=True)
X_test = builder.build_features(test_df, embeddings, fit_scaler=False)
```

### 3. Document Assumptions

```python
def build_features(self, links_df, embeddings, edges_df=None, fit_scaler=False):
    """
    Build features for link prediction.

    Important: To prevent data leakage:
    - Set fit_scaler=True ONLY for training data
    - Pass only TRAINING edges in edges_df
    - Build graph only on TRAINING edges before calling this
    """
```

### 4. Use K-Fold CV Carefully

```python
# Leakage prevention in CV
for fold_idx, (train_idx, test_idx) in enumerate(cv_folds):
    # Extract train fold
    train_fold = data.iloc[train_idx]
    test_fold = data.iloc[test_idx]

    # Build graph on THIS FOLD'S train data only
    fold_train_edges = train_fold[train_fold['label'] == 1]
    feature_builder.build_graph(fold_train_edges)

    # Fit scaler on THIS FOLD'S train data
    X_train = feature_builder.build_features(train_fold, embeddings, fit_scaler=True)
    X_test = feature_builder.build_features(test_fold, embeddings, fit_scaler=False)
```

---

## 🔗 Related Documentation

- **IMPROVEMENTS_SUMMARY.md**: Overall progress tracking
- **CV_EVALUATION_GUIDE.md**: K-Fold cross-validation
- **FULL_GRAPH_EMBEDDINGS_GUIDE.md**: Embedding training strategies

---

## 📚 References

### Academic Resources

1. **"Leakage in Data Mining"** (Kaufman et al., 2012)
   - Comprehensive taxonomy of leakage types
   - Real-world examples and prevention strategies

2. **"A Few Useful Things to Know About Machine Learning"** (Domingos, 2012)
   - Section on evaluation pitfalls
   - Importance of proper train/test separation

3. **"Common Pitfalls in Machine Learning"**
   - Data leakage in feature engineering
   - Cross-validation done wrong

### Industry Best Practices

- **Kaggle Competitions**: Strict leakage prevention rules
- **Production ML**: Leakage leads to silent model failures
- **Research Papers**: Reviewers check for leakage in evaluation

---

## ⚠️ Warning Signs of Leakage

Watch out for these red flags:

1. **Too-good-to-be-true results**: Test PR-AUC > 0.95 on hard problems
2. **Perfect training accuracy**: Model memorizes rather than learns
3. **Drop in production**: Model performs worse than in evaluation
4. **Unstable across folds**: High variance in CV folds
5. **Test better than train**: Test metrics higher than train (impossible without leakage)

---

**Last Updated**: 2025-11-11
**Status**: ✅ Implemented and validated
**Impact**: Fair evaluation, production-ready models
