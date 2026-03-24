# Fixes Applied - Model Improvements

## Summary

Applied fixes to address the issues identified in the score analysis:

1. ✅ **Increased QML dimension** (5 → 8 qubits)
2. ✅ **Fixed SVM-RBF** with proper grid search
3. ✅ **Regularized RandomForest** to reduce overfitting
4. ✅ **Added feature importance analysis** for RandomForest

---

## 1. QML Dimension Increase

**Change**: Default `qml_dim` increased from 5 to 8 qubits

**Location**: `scripts/run_optimized_pipeline.py` line 267

**Details**:
- Changed default from `default=5` to `default=8`
- Updated help text to indicate the change
- This should improve quantum model capacity and feature representation

**Impact**: 
- Quantum models will have more qubits (8 instead of 5)
- Better feature representation capacity
- May improve QSVC performance (was 0.5141 PR-AUC)

---

## 2. SVM-RBF Grid Search Fix

**Change**: Replaced fixed hyperparameters with proper grid search

**Location**: `scripts/run_optimized_pipeline.py` lines 89-170

**Details**:
- Created new function `train_svm_rbf_with_grid_search()`
- Uses `GridSearchCV` with proper cross-validation
- Parameter grid:
  - **Fast mode**: C=[0.3, 3.0], gamma=[0.03, 0.3] (4 combinations)
  - **Full mode**: C=[0.1, 0.3, 1.0, 3.0, 10.0], gamma=[0.01, 0.03, 0.1, 0.3, 1.0] (25 combinations)
- Scoring: `average_precision` (PR-AUC)
- Includes proper scaling pipeline

**Previous Issue**:
- Fixed C=3.0, gamma=0.1
- Model was predicting all positives (recall=1.0)
- PR-AUC = 0.5027 (essentially random)

**Expected Impact**:
- Should find optimal hyperparameters
- Should fix degenerate behavior
- Should improve PR-AUC significantly

---

## 3. RandomForest Regularization

**Change**: Added regularization parameters to reduce overfitting

**Location**: `scripts/run_optimized_pipeline.py` lines 674-683

**Previous Settings**:
```python
max_depth=20,
min_samples_split=5,
```

**New Settings**:
```python
max_depth=10,  # Reduced from 20
min_samples_split=10,  # Increased from 5
min_samples_leaf=5,  # Added (new)
max_features='sqrt',  # Added (new)
```

**Impact**:
- **Train PR-AUC**: Was 0.9676 (severe overfitting)
- **Test PR-AUC**: Was 0.6284
- **Gap**: 0.3392 (too large)

**Expected Results**:
- Reduced train/test gap (less overfitting)
- Slightly lower train PR-AUC
- Similar or better test PR-AUC
- More generalizable model

---

## 4. Feature Importance Analysis

**Change**: Added feature importance logging for RandomForest

**Location**: `scripts/run_optimized_pipeline.py` lines 188-194, 227-229

**Details**:
- Modified `train_classical_model()` to accept `feature_names` parameter
- Logs top 10 most important features
- Stores feature importances in results dictionary
- Helps identify which features are most predictive

**Output Example**:
```
Top 10 Feature Importances:
  emb_0                          : 0.023456
  emb_1                          : 0.021234
  degree_h                       : 0.019876
  ...
```

**Impact**:
- Better understanding of which features matter
- Can guide feature engineering improvements
- Helps identify redundant features

---

## Testing

To test these changes:

```bash
# Quick test (fast mode)
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --fast_mode

# Full test (no fast mode for better SVM grid search)
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings

# Analyze results
python scripts/analyze_results.py \
    --results results/optimized_results_*.json \
    --predictions results/predictions_*.csv
```

---

## Expected Improvements

### Before Fixes:
- **RandomForest**: PR-AUC 0.6284 (overfitting: train 0.9676)
- **QSVC**: PR-AUC 0.5141 (barely above random)
- **SVM-RBF**: PR-AUC 0.5027 (degenerate, recall=1.0)

### After Fixes:
- **RandomForest**: PR-AUC ~0.63-0.65 (less overfitting, better generalization)
- **QSVC**: PR-AUC ~0.55-0.60 (more qubits = better capacity)
- **SVM-RBF**: PR-AUC ~0.55-0.65 (proper hyperparameters)

---

## Files Modified

1. `scripts/run_optimized_pipeline.py`
   - Line 267: QML dimension default changed
   - Lines 89-170: New SVM-RBF grid search function
   - Lines 173-242: Updated `train_classical_model()` with feature importance
   - Lines 674-683: RandomForest regularization
   - Lines 800-818: Updated model training loop

---

## Next Steps

1. **Run pipeline** with fixes and compare results
2. **Analyze feature importances** to guide further improvements
3. **Tune QML encoding** if needed (try different strategies)
4. **Consider ensemble methods** if individual models improve
5. **Feature engineering** based on importance analysis

---

## Notes

- All changes are backward compatible (defaults changed, but can override)
- Fast mode still works (reduced grid search for SVM)
- Feature importance analysis only runs for RandomForest (has `feature_importances_` attribute)
- SVM grid search uses proper cross-validation to prevent overfitting

