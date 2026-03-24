# Improvements V2 - Advanced Model Enhancements

## Summary

Implemented comprehensive improvements based on results analysis:

1. ✅ **Expanded SVM-RBF hyperparameter grid** (higher C, scale/auto gamma)
2. ✅ **Added Linear SVM** as alternative to RBF
3. ✅ **Increased QML qubits** (8 → 10 qubits)
4. ✅ **Feature importance analysis** (RandomForest-based)
5. ✅ **Ensemble methods** (VotingClassifier: RF + LR)

---

## 1. SVM-RBF Improvements

### Expanded Hyperparameter Grid

**Previous Grid:**
- Fast mode: C=[0.3, 3.0], gamma=[0.03, 0.3] (4 combinations)
- Full mode: C=[0.1, 0.3, 1.0, 3.0, 10.0], gamma=[0.01, 0.03, 0.1, 0.3, 1.0] (25 combinations)

**New Grid:**
- Fast mode: C=[0.1, 1.0, 10.0], gamma=[0.01, 0.1, 'scale'] (9 combinations)
- Full mode: C=[0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], gamma=[0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 'scale', 'auto'] (56 combinations)

**Key Changes:**
- ✅ Added higher C values (up to 30.0)
- ✅ Added lower gamma values (0.001)
- ✅ Added 'scale' and 'auto' gamma options (sklearn defaults)
- ✅ More comprehensive search space

**Expected Impact:**
- Should find better hyperparameters
- 'scale'/'auto' gamma adapts to feature variance
- Higher C values allow more complex decision boundaries

**Location**: `scripts/run_optimized_pipeline.py` lines 102-114

---

## 2. Linear SVM Alternative

### New Model: SVM-Linear-Optimized

**Implementation:**
- Uses Linear kernel (faster, more interpretable)
- Grid search over C values: [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
- Proper scaling pipeline
- Cross-validation for hyperparameter selection

**Why Linear SVM?**
- Often performs better than RBF for high-dimensional data
- Faster training and prediction
- More interpretable (linear decision boundary)
- Less prone to overfitting

**Expected Performance:**
- Should outperform SVM-RBF (which was 0.4399 PR-AUC)
- Comparable to LogisticRegression (0.5724 PR-AUC)
- Fast training time

**Location**: `scripts/run_optimized_pipeline.py` lines 89-161

---

## 3. QML Dimension Increase

### Increased Default Qubits: 8 → 10

**Change:**
- Default `qml_dim` increased from 8 to 10 qubits
- Can still override with `--qml_dim` argument

**Rationale:**
- More qubits = more expressive quantum features
- QSVC improved from 0.5141 → 0.5446 with 8 qubits
- 10 qubits should provide even better capacity

**Expected Impact:**
- Better quantum feature representation
- Improved QSVC performance (target: >0.55 PR-AUC)
- More expressive quantum models

**Location**: `scripts/run_optimized_pipeline.py` line 267

---

## 4. Feature Importance Analysis

### RandomForest-Based Feature Analysis

**Implementation:**
- After RandomForest training, extracts feature importances
- Logs top 20 most important features
- Identifies top 50 features for potential selection

**Output Example:**
```
FEATURE IMPORTANCE ANALYSIS
Top 50 Most Important Features:
  1. emb_0                          : 0.023456
  2. emb_1                          : 0.021234
  3. degree_h                       : 0.019876
  ...
```

**Benefits:**
- Understand which features matter most
- Guide feature engineering improvements
- Identify redundant features
- Can be used for feature selection (future enhancement)

**Location**: `scripts/run_optimized_pipeline.py` lines 908-923

---

## 5. Ensemble Methods

### Voting Classifier: RandomForest + LogisticRegression

**Implementation:**
- Soft voting ensemble
- Weights: [2, 1] (RandomForest weighted more)
- Only trains if both models succeed
- Skips in fast mode

**Why This Ensemble?**
- RandomForest: Best performer (0.6898 PR-AUC)
- LogisticRegression: Good baseline (0.5724 PR-AUC)
- Complementary strengths:
  - RF: Non-linear, feature interactions
  - LR: Linear, interpretable, fast

**Expected Performance:**
- Should match or exceed RandomForest alone
- More robust predictions
- Better generalization

**Location**: `scripts/run_optimized_pipeline.py` lines 925-976

---

## 6. Optional SVM-RBF Skipping

### New Flag: `--skip_svm_rbf`

**Purpose:**
- Allows skipping SVM-RBF if it continues to fail
- Useful for faster runs
- Can be enabled if SVM-RBF doesn't improve

**Usage:**
```bash
python scripts/run_optimized_pipeline.py --skip_svm_rbf
```

**Location**: `scripts/run_optimized_pipeline.py` lines 367-368, 873-875

---

## Testing

### Quick Test (Fast Mode)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --fast_mode
```

### Full Test (All Improvements)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings
```

### Skip SVM-RBF (If It Fails)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --skip_svm_rbf
```

---

## Expected Improvements

### Before (Current Results):
- RandomForest: 0.6898 PR-AUC ✅
- LogisticRegression: 0.5724 PR-AUC
- QSVC: 0.5446 PR-AUC
- SVM-RBF: 0.4399 PR-AUC ❌

### After (Expected):
- **Ensemble-RF-LR**: 0.70-0.72 PR-AUC (target)
- **RandomForest**: 0.69-0.71 PR-AUC (similar or better)
- **SVM-Linear**: 0.55-0.60 PR-AUC (better than RBF)
- **QSVC**: 0.55-0.60 PR-AUC (more qubits)
- **SVM-RBF**: 0.50-0.60 PR-AUC (improved grid search)

---

## Files Modified

1. **`scripts/run_optimized_pipeline.py`**
   - Lines 40: Added `VotingClassifier` import
   - Lines 89-161: New `train_svm_linear()` function
   - Lines 164-232: Updated `train_svm_rbf_with_grid_search()` with expanded grid
   - Lines 267: QML dimension default changed to 10
   - Lines 367-368: Added `--skip_svm_rbf` argument
   - Lines 850-976: Updated model training with Linear SVM, feature importance, and ensemble

---

## Next Steps

1. **Run Tests**: Execute pipeline with all improvements
2. **Analyze Results**: Compare against previous results
3. **Feature Engineering**: Use feature importance to guide improvements
4. **Tune Ensemble**: Experiment with different weights/combinations
5. **Quantum Improvements**: Try different encodings with 10 qubits

---

## Notes

- All changes are backward compatible
- Fast mode still works (reduced models/grid search)
- Ensemble only trains if both base models succeed
- Feature importance analysis runs automatically after RandomForest
- SVM-RBF can be skipped if needed

---

## Summary of All Improvements

| Improvement | Status | Expected Impact |
|-------------|--------|-----------------|
| Expanded SVM-RBF grid | ✅ | Better hyperparameters |
| Linear SVM | ✅ | Better than RBF, faster |
| QML 10 qubits | ✅ | Better quantum features |
| Feature importance | ✅ | Better understanding |
| Ensemble RF+LR | ✅ | Best performance |

**Total Expected Improvement**: +0.01 to +0.05 PR-AUC for best model

