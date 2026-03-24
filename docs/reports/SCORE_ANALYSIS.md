# Score Analysis Summary

## Executive Summary

**Date**: 2025-11-18  
**Run ID**: 20251118-213459  
**Task**: Compound treats Disease (CtD) link prediction

### Key Findings

1. **RandomForest is the clear winner** (PR-AUC: 0.6284)
2. **Quantum model (QSVC) is underperforming** (PR-AUC: 0.5141) - barely above random
3. **SVM-RBF is essentially random** (PR-AUC: 0.5027) - major issue!
4. **Significant overfitting** in RandomForest (train PR-AUC: 0.9676 vs test: 0.6284)

---

## Detailed Breakdown

### 1. RandomForest-Optimized (Best Model)

**Test Performance:**
- PR-AUC: **0.6284** ✅ (best)
- ROC-AUC: 0.6822
- Accuracy: 0.6225
- Precision: 0.6331
- Recall: 0.5828
- F1: 0.6069

**Train Performance:**
- PR-AUC: 0.9676 ⚠️ (severe overfitting!)
- ROC-AUC: 0.9658
- Accuracy: 0.8982

**Analysis:**
- ✅ Best performing model
- ⚠️ **Severe overfitting**: Train PR-AUC (0.9676) >> Test PR-AUC (0.6284)
- Gap of **0.3392** indicates model memorizing training data
- Test performance is reasonable but could be better with regularization

**Recommendations:**
- Add more regularization (max_depth, min_samples_split, min_samples_leaf)
- Consider ensemble methods with different random states
- Feature selection might help reduce overfitting

---

### 2. QSVC-Optimized (Quantum Model)

**Test Performance:**
- PR-AUC: **0.5141** ⚠️ (barely above random)
- ROC-AUC: 0.4981 (below random!)
- Accuracy: 0.4801
- Precision: 0.4813
- Recall: 0.5099
- F1: 0.4952

**Analysis:**
- ⚠️ **ROC-AUC < 0.5** indicates model is performing worse than random!
- PR-AUC slightly above 0.5 suggests minimal signal
- Score distributions show very small separation between classes
- Test set: Mean scores for Class 0 (0.0732) vs Class 1 (0.0678) are nearly identical

**Key Issues:**
1. **Feature dimensionality**: Only 5 qubits (qml_dim=5) may be insufficient
2. **Encoding method**: "hybrid" encoding may not be optimal
3. **Kernel quality**: Precomputed kernel may not capture relevant patterns
4. **Feature engineering**: QML features may need better preprocessing

**Recommendations:**
- Increase `qml_dim` (try 8, 10, or more qubits)
- Experiment with different encoding strategies (amplitude, phase, tensor_product)
- Try different quantum feature maps
- Consider using more classical features before quantum encoding
- Check if quantum features have sufficient variance

---

### 3. SVM-RBF-Optimized (Classical Baseline)

**Test Performance:**
- PR-AUC: **0.5027** ❌ (essentially random)
- ROC-AUC: 0.5032
- Accuracy: 0.5099
- Precision: 0.5050
- Recall: **1.0000** ⚠️ (predicting all positives!)
- F1: 0.6711

**Analysis:**
- ❌ **Recall = 1.0** means model predicts ALL samples as positive class
- This is a degenerate model - essentially a constant predictor
- PR-AUC near 0.5 confirms random performance
- Model is not learning any useful patterns

**Root Cause:**
- Likely a hyperparameter issue (C, gamma)
- Model may be too simple or too complex
- Feature scaling or normalization issue

**Recommendations:**
- Check hyperparameter grid search results
- Verify feature scaling is correct
- Try different C/gamma ranges
- Consider using different kernel (linear, polynomial)

---

## Prediction Analysis (QSVC)

### Test Set Performance

**Class Balance:**
- Perfectly balanced: 151 samples per class (50/50)

**Confusion Matrix:**
```
            Pred 0    Pred 1
True 0        68        83
True 1        74        77
```

**Key Observations:**
- **157 errors out of 302** (51.99% error rate)
- Nearly balanced false positives (83) and false negatives (74)
- Model is essentially guessing randomly

**Score Distribution:**
- Mean scores for both classes are nearly identical:
  - Class 0: mean=0.0732, std=0.5642
  - Class 1: mean=0.0678, std=0.5627
- **No separation** between classes in score space
- Score range: [-1.29, 1.17] but distributions overlap completely

---

## Configuration Analysis

**Current Setup:**
- **Embeddings**: ComplEx, 64D → 128D (complex→real conversion)
- **Full-graph embeddings**: ✅ Enabled (good!)
- **Graph features**: ✅ Enabled
- **Domain features**: ✅ Enabled
- **Feature selection**: ✅ Enabled
- **QML encoding**: hybrid
- **QML dimension**: 5 qubits ⚠️ (may be too small)

---

## Critical Issues Identified

### 1. **Quantum Model Not Learning**
- ROC-AUC < 0.5 indicates worse than random
- Score distributions show no class separation
- Need to investigate quantum feature engineering

### 2. **SVM-RBF Degenerate**
- Predicting all positives (recall=1.0)
- Not a useful baseline
- Hyperparameter tuning failed

### 3. **RandomForest Overfitting**
- Train/test gap of 0.34 in PR-AUC
- Model memorizing training data
- Needs regularization

### 4. **Feature Quality**
- Despite having 1177 features, models struggle
- May indicate:
  - Redundant features
  - Poor feature engineering
  - Need for better feature selection

---

## Recommendations

### Immediate Actions

1. **Fix SVM-RBF**:
   - Review hyperparameter search
   - Check feature scaling
   - Try simpler configurations

2. **Improve Quantum Model**:
   - Increase `qml_dim` to 8-10 qubits
   - Try different encoding methods
   - Add diagnostics for quantum feature variance
   - Check if quantum features are being computed correctly

3. **Regularize RandomForest**:
   - Add max_depth limit
   - Increase min_samples_split/leaf
   - Use cross-validation for hyperparameter tuning

4. **Feature Analysis**:
   - Check feature importance (RandomForest)
   - Identify redundant features
   - Consider feature selection based on mutual information

### Long-term Improvements

1. **Better Feature Engineering**:
   - Analyze which features RandomForest uses most
   - Create domain-specific features for CtD
   - Consider interaction features

2. **Quantum Feature Optimization**:
   - Experiment with different quantum feature maps
   - Try variational quantum feature maps
   - Consider quantum kernel methods

3. **Model Ensembles**:
   - Combine RandomForest with other models
   - Use stacking/blending
   - Consider quantum-classical hybrid approaches

---

## Next Steps

1. ✅ **Fixed**: Zero-variance features issue (column inference bug)
2. 🔄 **In Progress**: Understanding why models underperform
3. ⏭️ **Next**: 
   - Increase QML dimension
   - Fix SVM-RBF hyperparameters
   - Add RandomForest regularization
   - Feature importance analysis

---

## Performance Summary Table

| Model | Type | PR-AUC | ROC-AUC | Accuracy | Status |
|-------|------|--------|---------|----------|--------|
| RandomForest | Classical | **0.6284** | 0.6822 | 0.6225 | ✅ Best (but overfitting) |
| QSVC | Quantum | 0.5141 | 0.4981 | 0.4801 | ⚠️ Barely above random |
| SVM-RBF | Classical | 0.5027 | 0.5032 | 0.5099 | ❌ Degenerate |

**Target**: PR-AUC > 0.70 for a good model  
**Current Best**: 0.6284 (RandomForest)  
**Gap to Target**: 0.0716

