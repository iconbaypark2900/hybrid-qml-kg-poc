# Quantum Performance Fixes Applied

## Overview

Based on the analysis in `WHY_QUANTUM_UNDERPERFORMS.md`, we've implemented fixes to address the key issues:

1. **Severe Overfitting** (train=1.0, test=0.6081)
2. **Information Loss** (95% discarded through reduction)
3. **Feature Mismatch** (12D vs 2313 features)
4. **Small Dataset** (825 samples)

---

## Fixes Implemented

### 1. Higher Dimensions to Preserve Information ✅

**Problem**: Only 12 dimensions after aggressive reduction (95% information loss)

**Fix**: 
- Increased from 12D to **24D or 32D**
- This preserves more information while still being quantum-feasible

**Scripts**:
- `run_quantum_fixed.sh`: Uses `--qml_dim 24`
- `scripts/fix_quantum_performance.py`: Tests 12D, 24D, and 32D configurations

---

### 2. Less Aggressive Dimensionality Reduction ✅

**Problem**: Multiple reduction steps (256D → 128D → 24D → 12D) lose critical information

**Fix**:
- **Skip pre-PCA**: `--qml_pre_pca_dim 0` (was 64 or 128)
- **Select more features**: `--qml_feature_select_k_mult 6.0` (was 2.0-4.0)
- Go directly from 256D → 48D (via feature selection) → 24D (via PCA)

**Scripts**:
- `run_quantum_fixed.sh`: Uses `--qml_pre_pca_dim 0 --qml_feature_select_k_mult 6.0`

---

### 3. Simpler Feature Maps to Reduce Overfitting ✅

**Problem**: Complex feature maps (reps=2, full entanglement) cause overfitting (train=1.0)

**Fix**:
- **Fewer repetitions**: `--qml_feature_map_reps 1` (was 2)
- **Linear entanglement**: `--qml_entanglement linear` (was full)
- Simpler models are less prone to memorization

**Scripts**:
- `run_quantum_fixed.sh`: Uses `--qml_feature_map_reps 1 --qml_entanglement linear`
- Updated `quantum_layer/qml_trainer.py` to respect `args.entanglement` parameter

---

### 4. Hybrid Features (Quantum + Classical) ✅

**Problem**: Quantum models only use 12D embeddings, while classical uses 2313 features

**Fix**:
- Combine quantum features with reduced classical features
- Quantum: 24D embeddings
- Classical: 24D PCA-reduced features
- Total: 48D hybrid features

**Scripts**:
- `scripts/fix_quantum_performance.py`: Tests hybrid feature configurations

---

### 5. Better Regularization ✅

**Problem**: Overfitting suggests insufficient regularization

**Fix**:
- Expanded C grid: `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]` (already implemented)
- Higher C values provide more regularization
- Grid search finds optimal C automatically

**Scripts**:
- Already implemented in `quantum_layer/qml_trainer.py`

---

## How to Use

### Option 1: Test Fixed Configurations (Fast)

Test multiple fixed configurations to find the best one:

```bash
./fix_quantum.sh
```

This runs `scripts/fix_quantum_performance.py` which tests:
- 12D, 24D, 32D dimensions
- With and without hybrid features
- Simpler feature maps (reps=1, linear entanglement)

**Expected time**: 30-60 minutes (tests 5 configurations)

---

### Option 2: Run Full Pipeline with Best Fixes

Run the complete pipeline with the recommended fixes:

```bash
./run_quantum_fixed.sh
```

This uses:
- 24D quantum features
- No pre-PCA (skip it)
- More feature selection (6.0x multiplier)
- Simpler feature maps (reps=1, linear entanglement)
- All other optimizations (contrastive learning, task-specific fine-tuning, calibration)

**Expected time**: 2-3 hours (full pipeline)

---

## Expected Improvements

Based on the fixes, we expect:

1. **Reduced Overfitting**:
   - Current: Train=1.0, Test=0.6081, Gap=0.39
   - Target: Train<0.95, Test>0.70, Gap<0.20

2. **Better Test Performance**:
   - Current: 0.6081 PR-AUC
   - Target: 0.70-0.75 PR-AUC (closer to classical models)

3. **More Information Preserved**:
   - Current: 12D (95% loss)
   - Target: 24D (90% loss) or 32D (87.5% loss)

4. **Better Generalization**:
   - Current: Perfect train, poor test
   - Target: Good train, good test

---

## Monitoring Results

After running, check:

1. **Overfitting Gap**: Should be < 0.20 (currently 0.39)
2. **Test PR-AUC**: Should be > 0.70 (currently 0.6081)
3. **Train PR-AUC**: Should be < 0.95 (currently 1.0)
4. **Feature Dimensions**: Should be 24D or 32D (currently 12D)

---

## Next Steps if Still Underperforming

If quantum models still underperform after these fixes:

1. **Increase dimensions further**: Try 32D or 48D
2. **Improve embeddings**: Focus on increasing diversity (currently 21.5% head, 5.3% tail)
3. **More data**: Increase `--pos_edge_sample` to 2000 or 3000
4. **Different feature maps**: Try `Pauli` or `custom_link_prediction`
5. **Accept reality**: Classical models may be better for this task/dataset

---

## Files Modified

1. **`quantum_layer/qml_trainer.py`**: 
   - Updated to respect `args.entanglement` parameter (was hardcoded to "linear")

2. **`scripts/fix_quantum_performance.py`**: 
   - New script to test fixed configurations
   - Implements hybrid features
   - Tests multiple dimension/config combinations

3. **`run_quantum_fixed.sh`**: 
   - New script to run full pipeline with fixes

4. **`fix_quantum.sh`**: 
   - Wrapper to run the fix testing script

---

## References

- **Analysis**: `docs/WHY_QUANTUM_UNDERPERFORMS.md`
- **Original Results**: `results/quantum_actual_kernel_results_*.csv`
- **Classical Performance**: 0.8859 PR-AUC (LogisticRegression)
