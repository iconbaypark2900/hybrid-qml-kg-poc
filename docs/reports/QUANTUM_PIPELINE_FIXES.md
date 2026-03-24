# Quantum Pipeline Performance Fixes

## Issue: Pipeline Stuck in Quantum Feature Selection

### Problem
The quantum feature importance computation was computing quantum kernels for **every feature removal**, which is extremely slow:
- For 1934 features → 1934 quantum kernel computations
- Each kernel computation takes significant time
- Pipeline appears to hang

### Solution
Optimized quantum feature importance computation with:

1. **Sampling**: Uses subset of samples (500) instead of full dataset
2. **Feature Ranking**: Uses mutual information for initial ranking (fast)
3. **Selective Evaluation**: Only evaluates top 100 features with quantum kernel
4. **Progress Logging**: Shows progress every 10 features

### Performance Improvement
- **Before**: O(n_features) quantum kernel computations (1934 kernels)
- **After**: O(100) quantum kernel computations + fast MI ranking
- **Speedup**: ~20x faster

---

## Updated Usage

### Recommended: Skip Quantum Feature Selection (Fastest)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_aware_embeddings \
    --use_quantum_feature_engineering \
    --qml_dim 12
```
**Note**: Quantum feature engineering still creates quantum-native features, just doesn't use quantum kernel for selection.

### With Quantum Feature Selection (Slower but More Optimal)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_aware_embeddings \
    --use_quantum_feature_engineering \
    --quantum_feature_selection \
    --qml_dim 12
```
**Note**: This will take longer but should complete (with progress logging).

---

## What Changed

### 1. Quantum Feature Importance (`quantum_layer/quantum_feature_engineering.py`)
- **Added sampling**: Uses 500 samples instead of full dataset
- **Added MI ranking**: Fast initial ranking with mutual information
- **Limited evaluation**: Only evaluates top 100 features with quantum kernel
- **Added progress logging**: Shows progress every 10 features

### 2. Error Handling
- **Graceful fallback**: If quantum feature selection fails, uses all features
- **Better error messages**: Clear logging of what's happening

---

## Performance Notes

### Quantum Feature Selection Timing
- **Without selection**: ~30 seconds (just creates features)
- **With selection**: ~5-10 minutes (evaluates top 100 features)
- **Progress**: Logs every 10 features evaluated

### If Still Too Slow
1. **Disable quantum feature selection**: Remove `--quantum_feature_selection`
2. **Use classical feature selection**: Use `--use_feature_selection` instead
3. **Reduce features**: Use `--max_interaction_features 25` to create fewer features

---

## Expected Behavior

### With Quantum Feature Selection Enabled
You should see:
```
INFO:quantum_layer.quantum_feature_engineering:Computing quantum feature importance (fast approximation)...
INFO:quantum_layer.quantum_feature_engineering:  Total features: 1934, Evaluating top 100
INFO:quantum_layer.quantum_feature_engineering:  Using 500 samples for kernel computation (for speed)
INFO:quantum_layer.quantum_feature_engineering:  Computing baseline kernel separability...
INFO:quantum_layer.quantum_feature_engineering:  Baseline separability: 1.001234
INFO:quantum_layer.quantum_feature_engineering:  Using mutual information for initial feature ranking...
INFO:quantum_layer.quantum_feature_engineering:  Evaluating top 100 features with quantum kernel...
INFO:quantum_layer.quantum_feature_engineering:    Progress: 10/100 features evaluated
INFO:quantum_layer.quantum_feature_engineering:    Progress: 20/100 features evaluated
...
```

### If It Still Hangs
1. **Check logs**: Look for the last progress message
2. **Kill and retry**: Without `--quantum_feature_selection`
3. **Report issue**: Include the last log line

---

## Quick Fix Commands

### Fastest (No Quantum Selection)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_aware_embeddings \
    --use_quantum_feature_engineering \
    --qml_dim 12
```

### Balanced (With Selection, Optimized)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_aware_embeddings \
    --use_quantum_feature_engineering \
    --quantum_feature_selection \
    --qml_dim 12
```

---

## Summary

✅ **Fixed**: Quantum feature importance computation optimized  
✅ **Added**: Progress logging  
✅ **Added**: Sampling and approximation for speed  
✅ **Added**: Graceful fallback on errors  

The pipeline should now complete successfully, though quantum feature selection will take 5-10 minutes with progress updates.

