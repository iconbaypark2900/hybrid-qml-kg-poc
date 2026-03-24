# Quantum Pipeline Results Analysis

## Run Summary
- **QSVC PR-AUC**: 0.4392 (worse than random 0.5)
- **Classical Baseline**: 0.5571 (better than quantum)
- **Status**: Quantum model underperforming

---

## Ideal vs Noisy Benchmarking
- Track execution metadata in `results/experiment_history.csv` (execution mode, noise model, backend label)
- Use the benchmark script and comparison utility to compare ideal vs noisy simulator runs:
```bash
bash scripts/benchmark_ideal_noisy.sh CtD results --fast_mode --quantum_only
python benchmarking/ideal_vs_noisy_compare.py --results_dir results
```

---

## Critical Issues Identified

### 1. **Kernel Separability Failure** ⚠️
```
Positive-Positive similarity: mean=0.004180
Negative-Negative similarity: mean=0.004213
Positive-Negative similarity: mean=0.004207
```
**Problem**: All similarities are nearly identical (~0.004), meaning the quantum kernel **cannot distinguish between classes**.

**Impact**: The kernel is essentially useless for classification.

### 2. **Severe Overfitting** ⚠️
- **Train PR-AUC**: 1.0000 (perfect)
- **Test PR-AUC**: 0.4392 (worse than random)
- **Gap**: 0.5608

**Problem**: Model memorized training data but can't generalize.

### 3. **Low Embedding Diversity** ⚠️
- **Head diversity**: 331/1208 (27.4%)
- **Tail diversity**: 73/1208 (6.0%)

**Problem**: Many entities share identical embeddings, reducing information content.

### 4. **Quantum-Aware Embeddings Didn't Help** ⚠️
- **Initial separability**: 1.001150
- **Final separability**: 1.000354
- **Change**: -0.000796 (decreased)

**Problem**: Fine-tuning actually made separability worse.

### 5. **Poor Raw Embedding Separability**
- **Separation ratio**: 1.0314 (barely above 1.0)
- **Mean differences**: max=0.000291 (very small)
- **Features with diff > 0.01**: 0/256

**Problem**: Raw embeddings don't separate classes well, so quantum kernel can't help.

---

## Root Cause Analysis

### Primary Issue: Embedding Quality
The fundamental problem is that **the embeddings themselves don't separate classes**:
- Mean differences between classes are tiny (0.000291 max)
- Separation ratio is barely above 1.0 (1.0314)
- Low diversity means many entities are identical

### Secondary Issue: Quantum Kernel Expressivity
Even with better embeddings, the quantum kernel may not be expressive enough:
- ZZ feature map with 3 reps may be too simple
- 12 qubits may not be enough for this problem
- Kernel similarities are all ~0.004 (very low)

---

## Recommendations

### Immediate Fixes (High Priority)

#### 1. **Improve Embedding Training**
```bash
# Use full-graph embeddings for richer context
--full_graph_embeddings

# Try different embedding methods
--embedding_method RotatE  # or DistMult

# Increase embedding dimension
--embedding_dim 128  # or 256
```

#### 2. **Fix Overfitting**
```bash
# Use cross-validation for evaluation
--use_cv_evaluation

# Reduce feature map complexity
--qml_feature_map_reps 2  # instead of 3

# Use simpler feature map
--qml_feature_map Z  # instead of ZZ
```

#### 3. **Increase Quantum Expressivity**
```bash
# More qubits
--qml_dim 16  # or 20

# More feature map repetitions
--qml_feature_map_reps 5

# Try different feature maps
--qml_feature_map Pauli
```

### Medium Priority

#### 4. **Better Feature Engineering**
```bash
# Use improved features (already available)
--use_improved_features

# But skip quantum feature selection (too slow, didn't help)
# Remove --quantum_feature_selection
```

#### 5. **Try Different Quantum Approaches**
```bash
# Use data re-uploading
--use_data_reuploading

# Try custom link prediction feature map
--qml_feature_map custom_link_prediction
```

### Long-term Solutions

#### 6. **Re-evaluate Quantum-Aware Embeddings**
The current implementation didn't help. Consider:
- Different loss function (not just separability)
- More epochs (100 may not be enough)
- Different learning rate
- Or skip it entirely if it doesn't help

#### 7. **Address Low Diversity**
- Check why embeddings are so similar
- Consider different negative sampling strategies
- Use more training epochs
- Try different embedding algorithms

---

## Recommended Next Steps

### Option 1: Focus on Embeddings (Recommended)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --full_graph_embeddings \
    --embedding_method RotatE \
    --embedding_dim 128 \
    --use_improved_features \
    --qml_dim 16 \
    --qml_feature_map_reps 2
```

### Option 2: Simplify Quantum Model
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_dim 8 \
    --qml_feature_map Z \
    --qml_feature_map_reps 2 \
    --use_cv_evaluation
```

### Option 3: Skip Quantum-Aware Embeddings
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_feature_engineering \
    --qml_dim 12
    # Remove --use_quantum_aware_embeddings
```

---

## Expected Improvements

### With Better Embeddings
- **Separation ratio**: 1.0314 → >1.1
- **Mean differences**: 0.000291 → >0.01
- **Kernel separability**: Should improve

### With More Qubits/Reps
- **Kernel expressivity**: Should increase
- **Similarity range**: Should widen beyond 0.004

### With Cross-Validation
- **Overfitting**: Should reduce
- **Generalization**: Should improve

---

## Key Takeaways

1. **Embedding quality is the bottleneck**: Fix embeddings first
2. **Quantum kernel can't fix bad embeddings**: Need good raw separability
3. **Overfitting is severe**: Need regularization or simpler models
4. **Quantum-aware embeddings didn't help**: May need different approach or skip
5. **Low diversity is concerning**: Many entities are identical

---

## Next Run Checklist

- [ ] Use full-graph embeddings
- [ ] Try RotatE or DistMult
- [ ] Increase embedding dimension
- [ ] Remove quantum-aware embeddings (didn't help)
- [ ] Use cross-validation
- [ ] Increase qubits to 16
- [ ] Reduce feature map reps to 2
- [ ] Monitor kernel separability metrics

