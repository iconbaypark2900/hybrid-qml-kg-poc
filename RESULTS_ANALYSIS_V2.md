# Results Analysis: Contrastive Learning + Improved Features

## Summary

✅ **Contrastive learning ran successfully** - Loss decreased from 0.999449 → 0.980838  
✅ **Improved features created** - 100 interaction features + 4 class difference features  
✅ **Quantum feature separability improved** - Ratio increased from 1.0018 → 1.0474  
❌ **Kernel separability still poor** - Within-class ≈ between-class similarity  
❌ **Performance unchanged** - Test PR-AUC: 0.5306 (same as before)

---

## Key Metrics

### Contrastive Learning
- **Loss reduction**: 0.999449 → 0.980838 (1.9% improvement)
- **Epochs**: 100
- **Margin**: 1.5
- **Status**: ✅ Completed successfully

### Feature Engineering
- **Base features**: 1163 (after variance filtering)
- **Interaction features**: 100 (product, ratio, difference, polynomial)
- **Class difference features**: 4 (distances to centroids)
- **Total features**: 1267
- **Status**: ✅ Created successfully

### Embedding Separability
- **Raw embedding separation ratio**: 1.0012 (slight improvement from 1.0017)
- **Quantum feature separation ratio**: 1.0474 (improved from 1.0018)
- **Significant features**: 7/12 (58%)
- **Silhouette score**: 0.0056 (still highly overlapping)

### Kernel Separability
- **Positive-Positive similarity**: 0.753518 ± 0.095471
- **Negative-Negative similarity**: 0.752439 ± 0.096293
- **Positive-Negative similarity**: 0.752578 ± 0.095638
- **Status**: ⚠️ **Kernel does NOT separate classes well**

### Model Performance
- **Train PR-AUC**: 0.6505
- **Test PR-AUC**: 0.5306
- **Test Accuracy**: 0.5331
- **Status**: ❌ No improvement over baseline

---

## Root Cause Analysis

### The Problem

Despite improvements in:
1. ✅ Contrastive learning (loss decreased)
2. ✅ Quantum feature separability (ratio improved)
3. ✅ Feature engineering (more features)

**The kernel matrix still shows no class separation**:
- Within-class similarity ≈ Between-class similarity
- All three similarity measures are nearly identical (~0.752-0.753)
- This means the quantum kernel cannot distinguish positive from negative samples

### Why Kernel Separability Matters

The quantum kernel is the core of QSVC. If the kernel doesn't separate classes, the SVM cannot learn a good decision boundary, regardless of:
- How many features we add
- How well embeddings are fine-tuned
- How many qubits we use

**The kernel reflects the fundamental separability of the input features in quantum feature space.**

---

## What Worked

1. **Contrastive Learning**: Successfully fine-tuned embeddings (loss decreased)
2. **Improved Features**: Successfully created interaction and class difference features
3. **Quantum Feature Separability**: Improved from 1.0018 → 1.0474 (23% improvement)

## What Didn't Work

1. **Kernel Separability**: Still shows no class separation
2. **Model Performance**: No improvement (still ~0.53 PR-AUC)
3. **Contrastive Learning Impact**: Only 1.9% loss reduction (may need more epochs/margin)

---

## Recommendations

### 1. Increase Contrastive Learning Intensity ⚡

The loss reduction was only 1.9%. Try:
- **More epochs**: 200-300 instead of 100
- **Larger margin**: 2.0-3.0 instead of 1.5
- **Better sampling**: Hard negative mining during contrastive learning

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_contrastive_learning \
    --contrastive_margin 2.5 \
    --contrastive_epochs 200 \
    --use_improved_features \
    --use_data_reuploading \
    --qml_feature_map custom_link_prediction
```

### 2. Try Different Feature Maps 🔄

The `custom_link_prediction` feature map may not be optimal. Try:
- **ZZ with more reps**: `--qml_feature_map ZZ --qml_feature_map_reps 5`
- **Pauli feature map**: `--qml_feature_map Pauli`
- **Optimize reps**: `--optimize_feature_map_reps`

### 3. Use Full-Graph Embeddings 🌐

Current embeddings are trained only on CtD edges. Try:
```bash
--full_graph_embeddings
```
This trains embeddings on all Hetionet relations, providing richer context.

### 4. Increase Qubits 📈

More qubits = more expressivity:
```bash
--qml_dim 16  # or 20
```

### 5. Try Different Embedding Methods 🔬

ComplEx may not be optimal. Try:
- **RotatE**: Better for complex relations
- **TransE**: Simpler but sometimes more effective

### 6. Hybrid Approach: Use Classical Features in Kernel 🎯

Try using classical features (reduced via PCA) in the quantum kernel:
```bash
--use_classical_features_in_kernel
```

### 7. Hard Negative Sampling 🎲

Use hard negatives to make the task more challenging:
```bash
--negative_sampling hard
```

---

## Next Steps (Priority Order)

### High Priority
1. **Increase contrastive learning intensity** (more epochs, larger margin)
2. **Try full-graph embeddings** (richer context)
3. **Experiment with different feature maps** (ZZ with more reps, Pauli)

### Medium Priority
4. **Increase qubits** (16-20 instead of 12)
5. **Try different embedding methods** (RotatE, TransE)
6. **Use classical features in kernel** (hybrid approach)

### Low Priority
7. **Hard negative sampling** (more challenging negatives)
8. **Optimize feature map reps** (kernel-target alignment)

---

## Expected Improvements

If contrastive learning and full-graph embeddings work:
- **Embedding separation ratio**: 1.0012 → >1.1 (target)
- **Kernel separability**: Within-class > Between-class (target)
- **Test PR-AUC**: 0.5306 → >0.60 (target)

---

## Conclusion

The improvements (contrastive learning, improved features) are working at the feature level, but **the quantum kernel still cannot separate classes**. This is the bottleneck.

**Focus on improving kernel separability** through:
1. Better embeddings (full-graph, more contrastive learning)
2. Better feature maps (more reps, different types)
3. More qubits (more expressivity)

The fact that quantum feature separability improved (1.0018 → 1.0474) is promising, but we need to push it further (>1.1) and ensure the kernel reflects this improvement.

