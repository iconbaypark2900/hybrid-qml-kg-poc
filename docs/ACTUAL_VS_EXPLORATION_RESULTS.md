# Actual vs Exploration Results: Critical Discrepancy

## Summary

**Major discrepancy discovered**: The exploration script predicted **0.9510 PR-AUC** for quantum models, but the actual quantum kernel achieved only **0.3729 PR-AUC** (worse than random!).

---

## Results Comparison

### Exploration Predictions (RBF Kernel Proxy)
- **Test PR-AUC**: 0.9510 (95.1%) ✅
- **CV PR-AUC**: 0.9771 (97.7%) ✅
- **Overfitting Gap**: 0.0450 (4.5%) ✅
- **Status**: Excellent!

### Actual Quantum Kernel Results
- **QSVC Test PR-AUC**: 0.3729 (37.3%) ❌
- **QSVC Train PR-AUC**: 0.6513 (65.1%)
- **QSVC ROC-AUC**: 0.4835 (worse than random 0.5!) ❌
- **Overfitting Gap**: 0.2784 (27.8%) ⚠️
- **Status**: **Terrible - worse than random!**

### Classical Models (Actual Results)
- **LogisticRegression-L2**: 0.8859 (88.6%) ✅ **Excellent!**
- **RandomForest-Optimized**: 0.7586 (75.9%) ✅
- **Ensemble-RF-LR**: 0.7498 (75.0%) ✅

---

## Root Cause Analysis

### 1. RBF Kernel Proxy Was Misleading

**Problem**: The exploration script used RBF kernel as a fast proxy for quantum kernel performance.

**Why it failed**:
- RBF kernel: `K(x, y) = exp(-γ||x-y||²)` (classical, smooth, well-behaved)
- Quantum kernel: `K(x, y) = |⟨φ(x)|φ(y)⟩|²` (quantum, can have different properties)
- **Quantum kernels can have very different separability properties** than RBF kernels

**Evidence**:
- RBF proxy: 0.9510 PR-AUC
- Actual quantum: 0.3729 PR-AUC
- **Gap: 0.5781 (57.8% difference!)**

### 2. Quantum Kernel Separability Failure

**Metrics from actual run**:
- **ROC-AUC**: 0.4835 (worse than random 0.5!)
- **Quantum feature separation ratio**: 1.1158 (moderate)
- **Silhouette score**: 0.0985 (highly overlapping)
- **Significant features**: 5/12 (42%)

**Problem**: Despite moderate separation ratio (1.1158), the quantum kernel cannot distinguish classes effectively.

### 3. Overfitting in Quantum Model

- **Train PR-AUC**: 0.6513
- **Test PR-AUC**: 0.3729
- **Gap**: 0.2784 (27.8%) ⚠️ **Severe overfitting!**

The quantum model is memorizing training data but failing to generalize.

---

## Why Quantum Kernel Performs Poorly

### 1. **Low Embedding Diversity**
- Head diversity: 177/825 (21.5%)
- Tail diversity: 44/825 (5.3%)
- Many entities share identical embeddings → reduces information content

### 2. **Quantum Feature Quality**
- Only 5/12 features are statistically significant
- Silhouette score: 0.0985 (highly overlapping)
- Separation ratio: 1.1158 (barely above 1.0)

### 3. **Kernel Matrix Properties**
- Quantum kernels can have different properties than classical kernels
- May not capture the same patterns as RBF kernel
- Quantum feature maps may not be expressive enough for this task

### 4. **Feature Engineering Mismatch**
- Pre-PCA: 768D → 128D
- Feature selection: 24 features
- Final PCA: 24D → 12D
- **Information loss** through multiple reduction steps

---

## What Worked Well

### ✅ Classical Models
- **LogisticRegression**: 0.8859 PR-AUC (excellent!)
- **RandomForest**: 0.7586 PR-AUC (good)
- **Ensemble**: 0.7498 PR-AUC (good)

### ✅ Feature Engineering
- 2329 features → 2313 after variance filtering
- Graph features (shortest_path) most important
- Embedding features contribute significantly

### ✅ Embedding Training
- RotatE embeddings: MRR=0.8911, Hits@10=0.9799
- Contrastive learning: Loss decreased
- Task-specific fine-tuning: Validation AUC=0.8750

---

## Lessons Learned

### 1. **RBF Kernel Proxy is Not Reliable**
- Cannot accurately predict quantum kernel performance
- Quantum kernels have fundamentally different properties
- Need to test with actual quantum kernels for accurate results

### 2. **Quantum Models Need Better Features**
- Current quantum features (12D) may be too low-dimensional
- Multiple reduction steps lose information
- Need to preserve more information for quantum kernels

### 3. **Classical Models Are More Reliable**
- LogisticRegression achieves 0.8859 PR-AUC
- More stable and predictable performance
- Better generalization

### 4. **Hybrid Ensemble Doesn't Help**
- Hybrid ensemble: 0.7421 PR-AUC
- Worse than best classical (0.8859)
- Quantum component (0.3729) drags down performance

---

## Recommendations

### Immediate Actions

1. **Focus on Classical Models**
   - LogisticRegression at 0.8859 is excellent
   - Already exceeds 0.80 target ✅
   - More reliable and interpretable

2. **Fix Quantum Feature Engineering**
   - Reduce information loss in dimensionality reduction
   - Try fewer reduction steps
   - Preserve more dimensions (e.g., 24D instead of 12D)

3. **Improve Embedding Diversity**
   - Current diversity is very low (21.5% head, 5.3% tail)
   - Need better embedding training
   - Consider different embedding methods

4. **Test Quantum Kernels Directly**
   - Don't rely on RBF proxy
   - Test actual quantum kernels during exploration
   - May need to accept slower exploration for accuracy

### Long-term Improvements

1. **Quantum-Aware Feature Engineering**
   - Design features specifically for quantum kernels
   - Consider quantum kernel properties when reducing dimensions
   - Test different quantum feature maps

2. **Hybrid Approaches**
   - Use quantum models only when they add value
   - Current hybrid ensemble is worse than classical alone
   - Need better integration strategy

3. **Better Exploration**
   - Test actual quantum kernels (slower but accurate)
   - Use smaller parameter grids
   - Focus on configurations that work with quantum kernels

---

## Conclusion

**The exploration script's predictions were misleading** due to using RBF kernel as a proxy. The actual quantum kernel performs much worse (0.3729 vs predicted 0.9510).

**Classical models are the clear winners**:
- LogisticRegression: 0.8859 PR-AUC ✅
- Already exceeds 0.80 target ✅
- More reliable and interpretable ✅

**Next steps**: Focus on improving classical models further or fixing quantum feature engineering to make quantum models competitive.
