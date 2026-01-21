# Why Quantum Models Underperform: Root Cause Analysis

## Performance Comparison

| Model Type | PR-AUC | ROC-AUC | Status |
|------------|--------|---------|--------|
| **Classical (LogisticRegression)** | **0.8859** | ~0.90 | ✅ Excellent |
| **Classical (RandomForest)** | **0.7586** | ~0.91 | ✅ Good |
| **Quantum (Best)** | **0.6081** | 0.7915 | ⚠️ Moderate |
| **Gap** | **-0.2778** | -0.11 | ❌ Significant |

---

## Critical Issues Identified

### 1. **Severe Overfitting** ⚠️⚠️⚠️

**The smoking gun:**
- **Train PR-AUC**: 1.0000 (perfect!)
- **Test PR-AUC**: 0.6081
- **Overfitting Gap**: 0.3919 (39.2%!)

**What this means:**
- The quantum model is **memorizing the training data perfectly**
- But it **cannot generalize** to new data
- This is a classic sign of overfitting

**Why it happens:**
- Quantum kernels can be very expressive (especially with full entanglement)
- With only 825 training samples, the model has enough capacity to memorize
- The 12D quantum feature space may be too complex for the amount of data

---

### 2. **Low Embedding Diversity** ⚠️

**From diagnostics:**
- **Head diversity**: 177/825 (21.5%) - only 21.5% unique embeddings!
- **Tail diversity**: 44/825 (5.3%) - only 5.3% unique embeddings!

**What this means:**
- Many entities share **identical embeddings**
- Reduces information content dramatically
- Quantum kernel can't distinguish between entities with same embeddings

**Example:**
- If 100 compounds have the same embedding, the quantum kernel sees them as identical
- Can't learn patterns when inputs are identical

---

### 3. **Information Loss Through Dimensionality Reduction** ⚠️

**Reduction pipeline:**
1. **Raw embeddings**: 256D (RotatE complex → real)
2. **Pre-PCA**: 256D → 128D (or 64D) - **50-75% information lost**
3. **Feature selection**: 128D → 24 features - **81% information lost**
4. **Final PCA**: 24D → 12D - **50% information lost**

**Total information loss:**
- Starting: 256 dimensions
- Ending: 12 dimensions
- **95.3% of information discarded!**

**Classical models:**
- Use all 2313 features (after variance filtering)
- No information loss
- Can capture complex patterns

---

### 4. **Quantum Kernel Properties** ⚠️

**Quantum kernels are fundamentally different:**
- **Classical RBF**: `K(x,y) = exp(-γ||x-y||²)` - smooth, well-behaved
- **Quantum**: `K(x,y) = |⟨φ(x)|φ(y)⟩|²` - can have different properties

**Issues:**
- Quantum kernels may not capture the same patterns as classical kernels
- Feature maps (ZZ with 2 reps) may not be expressive enough
- Or may be too expressive (causing overfitting)

**Evidence:**
- RBF proxy predicted 0.9510 PR-AUC
- Actual quantum kernel: 0.6081 PR-AUC
- **Gap: 0.3429 (36% difference!)**

---

### 5. **Feature Quality Mismatch** ⚠️

**Classical models use:**
- **2313 features** including:
  - Graph features (shortest_path, preferential_attachment, etc.)
  - Domain features (shared genes, etc.)
  - Embedding features (all 256 dimensions)
  - Interaction features

**Quantum models use:**
- **Only 12 dimensions** after aggressive reduction
- **No graph features** (can't encode in quantum space)
- **No domain features** (can't encode in quantum space)
- **Only embedding information** (heavily reduced)

**Result:**
- Classical models have **193x more information** (2313 vs 12)
- Classical models can use domain knowledge (graph structure, shared genes)
- Quantum models are information-starved

---

### 6. **Small Dataset Size** ⚠️

**Dataset:**
- **Training samples**: 825
- **Test samples**: 207
- **Total**: 1032 samples

**Why this hurts quantum:**
- Quantum models need more data to learn generalizable patterns
- With only 825 training samples, the model memorizes instead of learning
- Classical models are more data-efficient

**Evidence:**
- Train PR-AUC = 1.0 (perfect memorization)
- Test PR-AUC = 0.6081 (poor generalization)

---

## Why Classical Models Perform Better

### 1. **More Information**
- 2313 features vs 12 dimensions
- Can use graph structure, domain knowledge, full embeddings

### 2. **Better Regularization**
- Classical models have built-in regularization (L2, tree depth limits)
- Quantum models struggle with overfitting (train=1.0, test=0.6081)

### 3. **Feature Engineering**
- Graph features (shortest_path) are most important
- These can't be encoded in quantum space
- Classical models can leverage them directly

### 4. **Data Efficiency**
- Classical models are more data-efficient
- Can learn from 825 samples without severe overfitting
- Quantum models need more data or better regularization

---

## What Would Help Quantum Models

### 1. **Reduce Overfitting**
- **More regularization**: Higher C values, simpler feature maps
- **More data**: Increase training samples
- **Simpler models**: Fewer qubits, fewer feature map repetitions

### 2. **Improve Embedding Diversity**
- Better embedding training
- Different embedding methods
- Task-specific fine-tuning (already tried, but may need more)

### 3. **Preserve More Information**
- **Reduce dimensionality less aggressively**: 24D or 32D instead of 12D
- **Skip pre-PCA**: Go directly from 256D → 24D → 12D
- **Use more features**: Select 48 features instead of 24

### 4. **Hybrid Approach**
- **Combine quantum and classical features**: Use quantum kernel on quantum features + classical features
- **Stack models**: Use quantum features as input to classical models
- **Feature fusion**: Concatenate quantum and classical features

### 5. **Better Quantum Feature Maps**
- Try different feature maps (Pauli, custom)
- Optimize feature map repetitions
- Use quantum-aware feature selection

---

## Recommendations

### Immediate Actions

1. **Focus on Classical Models** ✅
   - Already achieving 0.8859 PR-AUC
   - More reliable and interpretable
   - Better generalization

2. **If Improving Quantum:**
   - **Increase dimensions**: Try 24D or 32D instead of 12D
   - **Reduce overfitting**: Use simpler feature maps, more regularization
   - **Improve embeddings**: Focus on increasing diversity
   - **Hybrid features**: Combine quantum and classical features

3. **Accept Reality:**
   - Quantum models may not be suitable for this task
   - Classical models are already excellent
   - Quantum advantage may require different problem structure

---

## Conclusion

**Quantum models underperform because:**
1. **Severe overfitting** (train=1.0, test=0.6081)
2. **Low embedding diversity** (21.5% head, 5.3% tail)
3. **Information loss** (95% discarded through reduction)
4. **Feature mismatch** (12D vs 2313 features)
5. **Small dataset** (825 samples insufficient for quantum)

**Classical models excel because:**
1. **More information** (2313 features)
2. **Better regularization** (less overfitting)
3. **Domain knowledge** (graph features, shared genes)
4. **Data efficiency** (work well with 825 samples)

**Bottom line**: For this task and dataset size, classical models are the better choice. Quantum models would need significant improvements (more data, better embeddings, less reduction) to be competitive.
