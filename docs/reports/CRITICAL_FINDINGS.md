# Critical Findings: Performance Degradation

## Summary

**Performance is getting WORSE**:
- Baseline: 0.5306 PR-AUC
- Full-graph + contrastive: 0.5037 PR-AUC ❌
- ZZ optimized reps: 0.4927 PR-AUC ❌❌

---

## Run 1: Full-Graph Embeddings + Contrastive Learning

### Configuration
- Full-graph embeddings: ✅ (116,983 edges, 14 relation types)
- Contrastive learning: ✅ (margin=2.0, epochs=150)
- Improved features: ✅
- Custom link prediction feature map: ✅

### Results
- **Test PR-AUC: 0.5037** (worse than baseline!)
- **Raw embedding separation ratio: 0.9933** ❌ (BELOW 1.0 - classes NOT separated!)
- **Quantum feature separation ratio: 0.9992** ❌ (still below 1.0)
- **Contrastive loss: 1.001165 → 0.998822** (barely changed, 0.2% improvement)
- **Kernel separability: Still poor** (within-class ≈ between-class)

### Key Issues
1. **Full-graph embeddings made separability WORSE** - ratio dropped from 1.0012 → 0.9933
2. **Contrastive learning ineffective** - loss barely changed despite 150 epochs
3. **Embeddings are LESS separable** after full-graph training

---

## Run 2: ZZ Feature Map with Optimized Reps

### Configuration
- Task-specific embeddings: ✅
- Contrastive learning: ✅ (margin=1.0, epochs=50)
- Improved features: ✅
- ZZ feature map: ✅
- Optimized reps: ✅ (selected reps=8, alignment=0.0287)

### Results
- **Test PR-AUC: 0.4927** (even worse!)
- **Train PR-AUC: 1.0000** (perfect!)
- **Severe overfitting**: Train accuracy=100%, Test accuracy=51%
- **Kernel values near-zero**: mean=0.000682, max=0.035496
- **Kernel separability: Still poor** (within-class ≈ between-class)

### Key Issues
1. **Kernel is degenerate** - values are extremely small (near-zero)
2. **Severe overfitting** - perfect train performance, random test performance
3. **Optimized reps (8) made things worse** - kernel became too expressive, memorized training data
4. **Kernel condition number: 2.13** (very low, indicating near-identity matrix)

---

## Root Cause Analysis

### Problem 1: Embeddings Lack Class Separability

**Evidence**:
- Raw embedding separation ratio: 0.9933-1.0019 (barely above random)
- Only 1/256 features have mean difference > 0.01
- Contrastive learning barely helps (0.2% loss reduction)

**Why**:
- Embeddings trained for link prediction (ranking) don't learn classification boundaries
- Full-graph embeddings add noise without improving separability
- Contrastive learning needs more aggressive training or different approach

### Problem 2: Quantum Kernel Degeneracy

**Evidence**:
- Kernel values near-zero (mean=0.000682)
- Perfect train performance, random test performance
- Condition number: 2.13 (too low, near-identity)

**Why**:
- Too many feature map repetitions (reps=8) → kernel memorizes training data
- Kernel becomes too expressive → overfitting
- Kernel values too small → numerical instability

### Problem 3: Feature Map Selection

**Evidence**:
- Custom link prediction: Better kernel values but still no separability
- ZZ with many reps: Degenerate kernel, severe overfitting
- Both show poor kernel separability

**Why**:
- Feature maps can't create separability if input features aren't separable
- More reps ≠ better performance (can cause overfitting)
- Need to focus on improving input separability first

---

## Critical Insights

### 1. Full-Graph Embeddings Are NOT Helping
- **Finding**: Full-graph embeddings made separability WORSE (0.9933 vs 1.0012)
- **Conclusion**: More data ≠ better embeddings for classification
- **Action**: Stick with task-specific embeddings

### 2. Contrastive Learning Is Ineffective
- **Finding**: Loss barely changes (0.2% improvement) despite 150 epochs
- **Conclusion**: Current contrastive learning approach isn't working
- **Action**: Need different approach or more aggressive training

### 3. Kernel Optimization Backfired
- **Finding**: Optimized reps (8) caused severe overfitting
- **Conclusion**: More repetitions ≠ better performance
- **Action**: Use fewer reps (2-3), focus on input separability

### 4. The Real Problem: Input Separability
- **Finding**: All approaches fail because embeddings aren't separable
- **Conclusion**: Can't fix kernel separability without fixing input separability
- **Action**: Focus on improving embedding training, not kernel optimization

---

## Recommendations

### Immediate Actions (Priority 1)

1. **Revert to Baseline Configuration**
   ```bash
   python scripts/run_optimized_pipeline.py \
       --relation CtD \
       --quantum_only \
       --qml_feature_map ZZ \
       --qml_feature_map_reps 3 \
       --qml_dim 12
   ```
   - Remove full-graph embeddings
   - Remove contrastive learning (for now)
   - Use standard ZZ feature map with 3 reps

2. **Focus on Embedding Quality**
   - Try different embedding methods (RotatE, TransE)
   - Increase embedding dimension (128 → 256)
   - Train for more epochs (100 → 200)
   - Use different negative sampling (hard negatives)

### Medium-Term Actions (Priority 2)

3. **Improve Contrastive Learning**
   - Use harder negatives during contrastive training
   - Increase margin significantly (2.0 → 5.0)
   - Use different loss function (e.g., contrastive loss with temperature)
   - Train contrastive learning for more epochs (150 → 300)

4. **Better Feature Engineering**
   - Use RandomForest feature importances more effectively
   - Create more targeted interaction features
   - Use domain knowledge to create better features

### Long-Term Actions (Priority 3)

5. **Alternative Approaches**
   - Try VQC (Variational Quantum Classifier) instead of QSVC
   - Use hybrid quantum-classical models
   - Try different quantum encodings (amplitude, phase)
   - Consider classical models if quantum doesn't improve

---

## Expected Improvements

If we fix embedding separability:
- **Raw embedding separation ratio**: 0.9933 → >1.1 (target)
- **Quantum feature separation ratio**: 0.9992 → >1.1 (target)
- **Kernel separability**: Within-class > Between-class (target)
- **Test PR-AUC**: 0.5037 → >0.60 (target)

---

## Conclusion

**The fundamental problem is embedding separability, not kernel optimization.**

All our improvements (full-graph embeddings, contrastive learning, optimized feature maps) failed because:
1. Embeddings aren't separable enough (ratio < 1.0)
2. Kernel can't create separability from non-separable inputs
3. Over-optimization causes overfitting

**Next steps**: Focus on improving embedding training quality, not kernel optimization.

