# Quantum-Native Improvements for QML

This document describes quantum-native techniques implemented to improve QML performance **without relying on classical features**.

## Problem

The quantum kernel was not separating classes well:
- **Kernel separability**: Within-class similarity ≈ Between-class similarity (all ~0.004)
- **Test PR-AUC**: 0.4855 (worse than random)
- **Root cause**: Standard feature maps (ZZFeatureMap) weren't capturing class-separable patterns

## Quantum-Native Solutions

### 1. **Data Re-Uploading Feature Maps** ⚡

**What it does**: Encodes features **multiple times** in the quantum circuit to increase expressivity without needing more qubits.

**Why it helps**: 
- Allows encoding high-dimensional features (e.g., 48D classical features) into fewer qubits (e.g., 12)
- Increases circuit expressivity without increasing qubit count
- Better captures complex relationships between head and tail embeddings

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_data_reuploading
```

**Reference**: Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier"

---

### 2. **Variational Feature Maps** ⚡

**What it does**: Uses **trainable parameters** in the feature encoding layer that can be optimized to maximize class separability.

**Why it helps**:
- Adapts the feature encoding to the specific problem
- Can learn optimal feature transformations for link prediction
- More flexible than fixed feature maps

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_variational_feature_map
```

**Note**: Requires additional optimization loop (can be slower but potentially more accurate)

---

### 3. **Custom Link Prediction Feature Map** 🎯

**What it does**: Specialized feature map designed specifically for link prediction tasks that emphasizes the relationship between head and tail entities.

**Why it helps**:
- Encodes head and tail embeddings separately but with strong entanglement
- Emphasizes the **relationship** between entities rather than treating them independently
- Domain-specific design for knowledge graph link prediction

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_feature_map custom_link_prediction
```

---

### 4. **Kernel-Target Alignment Optimization** 📊

**What it does**: Automatically finds the optimal number of feature map repetitions by maximizing kernel-target alignment.

**Why it helps**:
- Ensures the kernel aligns well with class labels
- Automatically tunes hyperparameters (reps) based on data
- Maximizes class separability in the kernel space

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --optimize_feature_map_reps
```

**How it works**:
1. Tries different numbers of repetitions (e.g., 1-5)
2. Computes kernel-target alignment for each
3. Selects the repetition count with highest alignment

---

## Recommended Combinations

### **Best for Expressivity** (most quantum-native)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_data_reuploading \
    --qml_feature_map_reps 5 \
    --qml_entanglement full
```

### **Best for Adaptability** (learns optimal encoding)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_variational_feature_map \
    --optimize_feature_map_reps
```

### **Best for Link Prediction** (domain-specific)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_feature_map custom_link_prediction \
    --qml_entanglement full \
    --qml_feature_map_reps 4
```

### **Best for Automatic Tuning** (let the algorithm decide)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --optimize_feature_map_reps \
    --qml_feature_map_reps 3  # Starting point
```

---

## Expected Improvements

With these quantum-native techniques, you should see:

1. **Better Kernel Separability**: 
   - Within-class similarity > Between-class similarity
   - Kernel-target alignment > 0.1 (vs. current ~0.0)

2. **Better Test Performance**:
   - PR-AUC > 0.55 (vs. current 0.4855)
   - More realistic training metrics (less overfitting)

3. **More Informative Kernels**:
   - Higher mean kernel similarity (vs. current 0.004)
   - Better distribution of similarities

---

## Implementation Details

### Files Created:
- `quantum_layer/quantum_feature_maps.py`: Advanced feature map implementations
- `quantum_layer/quantum_kernel_alignment.py`: Kernel optimization utilities

### Files Modified:
- `quantum_layer/qml_trainer.py`: Integrated new feature maps into QSVC training
- `scripts/run_optimized_pipeline.py`: Added command-line arguments

### Key Classes:
- `DataReuploadingFeatureMap`: Multi-layer feature encoding
- `VariationalFeatureMap`: Trainable feature encoding
- `LinkPredictionFeatureMap`: Domain-specific feature map
- `kernel_target_alignment()`: Measures kernel quality
- `optimize_feature_map_reps()`: Auto-tunes repetitions

---

## Next Steps

1. **Try data re-uploading first** (easiest, most likely to help):
   ```bash
   --use_data_reuploading
   ```

2. **If still not working, try custom link prediction feature map**:
   ```bash
   --qml_feature_map custom_link_prediction
   ```

3. **For best results, combine techniques**:
   ```bash
   --use_data_reuploading --optimize_feature_map_reps
   ```

4. **Monitor kernel diagnostics** to see if separability improves

---

## Comparison with Classical Features

| Approach | Pros | Cons |
|----------|------|------|
| **Classical Features** | Works well (PR-AUC ~0.61) | Not quantum-native, relies on classical feature engineering |
| **Data Re-Uploading** | Quantum-native, more expressive | Slightly slower (more circuit depth) |
| **Variational FM** | Adapts to problem, potentially best | Requires optimization, slower |
| **Custom Link Pred** | Domain-specific, designed for task | May need tuning |

**Recommendation**: Start with **data re-uploading** + **optimize_reps** for best balance of quantum-native approach and performance.

