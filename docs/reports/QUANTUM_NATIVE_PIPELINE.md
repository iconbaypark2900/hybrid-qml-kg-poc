# Quantum-Native Pipeline Guide

Complete guide to using quantum-specific feature engineering and quantum-aware embeddings.

## Overview

This pipeline provides **quantum-native** approaches specifically designed for quantum machine learning models:

1. **Quantum-Aware Embeddings**: Fine-tune embeddings using quantum kernel separability as the objective
2. **Quantum Feature Engineering**: Create features optimized for quantum circuits (amplitude, phase, entanglement)

---

## Quantum-Aware Embeddings

### What It Does

Fine-tunes embeddings specifically for quantum models by:
- Using quantum kernel separability as the loss function
- Maximizing class separation in quantum feature space
- Optimizing embeddings for the specific quantum feature map you'll use

### Why It's Better

- **Classical embeddings** are trained for ranking (link prediction), not classification
- **Quantum-aware embeddings** are optimized for quantum kernel separability
- **Direct optimization** of the metric that matters for quantum models

### Usage

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_aware_embeddings \
    --quantum_aware_epochs 100 \
    --qml_dim 12 \
    --qml_feature_map ZZ \
    --qml_feature_map_reps 3
```

### Parameters

- `--use_quantum_aware_embeddings`: Enable quantum-aware fine-tuning
- `--quantum_aware_epochs`: Number of fine-tuning epochs (default: 100)
- Uses same quantum feature map parameters as your QML model (`--qml_dim`, `--qml_feature_map`, etc.)

---

## Quantum Feature Engineering

### What It Does

Creates quantum-native features optimized for quantum circuits:

1. **Amplitude Encoding Features**: Normalized features for quantum amplitude encoding
2. **Phase Encoding Features**: Phase/angle features for quantum phase encoding
3. **Entanglement Features**: Correlation features inspired by quantum entanglement
4. **Quantum Distance Features**: Fidelity-like measures between head and tail embeddings

### Why It's Better

- **Quantum-native**: Features designed specifically for quantum circuits
- **Better encoding**: Optimized for amplitude/phase encoding strategies
- **Entanglement-aware**: Features that leverage quantum entanglement patterns

### Usage

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_feature_engineering \
    --quantum_feature_selection \
    --qml_dim 12
```

### Parameters

- `--use_quantum_feature_engineering`: Enable quantum feature engineering
- `--quantum_feature_selection`: Use quantum kernel for feature selection (optional)

---

## Complete Quantum-Native Pipeline

### Recommended Configuration

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_aware_embeddings \
    --quantum_aware_epochs 150 \
    --use_quantum_feature_engineering \
    --quantum_feature_selection \
    --qml_dim 12 \
    --qml_feature_map ZZ \
    --qml_feature_map_reps 3 \
    --qml_entanglement full \
    --use_data_reuploading
```

### Step-by-Step Explanation

1. **Train Base Embeddings**: Standard ComplEx/RotatE embeddings
2. **Quantum-Aware Fine-Tuning**: Optimize embeddings for quantum kernel separability
3. **Quantum Feature Engineering**: Create quantum-native features
4. **Quantum Feature Selection**: Select features using quantum kernel importance
5. **Train QSVC**: Use optimized embeddings and features

---

## Comparison: Classical vs Quantum-Native

### Classical Approach
```
Base Embeddings → Classical Features → PCA → Quantum Model
```
- Embeddings optimized for ranking, not classification
- Features designed for classical models
- May not be optimal for quantum circuits

### Quantum-Native Approach
```
Base Embeddings → Quantum-Aware Fine-Tuning → Quantum Features → Quantum Model
```
- Embeddings optimized for quantum kernel separability
- Features designed for quantum circuits
- End-to-end optimization for quantum models

---

## Expected Improvements

### Quantum-Aware Embeddings
- **Embedding separability**: Should increase from ~1.0 → >1.1
- **Kernel separability**: Within-class > Between-class similarity
- **Model performance**: Should improve PR-AUC significantly

### Quantum Feature Engineering
- **Feature quality**: Better features for quantum encoding
- **Feature selection**: Quantum kernel-based selection
- **Model performance**: Better input features → better model

---

## Troubleshooting

### Quantum-Aware Embeddings Fail
- **Check Qiskit installation**: `pip install qiskit qiskit-machine-learning`
- **Check PyTorch installation**: `pip install torch`
- **Reduce epochs**: Try `--quantum_aware_epochs 50` for faster testing
- **Check memory**: Quantum kernels can be memory-intensive

### Quantum Feature Engineering Fails
- **Check Qiskit installation**: `pip install qiskit qiskit-machine-learning`
- **Reduce feature count**: Use `--quantum_feature_selection` to limit features
- **Check embeddings**: Ensure embeddings are loaded correctly

### Performance Not Improving
- **Increase epochs**: Try `--quantum_aware_epochs 200`
- **Try different feature maps**: `--qml_feature_map Pauli` or `Z`
- **Increase qubits**: `--qml_dim 16` or `20`
- **Use data re-uploading**: `--use_data_reuploading`

---

## Advanced Usage

### Combine with Other Techniques

```bash
# Quantum-aware + Contrastive Learning
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_contrastive_learning \
    --use_quantum_aware_embeddings \
    --use_quantum_feature_engineering
```

### Full-Graph Embeddings + Quantum-Aware

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --full_graph_embeddings \
    --use_quantum_aware_embeddings \
    --quantum_aware_epochs 200 \
    --use_quantum_feature_engineering
```

### Different Feature Maps

```bash
# Try Pauli feature map
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_quantum_aware_embeddings \
    --qml_feature_map Pauli \
    --use_quantum_feature_engineering
```

---

## Next Steps

1. **Start with quantum-aware embeddings**: Most impactful improvement
2. **Add quantum feature engineering**: Further optimize features
3. **Experiment with parameters**: Find optimal settings for your data
4. **Compare results**: Measure improvements over baseline

---

## References

- **Quantum Kernel Methods**: [Qiskit Machine Learning Documentation](https://qiskit.org/ecosystem/machine-learning/)
- **Quantum Feature Maps**: [Qiskit Circuit Library](https://qiskit.org/documentation/apidoc/circuit_library.html)
- **Amplitude Encoding**: Standard quantum encoding strategy
- **Phase Encoding**: Alternative quantum encoding strategy

