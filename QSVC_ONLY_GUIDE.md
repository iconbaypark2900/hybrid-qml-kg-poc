# Running QSVC Only - Quick Guide

## ✅ Yes! You can now run just QSVC

Added a new `--quantum_only` flag that skips all classical models and runs only quantum models (QSVC and optionally VQC).

---

## Quick Command

### Run QSVC Only (Fastest)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --quantum_only
```

This will:
- ✅ Skip all classical models (RandomForest, SVM, LogisticRegression, Ensemble)
- ✅ Run QSVC only
- ✅ Skip VQC (for faster iteration)
- ⚡ Much faster - only ~15-20 seconds instead of minutes!

---

## Options

### QSVC Only (Skip VQC too)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --quantum_only \
    --fast_mode
```

### QSVC + VQC (Both Quantum Models)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --quantum_only
# (VQC runs automatically if not in fast_mode)
```

### Experiment with QSVC Parameters
```bash
# Try more qubits
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --quantum_only \
    --qml_dim 15

# Try Pauli feature map
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --quantum_only \
    --qml_feature_map Pauli

# Try more repetitions
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --quantum_only \
    --qml_feature_map_reps 4

# Try different encoding
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --quantum_only \
    --qml_encoding optimized_diff
```

---

## What Gets Skipped

When using `--quantum_only`:
- ❌ RandomForest training
- ❌ SVM-RBF training
- ❌ SVM-Linear training
- ❌ LogisticRegression training
- ❌ Ensemble training
- ❌ Feature importance analysis (from RandomForest)
- ✅ Embedding training/loading (still needed)
- ✅ Classical feature building (still needed for quantum feature prep)
- ✅ Quantum feature engineering
- ✅ QSVC training
- ✅ VQC training (unless `--fast_mode`)

---

## Speed Comparison

| Mode | Time | Models |
|------|------|--------|
| **Full Pipeline** | ~2-5 minutes | All classical + quantum |
| **Quantum Only** | ~15-20 seconds | QSVC only |
| **Quantum Only + VQC** | ~8-10 minutes | QSVC + VQC |

---

## Use Cases

### ✅ Use `--quantum_only` when:
- Testing QSVC improvements quickly
- Experimenting with quantum parameters
- Iterating on quantum feature engineering
- You only care about quantum model performance
- You want fast feedback loops

### ❌ Don't use `--quantum_only` when:
- You need to compare quantum vs classical
- You want feature importance analysis
- You need ensemble predictions
- You're doing final evaluation

---

## Example Output

When running with `--quantum_only`, you'll see:

```
================================================================================
STEP 1: LOADING DATA
================================================================================
...

================================================================================
STEP 2: TRAINING KNOWLEDGE GRAPH EMBEDDINGS
================================================================================
...

================================================================================
STEP 3: BUILDING ENHANCED FEATURES
================================================================================
...

================================================================================
STEP 5: PREPARING QUANTUM FEATURES
================================================================================
...

================================================================================
STEP 6: TRAINING QUANTUM MODELS
================================================================================
Training QSVC...
QSVC Configuration:
  Qubits: 12
  Feature Map: ZZ
  Repetitions: 3
  Entanglement: full
  Encoding Strategy: hybrid
...
  ✅ QSVC - Test PR-AUC: 0.5518

================================================================================
COMPREHENSIVE COMPARISON REPORT
================================================================================
Rank   | Model          | Type    | PR-AUC     | Accuracy   | Time (s)  
--------------------------------------------------------------------------------
1      | QSVC-Optimized | quantum | 0.5518     | 0.5298     | 13.99      
```

---

## Tips

1. **Use cached embeddings**: Always use `--use_cached_embeddings` for faster runs
2. **Skip VQC**: Use `--fast_mode` with `--quantum_only` to skip slow VQC
3. **Experiment freely**: Try different parameters without waiting for classical models
4. **Compare later**: Run full pipeline occasionally to compare against classical baselines

---

## Quick Reference

```bash
# Fastest: QSVC only
--quantum_only --use_cached_embeddings

# QSVC with custom params
--quantum_only --qml_dim 15 --qml_feature_map Pauli

# Both quantum models
--quantum_only  # (VQC runs automatically)

# QSVC only (explicit)
--quantum_only --fast_mode
```

