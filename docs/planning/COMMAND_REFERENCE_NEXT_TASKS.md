# Command Reference: Next Tasks Implementation

**Date:** March 6, 2026  
**Status:** All high-priority tasks completed

---

## Quick Start: Reproduce Best Results

### Baseline (Best Result: PR-AUC 0.7987)
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --optimize_feature_map_reps --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24 --fast_mode
```

---

## 1. Multi-Model Fusion (NEW FEATURE)

### Test All Fusion Methods
```bash
python scripts/test_multi_model_fusion.py
```

### Use in Your Pipeline
```python
from quantum_layer.multi_model_fusion import create_fusion_ensemble

# Prepare predictions from multiple models
model_predictions = {
    'quantum': quantum_pred_proba,      # From QSVC
    'random_forest': rf_pred_proba,
    'extra_trees': et_pred_proba,
    'gradient_boosting': gb_pred_proba
}

# Create fusion ensemble with optimized weights
fusion, metrics = create_fusion_ensemble(
    model_predictions,
    y_train,
    fusion_method='optimized_weights'  # Options: weighted_average, optimized_weights, 
                                       #          bayesian_averaging, rank_fusion, 
                                       #          confidence_weighted, neural_metalearner
)

# Get fused predictions
fused_pred = fusion.predict(test_predictions)

# Evaluate
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_test, fused_pred)
print(f"Fused PR-AUC: {pr_auc:.4f}")
```

### Recommended Fusion Methods
| Method | Best For | Runtime |
|--------|----------|---------|
| `bayesian_averaging` | Best overall performance | Fast |
| `optimized_weights` | Customizable, interpretable | Medium |
| `neural_metalearner` | Complex non-linear combinations | Slow |
| `rank_fusion` | Ranking-focused tasks | Fast |

---

## 2. Iterative Learning with Hard Pairs

### Use Hard Pairs Extraction
```python
from quantum_layer.iterative_learning import IterativeLearningFramework

# Initialize framework
framework = IterativeLearningFramework(
    quantum_model=qsvc_model,
    classical_embedder=embedder,
    num_iterations=5
)

# Run iterative refinement with hard pairs extraction
results = framework.iterative_refinement(
    X_train, y_train, X_val, y_val,
    train_df=train_df  # IMPORTANT: Pass DataFrame with source/target columns
)

# Access refinement history
for entry in results['refinement_history']:
    print(f"Iter {entry['iteration']}: "
          f"{entry['num_hard_pairs_extracted']} hard pairs, "
          f"PR-AUC: {entry['val_pr_auc']:.4f}")
```

---

## 3. VQC Optimization

### Run Optimizer Comparison
```bash
# Quick test (100 iterations)
python experiments/vqc_optimization_analysis.py \
  --experiment optimizers \
  --max_iter 100 \
  --relation CtD \
  --max_entities 100 \
  --qml_dim 5 \
  --embedding_dim 32

# Full analysis (200 iterations)
python experiments/vqc_optimization_analysis.py \
  --experiment all \
  --max_iter 200 \
  --relation CtD
```

### Recommended VQC Configuration
```python
from qiskit_algorithms.optimizers import NFT
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

# Best optimizer: NFT
optimizer = NFT(maxiter=200)

# Feature map
feature_map = ZZFeatureMap(
    feature_dimension=5,
    reps=2,
    entanglement='linear'
)

# Ansatz
ansatz = RealAmplitudes(
    num_qubits=5,
    reps=3
)
```

**Note:** QSVC still outperforms VQC. Use VQC for research/experimentation only.

---

## 4. Performance Tuning Experiments

### 4.1 Test Graph Features in QML
```bash
# Baseline (no graph features)
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 \
  --run_ensemble --ensemble_method stacking \
  --fast_mode

# With graph features
python scripts/run_optimized_pipeline.py --relation CtD \
  # ... same args as above ...
  --use_graph_features_in_qml
```

**Expected Impact:** +0.01 to +0.03 PR-AUC

### 4.2 Extended Optuna Search
```bash
# Ensemble optimization (50 trials, ~8-12 hours)
python scripts/optuna_pipeline_search.py \
  --n_trials 50 \
  --objective ensemble

# QSVC optimization (30 trials, ~5-7 hours)
python scripts/optuna_pipeline_search.py \
  --n_trials 30 \
  --objective qsvc
```

### 4.3 Reduce Information Loss
```bash
# Higher PCA dimension (32 instead of 24)
python scripts/run_optimized_pipeline.py --relation CtD \
  # ... standard args ...
  --qml_pre_pca_dim 32

# Combine with multi-model fusion (uses full 128-D classical features)
# See Section 1 for fusion usage
```

**Expected Impact:** +0.02 to +0.05 PR-AUC

### 4.4 Scale Up Data (Remove Entity Limit)
```bash
# Full-scale run (no --max_entities limit)
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 \
  --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 32
  # Note: No --max_entities = use all entities
```

**Expected Impact:** +0.03 to +0.05 PR-AUC  
**Warning:** 2-3x longer training time, consider GPU acceleration

---

## 5. Comprehensive Experiment Plan

### Phase 1: Quick Wins (1-2 hours)
```bash
# Test 1: Higher PCA dimension
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 \
  --run_ensemble --ensemble_method stacking \
  --qml_pre_pca_dim 32 --fast_mode

# Test 2: Graph features
python scripts/run_optimized_pipeline.py --relation CtD \
  # ... same args ...
  --use_graph_features_in_qml

# Test 3: Multi-model fusion
python scripts/implementations/implement_sophisticated_ensembles.py --relation CtD --fast_mode
```

### Phase 2: Extended Optimization (Overnight)
```bash
# Optuna hyperparameter search
python scripts/optuna_pipeline_search.py --n_trials 50 --objective ensemble
```

### Phase 3: Full-Scale Training (Weekend)
```bash
# Remove all limits
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 \
  --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 32
```

---

## 6. File Reference

### New Files Created
- `quantum_layer/multi_model_fusion.py` - Multi-model fusion implementation
- `scripts/test_multi_model_fusion.py` - Fusion demonstration script
- `IMPLEMENTATION_STATUS_REPORT.md` - Detailed implementation report
- `COMMAND_REFERENCE_NEXT_TASKS.md` - This document

### Modified Files
- `quantum_layer/iterative_learning.py` - Hard pairs extraction
- `kg_layer/kg_embedder.py` - Embedding dimension validation
- `experiments/vqc_optimization_analysis.py` - Bug fixes
- `docs/OPTIMIZATION_QUICKSTART.md` - Fusion documentation

### Results Directory
- `results/vqc_analysis/` - VQC experiment results
  - `optimizer_comparison.json` - Optimizer comparison results
  - `loss_curves_optimizers.png` - Training loss curves
  - `all_results.json` - Complete experiment data

---

## 7. Troubleshooting

### Issue: Embedding dimension mismatch
```
ValueError: X has 32 features, but PCA is expecting 64 features
```
**Solution:** The embedder now auto-detects and retrains. Delete old embeddings:
```bash
rm data/entity_embeddings.npy data/entity_ids.json
```

### Issue: VQC training too slow
**Solution:** Reduce iterations or use QSVC instead:
```bash
# Faster VQC
python experiments/vqc_optimization_analysis.py --max_iter 50

# Or use QSVC (recommended)
python scripts/run_optimized_pipeline.py --relation CtD \
  # ... args ...
  --qsvc_C 0.1
```

### Issue: Out of memory with full dataset
**Solution:** Use GPU or increase batch size:
```bash
# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0

# Or use smaller embedding dimension
python scripts/run_optimized_pipeline.py --relation CtD \
  --embedding_dim 64  # Instead of 128
```

---

## 8. Expected Performance Improvements

| Enhancement | Expected PR-AUC Gain | Runtime Impact |
|-------------|---------------------|----------------|
| Multi-model fusion | +0.01 to +0.03 | Minimal |
| Higher PCA dim (32) | +0.01 to +0.02 | Minimal |
| Graph features | +0.01 to +0.03 | +10% |
| Optuna HPO (50 trials) | +0.02 to +0.05 | 8-12 hours |
| Full-scale data | +0.03 to +0.05 | 2-3x longer |
| **Combined (estimated)** | **+0.08 to +0.18** | **Varies** |

**Current Best:** 0.7987  
**Target with improvements:** 0.85 to 0.95+

---

## 9. Contact & Support

- **Implementation details:** See `IMPLEMENTATION_STATUS_REPORT.md`
- **Fusion usage:** Run `python scripts/test_multi_model_fusion.py`
- **VQC results:** Check `results/vqc_analysis/`
- **General questions:** Review `docs/OPTIMIZATION_QUICKSTART.md`

---

**Last Updated:** March 6, 2026  
**Best Result:** PR-AUC 0.7987 (Ensemble-QC-stacking, Pauli)
