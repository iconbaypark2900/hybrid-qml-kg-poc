# Next Steps to Improve Performance

This document outlines a prioritized roadmap for improving PR-AUC and model performance in the hybrid QML-KG link prediction system. Recommendations are based on root-cause analysis from the codebase and existing documentation.

---

## Latest Results (2026-02-16)

The best run combined full-graph RotatE embeddings (128D, 200 epochs), hard negative sampling, 16 qubits, 3 feature map reps, QSVC regularization C=0.1, kernel-target alignment, stacking ensemble, and GridSearchCV classical tuning.

| Model | Test PR-AUC | Status |
|-------|-------------|--------|
| Ensemble-QC-stacking (Pauli) | **0.7987** | **Best overall** (with `--qml_feature_map Pauli --qml_feature_map_reps 2`) |
| RandomForest-Optimized | **0.7838** | Best classical |
| ExtraTrees-Optimized | **0.7807** | Strong classical |
| Ensemble-QC-stacking | **0.7408** | Stacking with ZZ feature map |
| QSVC-Optimized | **0.7216** | Best quantum (gap to classical: -0.06) |

**Recommended command (reproduces best ensemble result -- 0.7987):**
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --optimize_feature_map_reps --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24 --fast_mode
```

**Recommended command (reproduces best classical result -- 0.7838):**
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map_reps 3 --qsvc_C 0.1 --optimize_feature_map_reps \
  --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24 --fast_mode
```

### Experiment Log

| Variant | RF | ET | QSVC | Ensemble | Notes |
|---------|-----|-----|------|----------|-------|
| Base (200 ep, stacking, tune_classical, pre_pca 24) | **0.7838** | **0.7807** | **0.7216** | 0.7408 | Best classical |
| + Pauli feature map (reps=2) | 0.7838 | 0.7807 | 0.6343 | **0.7987** | **Best ensemble** |
| + diverse negatives (dw=0.5) | 0.7144 | 0.7298 | 0.6689 | 0.6919 | Lower; diverse hurts here |
| + qsvc_C=0.05 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Same as C=0.1 |
| + ensemble_quantum_weight=0.4 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | No effect (stacking learns) |
| + qml_reduction_method kpca | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Same (cached embeddings) |
| + qml_dim 20, qsvc_C 0.05, pre_pca 32 | -- | -- | -- | -- | Killed: K_mm=51 min, K_nm ~5hrs (20 qubits too slow on CPU) |

**VQC optimizer experiment results (SPSA is now the default):**

| Optimizer | Test PR-AUC | Train PR-AUC |
|-----------|-------------|-------------|
| SPSA | **0.5456** | 0.6048 |
| COBYLA | 0.5086 | 0.6525 |
| NFT | 0.4782 | 0.5248 |

**VQC ansatz comparison (SPSA optimizer, 50 iterations, 8 qubits):**

| Ansatz | Test PR-AUC | Train PR-AUC | Time (s) |
|--------|-------------|-------------|----------|
| RealAmplitudes reps=4 | **0.5474** | 0.5750 | 222 |
| RealAmplitudes reps=3 | 0.5342 | 0.5691 | 195 |
| EfficientSU2 reps=3 | 0.5173 | 0.6014 | 234 |
| RealAmplitudes reps=2 | 0.5109 | 0.6051 | 189 |
| EfficientSU2 reps=2 | 0.5077 | 0.5468 | 207 |
| TwoLocal reps=2 | 0.5035 | 0.5585 | 216 |
| TwoLocal reps=3 | 0.4678 | 0.5469 | 236 |

---

## Previous Baseline (Pre-Optimization)

| Model | Test PR-AUC | Status |
|-------|-------------|--------|
| Regularized_QuantumReady_LR | 0.5667 | Best validated |
| Classical Logistic Regression | ~0.60 | Strong but overfits |
| QSVC | ~0.65 | Generalizes well |
| VQC | ~0.49 | Needs re-tuning |

**Target**: PR-AUC > 0.70 -- **ACHIEVED**

---

## 1. High Impact – Data & Embeddings

### 1.1 Full-Graph Embeddings

Train embeddings on all Hetionet relations (not just CtD) for richer entity context.

**Command:**
```bash
--full_graph_embeddings
```

**Expected improvement**: +5–15% relative (per `docs/improvements/IMPROVEMENTS_SUMMARY.md`)

**Example:**
```bash
python scripts/run_optimized_pipeline.py --relation CtD --full_graph_embeddings --use_cached_embeddings --fast_mode
```

---

### 1.2 Improve Embedding Diversity

Low head/tail diversity (e.g., 5–21% unique embeddings) limits information content. Entities sharing identical embeddings reduce the quantum kernel’s ability to discriminate.

**Actions:**
- Increase embedding dimension: `--embedding_dim 64` or `128`
- Try different methods: `--embedding_method RotatE` or `ComplEx`
- Train longer (more epochs) when fitting KG embeddings

**Relevant script:** `experiments/embedding_diversity_report.py`

---

### 1.3 Hard Negative Sampling

Hard negatives improve discrimination and generalization. The pipeline and experiment scripts already support this.

**Command:**
```bash
--negative_sampling hard
```

**Relevant script:** `scripts/hard_negatives_experiment.py`

---

## 2. High Impact – Quantum Feature Space

### 2.1 Increase Qubits / QML Dimension

More qubits → more expressivity for the quantum kernel. Current 8–12 qubits may be too low for good class separation.

**Command:**
```bash
--qml_dim 16   # or 20
```

**Note:** Watch for overfitting; consider stronger regularization if train/test gap grows.

---

### 2.2 Try Different Feature Maps

Current default: ZZFeatureMap with few reps. Experiment with:

- More ZZ reps (2 → 3–4)
- PauliFeatureMap
- Data re-uploading (encode data multiple times in circuit)

**Available options in pipeline:** `--qml_feature_map ZZ|Z`, entanglement: linear/full/circular

**Relevant file:** `quantum_layer/quantum_kernel_engineering.py`

---

### 2.3 Kernel-Target Alignment

Kernel separability (within-class vs between-class) is a known bottleneck. Target: separability ratio > 1.1.

**Actions:**
- Use kernel alignment in `quantum_layer/quantum_kernel_engineering.py`
- Optimize feature maps and embeddings for better kernel separability

---

## 3. Medium Impact – Classical and Hybrid

### 3.1 Add Graph Features to Quantum Inputs

Classical models use ~2313 features (graph + domain), quantum only ~12 from PCA. Add graph-derived features into the quantum input space:

- Node degree (in/out)
- Common neighbors
- Shortest path length
- Jaccard coefficient, Adamic-Adar, preferential attachment

**Relevant files:** `kg_layer/enhanced_features.py`, `kg_layer/advanced_embeddings.py`

---

### 3.2 Reduce Information Loss in Dimensionality Reduction

Current pipeline discards ~95% of information: 256D → 128D → 24D → 12D.

**Actions:**
- Keep more PCA components (e.g., 16–24)
- Try Kernel PCA for non-linear reduction
- Use mutual information for feature selection instead of PCA alone

---

### 3.3 Quantum-Classical Ensemble

Combine quantum and classical predictions for more robust performance.

**Relevant file:** `quantum_layer/quantum_classical_ensemble.py`

**Ideas:**
- Stack quantum and classical models
- Soft voting with tuned weights
- Quantum pre-filter → classical refinement

---

## 4. Medium Impact – VQC Optimization

VQC currently performs near random. Use the existing VQC analysis script to tune optimizers and ansatzes.

**Command:**
```bash
python experiments/vqc_optimization_analysis.py --experiment optimizers --max_iter 100
```

**Experiments covered:**
- Optimizer comparison: COBYLA, SPSA, NFT, ADAM
- Ansatz search: RealAmplitudes, EfficientSU2, TwoLocal
- Hyperparameter grid search
- Loss tracking during training

**Relevant file:** `experiments/vqc_optimization_analysis.py`

---

## 5. Lower Priority

### 5.1 Scale Up Data

Larger entity/sample sets may improve generalization. Reduce or remove `--max_entities` limits if compute allows.

---

### 5.2 Hyperparameter Optimization

Use Bayesian optimization (e.g., Optuna) for:

- Embedding dimension
- `qml_dim`
- Feature map reps
- Regularization strength

---

## Quick Start – Best Known Configuration

The command that produced the best results (0.77 classical, 0.72 quantum):

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
    --full_graph_embeddings \
    --embedding_method RotatE \
    --embedding_dim 128 \
    --embedding_epochs 150 \
    --negative_sampling hard \
    --qml_dim 16 \
    --qml_feature_map_reps 3 \
    --qsvc_C 0.1 \
    --optimize_feature_map_reps \
    --run_ensemble --ensemble_method weighted_average \
    --fast_mode
```

**Note:** Drop `--use_cached_embeddings` if embeddings need to be retrained. Add `--tune_classical` for GridSearchCV hyperparameter tuning. Add `--use_kernel_alignment` for kernel alignment diagnostics.

---

## New Pipeline Features

| Feature | Flag | Description |
|---------|------|-------------|
| Quantum-classical ensemble | `--run_ensemble --ensemble_method weighted_average\|stacking` | Combines quantum + classical predictions |
| QSVC regularization | `--qsvc_C 0.1` | Reduces quantum overfitting (default 1.0) |
| Kernel-target alignment | `--optimize_feature_map_reps` | Auto-selects feature map reps by alignment |
| Kernel alignment diagnostics | `--use_kernel_alignment` | Runs alignment analysis before QSVC |
| Classical hyperparameter tuning | `--tune_classical` | GridSearchCV over ET/RF/LR hyperparameters |
| Embedding diversity check | Automatic | Warns if embedding diversity is below 30% |
| VQC default optimizer | Changed to SPSA | SPSA generalizes better than COBYLA |
| Graph features in QML path | `--use_graph_features_in_qml` | Appends degree/neighbor features to quantum input before reduction |
| VQC ansatz CLI | `--vqc_ansatz_type`, `--vqc_ansatz_reps`, `--vqc_optimizer` | Tune VQC architecture from experiment results |
| Optuna HPO | `scripts/optuna_pipeline_search.py` | Bayesian hyperparameter search over full pipeline |

---

## Prioritized Action Summary

| Priority | Action | Command / File |
|----------|--------|----------------|
| 1 | Enable full-graph embeddings | `--full_graph_embeddings` |
| 2 | Use RotatE 128D embeddings | `--embedding_dim 128 --embedding_method RotatE --embedding_epochs 200` |
| 3 | Use hard negative sampling | `--negative_sampling hard` |
| 4 | Increase quantum dimension | `--qml_dim 16` |
| 5 | Regularize QSVC | `--qsvc_C 0.1` |
| 6 | More feature map reps + alignment | `--qml_feature_map_reps 3 --optimize_feature_map_reps` |
| 7 | Use stacking ensemble | `--run_ensemble --ensemble_method stacking` |
| 8 | Tune classical hyperparameters | `--tune_classical` |
| 9 | Pauli feature map for best ensemble | `--qml_feature_map Pauli --qml_feature_map_reps 2` |
| 10 | Pre-PCA dimension | `--qml_pre_pca_dim 24` |
| 11 | Graph features in QML path | `--use_graph_features_in_qml` |
| 12 | VQC ansatz tuning | `--vqc_ansatz_type EfficientSU2 --vqc_ansatz_reps 3` |
| 13 | Optuna HPO | `python scripts/optuna_pipeline_search.py --n_trials 30` |

---

## Optuna Hyperparameter Search

Run automated hyperparameter optimization over the full pipeline:

```bash
python scripts/optuna_pipeline_search.py --n_trials 30 --objective ensemble
python scripts/optuna_pipeline_search.py --n_trials 20 --objective qsvc
python scripts/optuna_pipeline_search.py --n_trials 20 --objective classical
```

Results are saved to `results/optuna/optuna_trials.csv` and `results/optuna/optuna_best.json`.

---

## References

- `docs/WHY_QUANTUM_UNDERPERFORMS.md` -- Root cause analysis
- `docs/OPTIMIZATION_PLAN.md` -- Full optimization roadmap
- `experiments/IMPLEMENTATION_SUMMARY.md` -- Experiment setup and next steps
- `docs/reports/RESULTS_ANALYSIS_V2.md` -- Kernel separability bottleneck
- `docs/improvements/IMPROVEMENTS_SUMMARY.md` -- Completed and planned improvements
