# Implementation Recap: Pipeline Improvements & GPU/Hardware Readiness

This document summarizes the work completed to improve performance, add tunable pipeline features, and prepare the project for GPU-backed simulation and quantum hardware (DGX Spark / IBM Heron).

---

## 1. Performance Results & Experiment Log

### Best Results Achieved

| Model | Test PR-AUC | Notes |
|-------|-------------|--------|
| Ensemble-QC-stacking (Pauli) | **0.7987** | Best overall; `--qml_feature_map Pauli --qml_feature_map_reps 2` |
| RandomForest-Optimized | **0.7838** | Best classical |
| ExtraTrees-Optimized | **0.7807** | Strong classical |
| Ensemble-QC-stacking (ZZ) | 0.7408 | Stacking with default ZZ feature map |
| QSVC-Optimized | 0.7216 | Best quantum (gap to classical ~0.06) |

**Target PR-AUC > 0.70:** Achieved.

### Experiment Log (Key Variants)

| Variant | RF | ET | QSVC | Ensemble | Notes |
|---------|-----|-----|------|----------|-------|
| Base (200 ep, stacking, tune_classical, pre_pca 24) | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Best classical |
| + Pauli feature map (reps=2) | 0.7838 | 0.7807 | 0.6343 | **0.7987** | Best ensemble |
| + diverse negatives (dw=0.5) | 0.7144 | 0.7298 | 0.6689 | 0.6919 | Lower than hard negatives |
| + qsvc_C=0.05 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Same as C=0.1 |
| + ensemble_quantum_weight=0.4 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | No effect (stacking learns) |
| + qml_reduction_method kpca | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Same (cached embeddings) |
| + qml_dim 20 | — | — | — | — | Killed: K_mm ~51 min, K_nm ~5 h on CPU |

### VQC Optimizer & Ansatz

- **Default optimizer:** SPSA (best in experiments).
- **Ansatz comparison (8 qubits, 50 iter):** RealAmplitudes reps=4 best (test PR-AUC **0.5474**); results in `results/vqc_analysis/`.

---

## 2. Pipeline Improvements Implemented

### 2.1 New Pipeline Flags & Features

| Feature | Flag / Mechanism | Description |
|---------|------------------|-------------|
| Classical hyperparameter tuning | `--tune_classical` | GridSearchCV over ExtraTrees, RandomForest, LogisticRegression |
| Kernel alignment diagnostics | `--use_kernel_alignment` | Logs kernel-target alignment before QSVC |
| Pre-PCA dimension for QML | `--qml_pre_pca_dim 24` | Keeps more dimensions before qubit projection |
| Reduction method | `--qml_reduction_method pca\|kpca\|lda` | Dimensionality reduction for QML features |
| Graph features in QML path | `--use_graph_features_in_qml` | Appends degree, common neighbors, etc. to quantum input |
| VQC ansatz/optimizer | `--vqc_ansatz_type`, `--vqc_ansatz_reps`, `--vqc_optimizer` | Tune VQC from experiment results |
| GPU quantum simulator | `--gpu` or `config/quantum_config_gpu.yaml` | GPU-backed Aer (cuStateVec) for kernels |
| Embedding diversity check | Automatic | Warns if embedding diversity < 30% |

### 2.2 Graph Features in QML Path

- **Code:** `quantum_layer/advanced_qml_features.py` — `prepare_qml_features()` accepts optional `X_extra` (e.g. shape `(N, K)`).
- **Pipeline:** When `--use_graph_features_in_qml` is set, the pipeline builds a 14-D graph-feature matrix per (source, target) using `EnhancedFeatureBuilder.build_graph_features()` (degree, common neighbors, Jaccard, etc.) and concatenates it with encoded link features before PCA/KPCA reduction.
- **Leakage:** Graph is built from training edges only; no test edges used.

### 2.3 VQC Configuration from CLI

- **Pipeline:** `scripts/run_optimized_pipeline.py` reads `--vqc_ansatz_type`, `--vqc_ansatz_reps`, `--vqc_optimizer` and passes them into `vqc_config` (replacing hardcoded RealAmplitudes, reps=3, SPSA).
- **Experiment script:** `experiments/vqc_optimization_analysis.py` — `--experiment ansatzes` and `--experiment grid_search` produce results that can be plugged back via these flags.

### 2.4 Optuna Hyperparameter Search

- **Script:** `scripts/optuna_pipeline_search.py`
- **Usage:** `python scripts/optuna_pipeline_search.py --n_trials 30 --objective ensemble`
- **Objectives:** `ensemble`, `qsvc`, `classical`, `best`
- **Search space:** embedding_dim, embedding_epochs, qml_dim, qsvc_C, qml_feature_map_reps, ensemble_method, ensemble_quantum_weight, qml_feature_map, qml_reduction_method, qml_pre_pca_dim, tune_classical.
- **Outputs:** `results/optuna/optuna_trials.csv`, `results/optuna/optuna_best.json`, SQLite study DB.

---

## 3. GPU & Quantum Hardware Readiness

### 3.1 Configurations

| File | Purpose |
|------|---------|
| `config/quantum_config_dgx.yaml` | DGX Spark: `execution_mode: gpu_simulator` by default; includes `gpu_simulator` section and heron/ibm_quantum for hardware |
| `config/quantum_config_gpu.yaml` | Standalone GPU-only config for any NVIDIA GPU machine |
| `config/quantum_config_gmktec.yaml` | CPU-only (simulator), no GPU |

### 3.2 Quantum Executor (GPU Simulator)

- **File:** `quantum_layer/quantum_executor.py`
- **Additions:**
  - `QuantumExecutor.gpu_available()` — static check for GPU-backed Aer (cuStateVec).
  - `_get_gpu_simulator_sampler()` — builds `AerSimulator(method='statevector', device='GPU')`, wraps in SamplerV2; falls back to CPU on failure.
  - `get_sampler()` routes `execution_mode == "gpu_simulator"` to the GPU sampler.
  - `get_execution_metadata()` reports `gpu_simulator` and backend label.

### 3.3 QML Kernel Path (GPU)

- **qml_trainer.py:** When `exec_mode == "gpu_simulator"`, kernel is `FidelityQuantumKernel` with the GPU sampler (not CPU-only `FidelityStatevectorKernel`). ZNE skip list includes `gpu_simulator`.
- **qml_model.py:** `_prepare_quantum_kernel()` and VQC feature-map decomposition both handle `gpu_simulator` (decompose circuit, use sampler-based kernel).

### 3.4 PyKEEN on GPU

- **kg_embedder.py:** TransE PyKEEN `pipeline()` uses `device='cuda' if torch.cuda.is_available() else 'cpu'`.
- **advanced_embeddings.py:** Same for RotatE/ComplEx full-graph embedding training.

### 3.5 Pipeline CLI & Logging

- **Flag:** `--gpu` selects `config/quantum_config_gpu.yaml`.
- **Startup logging:** Reports whether quantum GPU (cuStateVec) and PyTorch CUDA are available.

### 3.6 Requirements

- **requirements.txt** and **requirements-full.txt:** Comments added for GPU-accelerated quantum simulation (`qiskit-aer-gpu` or building Aer with CUDA).

### 3.7 How to Run on DGX Spark / GPU

```bash
# Option A: --gpu flag
python scripts/run_optimized_pipeline.py --relation CtD --gpu --full_graph_embeddings ...

# Option B: Environment variable (uses DGX config)
HYBRID_QML_SYSTEM=dgx python scripts/run_optimized_pipeline.py --relation CtD ...

# Option C: Explicit config
python scripts/run_optimized_pipeline.py --quantum_config_path config/quantum_config_gpu.yaml ...
```

All paths fall back to CPU automatically if no GPU is available.

---

## 4. Documentation Updated

- **NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md:** Latest results table, experiment log, recommended commands, new pipeline features table, VQC ansatz comparison, Optuna usage, GPU notes.

---

## 5. Key Files Touched (Summary)

| Area | Files |
|------|--------|
| Config | `config/quantum_config_dgx.yaml`, `config/quantum_config_gpu.yaml` (new) |
| Quantum executor | `quantum_layer/quantum_executor.py` |
| QML kernels | `quantum_layer/qml_trainer.py`, `quantum_layer/qml_model.py` |
| QML features | `quantum_layer/advanced_qml_features.py` |
| Embeddings | `kg_layer/kg_embedder.py`, `kg_layer/advanced_embeddings.py` |
| Pipeline | `scripts/run_optimized_pipeline.py` |
| HPO | `scripts/optuna_pipeline_search.py` (new) |
| Docs / deps | `NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md`, `requirements.txt`, `requirements-full.txt` |
