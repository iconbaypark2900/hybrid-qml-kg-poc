# Hybrid Quantum-Classical Link Prediction on a Biomedical Knowledge Graph

**Technical Report**

Quantum Global Group  
Hybrid QML-KG Proof of Concept  
2026

---

## Abstract

We present a hybrid quantum-classical machine learning system for link prediction on the Hetionet biomedical knowledge graph. The task is to predict *Compound-treats-Disease* (CtD) relationships. The pipeline combines full-graph knowledge graph embeddings (RotatE), pair-wise feature construction, classical models (logistic regression, random forest, extra trees) with hyperparameter tuning, and quantum kernel methods (QSVC with Pauli and ZZ feature maps) in a stacking ensemble. We report a best test PR-AUC of **0.7987** using a stacking ensemble that integrates a quantum support vector classifier (QSVC) with classically tuned random forest and extra trees. The quantum component alone reaches 0.7216 PR-AUC; the ensemble gain over the best classical model (0.7838) is +1.5 percentage points, demonstrating that a carefully configured quantum kernel can contribute to an overall improvement when combined with classical baselines. We describe the architecture, experimental protocol, key design choices (full-graph embeddings, hard negative sampling, Pauli feature map, pre-PCA dimensionality, stacking), and discuss why the variational quantum classifier (VQC) underperforms and how GPU simulation and IBM Quantum hardware are supported for future scaling.

---

## 1. Introduction

### 1.1 Motivation

Knowledge graph (KG) link prediction is central to drug repurposing and discovery: identifying missing *Compound-treats-Disease* links can suggest new therapeutic indications. Classical methods based on graph embeddings and classifiers are well established; quantum machine learning (QML) offers alternative feature maps and kernels that might capture different structure in the data. A practical question is whether a **hybrid** system—combining classical and quantum models—can outperform strong classical baselines on a real-world biomedical KG.

### 1.2 Objectives

- Build an end-to-end pipeline for CtD link prediction on Hetionet using full-graph embeddings and pair-wise features.
- Integrate quantum kernel methods (QSVC) and, optionally, variational circuits (VQC) with classical models.
- Combine quantum and classical predictions via stacking and compare against classical-only and quantum-only baselines.
- Identify configurations that meet or exceed a target PR-AUC of 0.70 and document reproducibility.

### 1.3 Contributions

- A reproducible hybrid pipeline with configurable embeddings (RotatE, ComplEx, DistMult), feature maps (ZZ, Pauli), dimensionality reduction (PCA, optional KPCA), and ensemble strategies (stacking, weighted average).
- Experimental evidence that the Pauli feature map substantially improves ensemble performance over the ZZ feature map (0.7987 vs 0.7408 PR-AUC) when used in a stacking ensemble.
- Root-cause analysis of the quantum–classical performance gap (embedding diversity, information loss, kernel expressivity) and mitigation via full-graph embeddings, hard negatives, and pre-PCA dimension.
- GPU-accelerated quantum simulation and IBM Quantum hardware support for future scaling.

---

## 2. Background and Related Work

### 2.1 Knowledge Graph Link Prediction

Link prediction in KGs is typically framed as scoring triples (head, relation, tail). For a single relation such as CtD, it reduces to binary classification over (compound, disease) pairs: positive edges are known treatments; negatives are sampled or generated. Metrics such as area under the precision-recall curve (PR-AUC) are appropriate for imbalanced settings.

### 2.2 Knowledge Graph Embeddings

TransE, RotatE, ComplEx, and DistMult embed entities and relations into a continuous space so that valid triples have higher scores. **Full-graph** training (all relations in the KG) yields richer entity representations than training only on the target relation; we use PyKEEN for RotatE/ComplEx/DistMult with configurable dimension and epochs.

### 2.3 Quantum Kernel Methods

Quantum kernels compute similarity in a quantum feature space: \(k(x,y) = |\langle \phi(x)|\phi(y)\rangle|^2\), where \(\phi\) is a parameterized encoding circuit (feature map). QSVC uses such a kernel in a support vector classifier. Expressivity depends on the feature map (e.g., ZZ, Pauli) and number of qubits/repetitions. Variational quantum classifiers (VQC) train a parameterized circuit on top of the encoding; they are more flexible but harder to train and in our experiments underperform QSVC.

### 2.4 Hybrid Ensembles

Stacking meta-learners combine base model predictions to improve generalization. We use a stacking ensemble over classical (logistic regression, random forest, extra trees) and quantum (QSVC) base models, with optional GridSearchCV tuning for the classical components.

---

## 3. Methods

### 3.1 Data and Task

- **Knowledge graph:** Hetionet (het.io), a heterogeneous biomedical KG with multiple relation types (e.g., CtD, DaG, GiG).
- **Target relation:** Compound-treats-Disease (CtD).
- **Splits:** Positive CtD edges are split into train and test (default 80/20). Negatives are sampled per split at a configurable ratio (default 1:1 with positives) using **hard negative sampling** (negatives that are “close” in the graph to make the task non-trivial).
- **Evaluation:** Test PR-AUC (and optionally ROC-AUC, F1, precision, recall). When enabled, 5-fold stratified cross-validation is used for more robust estimates.

### 3.2 Pipeline Overview

1. **Load Hetionet** and extract CtD edges; optionally use full graph for context.
2. **Train full-graph embeddings** (RotatE, ComplEx, or DistMult) on all relations with PyKEEN; default 128 dimensions, 200 epochs.
3. **Build train/test pair sets** with positives and hard negatives; derive (source, target) pairs.
4. **Feature construction:** For each (compound, disease) pair, build a vector from entity embeddings:
   - Concatenation of head and tail embeddings.
   - Difference and Hadamard product (optional) for link-specific features.
   - Optional graph-topology features (degree, common neighbors, etc.) from the training graph only (no leakage).
5. **Classical path:** Train logistic regression, random forest, and extra trees on the full feature matrix; optionally tune with GridSearchCV.
6. **Quantum path:** Reduce dimensionality (e.g., PCA from 24 to 16 qubits), encode into a quantum feature map (ZZ or Pauli), train QSVC (or VQC) with a fidelity kernel.
7. **Ensemble:** Stack base model predictions (e.g., logistic regression meta-learner) or use a fixed weighted average.
8. **Evaluate** on the test set and report PR-AUC and other metrics.

### 3.3 Embeddings

- **Model:** RotatE (default), ComplEx, or DistMult via PyKEEN.
- **Scope:** Full-graph training (`--full_graph_embeddings`) so that compound and disease entities benefit from all relation types.
- **Dimension:** 128 (default); 64 or 256 can be used.
- **Epochs:** 200 (default).
- **Negatives:** Hard negative sampling for embedding training when supported.

### 3.4 Pair Feature Construction

- **Input:** Head and tail entity embeddings (e.g., 128D each).
- **Operations:** Concat, diff, Hadamard (configurable).
- **Optional:** Graph features (degree, common neighbors, Jaccard, etc.) from the training graph, appended to the link vector before splitting into classical and quantum inputs.
- **Classical branch:** Uses the full feature vector (after variance filtering); no extra reduction.
- **Quantum branch:** Pre-PCA to 24 dimensions (configurable), then projection to 16 qubits (or 8/12/20); optional Kernel PCA or LDA.

### 3.5 Classical Models

- Logistic regression (L2), random forest, extra trees.
- Optional GridSearchCV over key hyperparameters (`--tune_classical`).
- Calibration (e.g., isotonic) can be applied for better probability estimates.

### 3.6 Quantum Models

- **QSVC:** Fidelity quantum kernel with ZZ or Pauli feature map; regularization C (default 0.1). Kernel-target alignment can be computed for diagnostics.
- **Feature map:** Pauli feature map with 2 repetitions gave the best ensemble result; ZZ with 2–3 reps is the default alternative.
- **VQC:** RealAmplitudes, EfficientSU2, or TwoLocal ansatz; SPSA optimizer (default); used for ablations but not in the best ensemble.
- **Backends:** Statevector simulator (default), GPU simulator (cuStateVec when available), noisy simulator, IBM Quantum (e.g., Heron) via configuration.

### 3.7 Ensemble

- **Stacking:** Base models (e.g., RF, ET, QSVC) produce predictions; a meta-learner (e.g., logistic regression) is trained on these predictions. No manual weight needed; the meta-learner learns the combination.
- **Weighted average:** Optional fixed weights (e.g., `ensemble_quantum_weight=0.4`); in our experiments stacking outperformed and made manual weights redundant.

---

## 4. Experimental Setup

### 4.1 Default and Best-Run Configuration

- **Embeddings:** Full-graph RotatE, 128D, 200 epochs.
- **Negatives:** Hard negative sampling, 1:1 ratio with positives.
- **Quantum:** 16 qubits, Pauli feature map, 2 reps, QSVC C=0.1, pre-PCA 24D.
- **Classical:** GridSearchCV tuning for RF, ET, LR.
- **Ensemble:** Stacking.
- **Evaluation:** Single train/test split (0.2 test size) or 5-fold CV; reported metrics are test PR-AUC unless stated otherwise.

### 4.2 Reproducibility

- Fixed random seeds (e.g., 42) for splits and training.
- Reproduce best ensemble (0.7987) with:

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --optimize_feature_map_reps --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24 --fast_mode
```

### 4.3 Hardware and Software

- Python 3.9+, PyKEEN, Qiskit, scikit-learn, PyTorch (for embeddings).
- Quantum: statevector simulator (default); optional GPU (cuStateVec) and IBM Quantum.
- 8–16 GB RAM recommended for 16-qubit runs.

---

## 5. Results

### 5.1 Main Results

| Model | Test PR-AUC | Type |
|-------|-------------|------|
| Ensemble-QC-stacking (Pauli) | **0.7987** | Hybrid |
| RandomForest-Optimized | 0.7838 | Classical |
| ExtraTrees-Optimized | 0.7807 | Classical |
| Ensemble-QC-stacking (ZZ) | 0.7408 | Hybrid |
| QSVC-Optimized | 0.7216 | Quantum |

Target PR-AUC > 0.70: **achieved**. The best result is the stacking ensemble with the Pauli feature map; the quantum-only QSVC is 0.7216, and the best classical (RF) is 0.7838. The ensemble improves over the best classical model by about 1.5 percentage points.

### 5.2 Ablations and Variants

| Variant | RF | ET | QSVC | Ensemble | Notes |
|---------|-----|-----|------|----------|------|
| Base (ZZ, stacking, tune, pre-PCA 24) | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Best classical |
| + Pauli feature map (reps=2) | 0.7838 | 0.7807 | 0.6343 | **0.7987** | Best ensemble |
| + Diverse negatives (dw=0.5) | 0.7144 | 0.7298 | 0.6689 | 0.6919 | Worse than hard |
| + qsvc_C=0.05 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Same as C=0.1 |
| + ensemble_quantum_weight=0.4 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | No gain (stacking learns) |

- **Pauli vs ZZ:** Switching to Pauli (reps=2) lowers standalone QSVC (0.6343 vs 0.7216) but **raises** ensemble PR-AUC (0.7987 vs 0.7408), indicating that the Pauli kernel provides complementary information to the classical models.
- **Diverse negatives:** Using diverse negative sampling (e.g., degree-weighted) reduced performance relative to hard negatives in this setup.
- **VQC:** Best VQC (RealAmplitudes, reps=4, SPSA) reached ~0.5474 test PR-AUC; SPSA outperformed COBYLA and NFT. VQC is not used in the reported best ensemble.

### 5.3 VQC Optimizer and Ansatz (Ablation)

| Optimizer | Test PR-AUC |
|-----------|-------------|
| SPSA | **0.5456** |
| COBYLA | 0.5086 |
| NFT | 0.4782 |

| Ansatz (8 qubits, 50 iter) | Test PR-AUC |
|----------------------------|-------------|
| RealAmplitudes reps=4 | **0.5474** |
| RealAmplitudes reps=3 | 0.5342 |
| EfficientSU2 reps=3 | 0.5173 |

---

## 6. Discussion

### 6.1 Why the Ensemble Beats Classical-Only

The stacking meta-learner combines RF, ET, and QSVC. Even when QSVC alone is below RF (e.g., 0.6343 with Pauli), its predictions are sufficiently **uncorrelated** with the classical ones that the meta-learner can exploit them, yielding a net gain (0.7987 vs 0.7838). The choice of feature map (Pauli) changes the quantum kernel’s bias in a way that complements the trees.

### 6.2 Why Quantum-Only Lags Classical

Earlier analysis (see `docs/WHY_QUANTUM_UNDERPERFORMS.md`) highlighted:

- **Embedding diversity:** Low head/tail uniqueness in the training set reduces effective information for the quantum kernel; full-graph embeddings and higher dimension help.
- **Information loss:** Aggressive reduction (e.g., 256D → 24D → 16D) discards much of the signal; we mitigate with pre-PCA 24 and optional graph features in the QML path.
- **Overfitting:** Quantum kernels can overfit small data; regularization (e.g., QSVC C=0.1) and ensemble diversity help.
- **Kernel expressivity:** ZZ vs Pauli and number of reps affect separability; kernel-target alignment is used for diagnostics and feature-map tuning.

### 6.3 Limitations

- Single relation (CtD) and single KG (Hetionet); generalization to other relations and KGs is untested.
- Quantum runs are simulation-based for reproducibility; hardware runs would be needed for scaling and noise sensitivity.
- VQC is not competitive with QSVC in the current setup; further work on ansatz and optimizer would be required to make it useful in the ensemble.

### 6.4 GPU and Hardware Readiness

The pipeline supports GPU-accelerated quantum simulation (e.g., cuStateVec) and IBM Quantum (Heron) via configuration files and flags (`--gpu`, `quantum_config_path`). All paths fall back to CPU when GPU or hardware is unavailable. See `docs/overview/IMPLEMENTATION_RECAP.md` for details.

---

## 7. Conclusion

We implemented and evaluated a hybrid quantum-classical pipeline for Compound-treats-Disease link prediction on Hetionet. The best configuration achieves **0.7987 test PR-AUC** using full-graph RotatE embeddings, hard negative sampling, a 16-qubit Pauli QSVC, and a stacking ensemble with tuned classical models. The quantum component adds value in the ensemble even when its standalone performance is below the best classical model. Key factors are: full-graph embeddings, hard negatives, Pauli feature map, pre-PCA dimension, and stacking rather than manual weighting. We release the pipeline, configuration, and documentation to support reproduction and extension to other relations and backends.

---

## 8. References and Resources

### Code and Data

- Repository: [hybrid-qml-kg-poc](https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc) (Quantum Global Group).
- Hetionet: [het.io](https://het.io/).
- Dashboard: [Hugging Face Space – QGG-HYBRID-PROJECT](https://huggingface.co/spaces/rocRevyAreGoals15/QGG-HYBRID-PROJECT).

### Documentation (in-repo)

- `README.md` — Quick start, architecture, reproduce command.
- `docs/planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md` — Experiment log, recommended commands, optimization roadmap.
- `docs/overview/IMPLEMENTATION_RECAP.md` — Pipeline flags, GPU/hardware setup, Optuna usage.
- `docs/WHY_QUANTUM_UNDERPERFORMS.md` — Root-cause analysis of quantum–classical gap.

### Software

- PyKEEN: knowledge graph embedding training.
- Qiskit: quantum circuits, feature maps, QSVC, Aer/cuStateVec, IBM Quantum.
- scikit-learn: classical models, stacking, GridSearchCV, metrics.

---

*Document version: 2026-02. Corresponds to branch `roc/featuremap-dashboard-combined` and best-run configuration described in `docs/planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md`.*
