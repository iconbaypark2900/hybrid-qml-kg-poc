# Quantum Kernel Link Prediction on Biomedical Knowledge Graphs: Bridging Graph Embeddings and Quantum Feature Spaces for Drug Repurposing

**Technical Report**

Quantum Global Group  
2026

---

## Abstract

We present the first system to combine knowledge graph (KG) embeddings with quantum kernel classifiers for biomedical link prediction — a hybrid quantum-classical pipeline for predicting *Compound-treats-Disease* (CtD) relationships on the Hetionet knowledge graph. While prior work has applied quantum kernels to molecular fingerprints and classical KG methods to drug repurposing independently, no existing approach feeds learned KG embedding vectors into quantum feature spaces for link prediction. Our pipeline trains full-graph RotatE embeddings (128D, 200 epochs) across all 24 Hetionet relation types, constructs pair-wise features enriched with graph topology, pharmacological domain, and novel mechanism-of-action (MoA) signals, then classifies candidate links using both classical ensembles and a quantum support vector classifier (QSVC) with Pauli feature maps in a 16-qubit circuit. A stacking ensemble achieves a best test PR-AUC of **0.7987**, a +1.5 percentage point gain over the strongest classical baseline (RandomForest, 0.7838). We validate predictions against ClinicalTrials.gov, finding that 33% of top novel predictions are confirmed by existing clinical trials and one (Ezetimibe for gout) represents a biologically plausible novel hypothesis. The MoA feature module — which encodes binding target overlap, pharmacologic class membership, and chemical/disease similarity to known treatments — is designed to recalibrate scores by penalizing structurally plausible but mechanistically implausible predictions. We release the full pipeline, configuration, and documentation to support reproduction and extension to additional relation types and quantum hardware backends.

---

## 1. Introduction

### 1.1 Motivation

Knowledge graph (KG) link prediction is central to drug repurposing and discovery: identifying missing *Compound-treats-Disease* links can suggest new therapeutic indications for existing drugs. Classical methods based on graph embeddings and ensemble classifiers are well established and achieve strong baselines. Quantum machine learning (QML) offers alternative feature maps and kernels that may capture different structure in the data — particularly in the high-dimensional, sparse feature spaces that arise from KG embeddings.

However, the two fields have developed in parallel without convergence. Classical KG-based drug repurposing uses TransE, RotatE, ComplEx, or GNN-based methods to score candidate links (Mayers et al., 2023; Himmelstein et al., 2017). Quantum drug discovery focuses on molecular-level tasks — QSAR, protein-ligand binding affinity, and virtual screening using quantum kernels on molecular fingerprints (QKDTI, Sci. Reports 2025; Kruger et al., 2023). **No existing work feeds learned KG embedding vectors into quantum feature spaces for link prediction.** This paper bridges that gap.

### 1.2 Objectives

- Build an end-to-end pipeline for CtD link prediction on Hetionet using full-graph embeddings and pair-wise features.
- Integrate quantum kernel methods (QSVC) and, optionally, variational circuits (VQC) with classical models.
- Introduce mechanism-of-action (MoA) features derived from multi-relational Hetionet structure to improve prediction plausibility.
- Combine quantum and classical predictions via stacking and compare against classical-only and quantum-only baselines.
- Validate top predictions against ClinicalTrials.gov to assess clinical relevance.

### 1.3 Contributions

1. **Novel intersection.** To our knowledge, this is the first system to combine KG embedding vectors (RotatE) with quantum kernel classifiers (QSVC) for biomedical link prediction, bridging two previously disjoint research tracks.
2. **Mechanism-of-action feature module.** We introduce 10 pharmacological plausibility features derived from binding targets (CbG), disease-gene associations (DaG), pathway overlap (GpPW), pharmacologic class membership (PCiC), and chemical/disease similarity (CrC, DrD) to recalibrate predictions and reduce mechanistically implausible false positives.
3. **Pauli feature map ensemble effect.** We show that the Pauli feature map substantially improves stacking ensemble performance over the ZZ feature map (0.7987 vs 0.7408 PR-AUC) even though standalone QSVC performance is lower (0.6343 vs 0.7216), because the Pauli kernel provides predictions sufficiently uncorrelated with classical models for the meta-learner to exploit. With 256D embeddings, the Pauli ensemble reaches 0.8581.
4. **Clinical validation.** We validate predictions against ClinicalTrials.gov, finding 33% confirmed by existing trials and identifying one novel, biologically plausible drug repurposing hypothesis.
5. **Reproducible pipeline** with configurable embeddings (RotatE, ComplEx, DistMult), feature maps (ZZ, Pauli), dimensionality reduction, ensemble strategies, GPU-accelerated simulation, and IBM Quantum hardware support.

---

## 2. Background and Related Work

### 2.1 Knowledge Graph Link Prediction for Drug Repurposing

Link prediction in KGs is typically framed as scoring triples (head, relation, tail). For a single relation such as CtD, it reduces to binary classification over (compound, disease) pairs: positive edges are known treatments; negatives are sampled or generated. Metrics such as area under the precision-recall curve (PR-AUC) are appropriate for imbalanced settings.

Recent work has advanced KG-based drug repurposing substantially. Mayers et al. (bioRxiv, 2023; updated 2024) benchmarked seven link prediction methods including TransE, ComplEx, and RotatE on a biomedical KG, achieving MRR of 0.9792 via ensemble. A December 2025 preprint combined KG embeddings (TransE on DRKG, which includes Hetionet data) with LLM-based GraphRAG for explainable repurposing. Giampaolo et al. (bioRxiv, 2024) built the PATHOS KG from 24 databases for Alzheimer's drug repurposing. These are exclusively classical methods.

### 2.2 Knowledge Graph Embeddings

TransE, RotatE, ComplEx, and DistMult embed entities and relations into a continuous space so that valid triples have higher scores. **Full-graph** training (all relations in the KG) yields richer entity representations than training only on the target relation; we use PyKEEN for RotatE/ComplEx/DistMult with configurable dimension and epochs. RotatE models relations as rotations in complex space, which is well-suited to the heterogeneous relation types in Hetionet.

### 2.3 Quantum Kernel Methods in Drug Discovery

Quantum kernels compute similarity in a quantum feature space: $k(x,y) = |\langle \phi(x)|\phi(y)\rangle|^2$, where $\phi$ is a parameterized encoding circuit (feature map). QSVC uses such a kernel in a support vector classifier. Expressivity depends on the feature map (e.g., ZZ, Pauli) and number of qubits/repetitions.

Recent quantum drug discovery work has focused on molecular-level tasks rather than KG-level tasks. QKDTI (Scientific Reports, 2025) applied Quantum Support Vector Regression with quantum feature mapping to drug-target interaction prediction using molecular features. Kruger et al. (Machine Learning: Science and Technology, 2023) demonstrated QSVC for ligand-based virtual screening on molecular fingerprints. A February 2026 preprint used a 2000-node Coherent Ising Machine for allosteric site detection and molecular docking — quantum hardware for drug discovery, but not for KG reasoning. A January 2024 preprint described a hybrid classical-quantum pipeline for real-world drug discovery, again operating at the molecular rather than graph level.

**The gap our work fills:** All prior quantum drug discovery applies quantum kernels or circuits to *molecular-level* features (fingerprints, QSAR descriptors, binding affinities). All prior KG-based drug repurposing uses *classical* methods exclusively. Our work is the first to bridge these tracks by feeding learned KG embedding vectors into quantum feature spaces for link prediction.

### 2.4 Hybrid Ensembles

Stacking meta-learners combine base model predictions to improve generalization. We use a stacking ensemble over classical (logistic regression, random forest, extra trees) and quantum (QSVC) base models, with optional GridSearchCV tuning for the classical components. The intuition is that even when the quantum model's standalone performance is below classical baselines, its predictions may be sufficiently *uncorrelated* with classical model errors to provide complementary signal.

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

### 3.5 Mechanism-of-Action (MoA) Features

A key limitation of embedding-based link prediction is that high scores may reflect *structural proximity* in the graph without corresponding *mechanistic plausibility*. For example, a compound and disease may share graph neighborhoods through comorbidity patterns (e.g., HIV-cancer comorbidity inflating Abacavir-ocular cancer scores) without any pharmacological basis for treatment.

To address this, we introduce 10 mechanism-of-action features per compound-disease pair, extracted from the multi-relational Hetionet structure:

| Feature | Source Relation(s) | Signal |
|---------|-------------------|--------|
| `moa_binding_targets` | CbG | Compound's gene binding breadth |
| `moa_disease_genes` | DaG | Disease's genetic association breadth |
| `moa_shared_targets` | CbG ∩ DaG | **Direct mechanistic overlap** — compound binds disease-linked genes |
| `moa_target_overlap` | Jaccard(CbG, DaG) | Normalized mechanistic overlap |
| `moa_shared_pathway_genes` | CbG ∩ DaG via GpPW | Shared targets in same biological pathway |
| `moa_pharmacologic_classes` | PCiC | Drug class membership count |
| `moa_compound_similarity` | CrC | Chemical neighborhood size |
| `moa_similar_compounds_treat` | CrC ∩ CtD(train) | **Analogical evidence** — do chemically similar compounds treat anything? |
| `moa_disease_similarity` | DrD | Disease neighborhood size |
| `moa_similar_diseases_treated` | DrD ∩ CtD(train) | **Analogical evidence** — are similar diseases treatable? |

The MoA index is built from the full 2.25M-edge Hetionet graph but uses only *training* CtD edges for known treatment lookup (features 8 and 10), preventing data leakage. Features 3–5 capture *direct mechanistic evidence* (does the compound bind genes implicated in the disease?), while features 8 and 10 capture *analogical evidence* (do structurally similar compounds/diseases participate in known treatments?).

These features are activated with the `--use_moa_features` flag and are appended to the classical feature vector. They flow through to the quantum path when `--use_graph_features_in_qml` is enabled.

### 3.6 Classical Models

- Logistic regression (L2), random forest, extra trees.
- Optional GridSearchCV over key hyperparameters (`--tune_classical`).
- Calibration (e.g., isotonic) can be applied for better probability estimates.

### 3.7 Quantum Models

- **QSVC:** Fidelity quantum kernel with ZZ or Pauli feature map; regularization C (default 0.1). Kernel-target alignment can be computed for diagnostics.
- **Feature map:** Pauli feature map with 2 repetitions gave the best ensemble result; ZZ with 2–3 reps is the default alternative.
- **VQC:** RealAmplitudes, EfficientSU2, or TwoLocal ansatz; SPSA optimizer (default); used for ablations but not in the best ensemble.
- **Backends:** Statevector simulator (default), GPU simulator (cuStateVec when available), noisy simulator, IBM Quantum (e.g., Heron) via configuration.

### 3.8 Ensemble

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

**Table 1.** Primary benchmark — RotatE-128D, 200 epochs, full-graph, Pauli 16-qubit (reps=2), C=0.1, hard negatives, stacking ensemble. Source: `results/optimized_results_20260216-100431.json`. QSVC kernel computation time: 2,619 s (genuine full-dataset computation).

| Model | Test PR-AUC | Type |
|-------|-------------|------|
| Ensemble-QC-stacking (Pauli) | **0.7987** | Hybrid |
| RandomForest-Optimized | 0.7838 | Classical |
| ExtraTrees-Optimized | 0.7807 | Classical |
| Ensemble-QC-stacking (ZZ) | 0.7408 | Hybrid |
| QSVC-Optimized (ZZ)† | 0.7216 | Quantum |
| QSVC-Optimized (Pauli) | 0.6343 | Quantum |

†QSVC 0.7216 is from the ZZ feature map configuration; the Pauli QSVC scores 0.6343 but its complementary predictions drive the ensemble from 0.7838 to 0.7987.

Target PR-AUC > 0.70: **achieved**. The best result is the stacking ensemble with the Pauli feature map; the best classical baseline (RF) is 0.7838. The ensemble improves over the best classical model by +1.5 percentage points.

**Table 2.** Extended results — RotatE-256D, 250 epochs, full-graph, Pauli 12-qubit (reps=1), C=0.676 (Optuna-tuned), hard negatives, stacking. Source: `results/optimized_results_20260323-134844.json`. Note: quantum kernel was pre-computed once (~99 s) and reused across Optuna trials (Optuna swept classical hyperparameters with fixed cached quantum kernel).

| Model | Test PR-AUC | Type |
|-------|-------------|------|
| Ensemble-QC-stacking (Pauli-256D) | **0.8581** | Hybrid |
| RandomForest-Optimized | 0.8569 | Classical |
| ExtraTrees-Optimized | 0.8498 | Classical |
| QSVC-Optimized (Pauli-256D) | 0.7222 | Quantum |

The 256D RotatE embeddings with Optuna-tuned classical components yield a substantially higher ceiling (0.8581 vs 0.7987), confirming that embedding quality is the dominant performance driver. The QSVC contribution (0.7222 standalone) and its ensemble synergy hold under the larger embedding configuration.

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

### 6.3 Clinical Validation of Predictions

To assess whether the model produces clinically meaningful predictions, we validated the top 6 novel compound-disease predictions against ClinicalTrials.gov:

| Prediction | Score | Clinical Trials | Verdict |
|---|---|---|---|
| Abacavir → Ocular Cancer | 0.793 | 0 | No support — graph artifact |
| Ezetimibe → Gout | 0.693 | 0 direct, 4 anti-inflammatory | **Novel plausible hypothesis** |
| Ramipril → Stomach Cancer | 0.597 | 0 | No support |
| Losartan → Atherosclerosis | 0.528 | 7+ trials (Phase 4) | **Strongly validated** |
| Mitomycin → Liver Cancer | 0.525 | 7 trials (TACE) | **Strongly validated** |
| Salmeterol → Liver Cancer | 0.520 | 0 | No support — noise |

**Key finding:** The model's highest-scoring prediction (Abacavir → ocular cancer, 0.793) has zero clinical evidence, while well-validated predictions (Losartan → atherosclerosis, Mitomycin → liver cancer) score lower (~0.52–0.53). This inversion — where structurally plausible but mechanistically implausible pairs outscore genuinely valid pairs — motivates the MoA feature module (Section 3.5).

The **Ezetimibe → gout** prediction (0.693) is particularly interesting. While no trial directly targets this indication, four trials investigate ezetimibe's anti-inflammatory properties, and emerging literature links lipid metabolism to urate levels. This represents a genuine novel hypothesis that could be worth further investigation.

### 6.4 Limitations and Future Work

- **Single relation (CtD):** The pipeline is designed for multi-relational expansion; CpD (Compound-palliates-Disease, 390 edges) and DrD (Disease-resembles-Disease, 543 edges) are natural next targets with nearly identical data characteristics.
- **Simulation-based quantum:** All results use statevector simulation; IBM Quantum hardware runs would assess noise robustness.
- **VQC underperformance:** VQC (best: 0.5474 with RealAmplitudes reps=4) remains near random; QSVC is the effective quantum model. Future work on ansatz design and optimizer may improve VQC.
- **Score recalibration:** The MoA features are designed to address the score-validity inversion identified in Section 6.3; empirical evaluation with MoA features enabled is in progress.
- **Larger KGs:** Extending to DRKG (4.4M edges, 97K entities) or custom multi-database KGs would test scalability.

### 6.5 GPU and Hardware Readiness

The pipeline supports GPU-accelerated quantum simulation (e.g., cuStateVec) and IBM Quantum (Heron) via configuration files and flags (`--gpu`, `quantum_config_path`). All paths fall back to CPU when GPU or hardware is unavailable. See `docs/overview/IMPLEMENTATION_RECAP.md` for details.

---

## 7. Conclusion

We present the first hybrid quantum-classical pipeline that bridges knowledge graph embeddings with quantum kernel classifiers for biomedical link prediction. The system predicts Compound-treats-Disease relationships on Hetionet, achieving **0.7987 test PR-AUC** (primary result, 16-qubit Pauli, genuine quantum computation) and **0.8581 PR-AUC** (extended result, 12-qubit Pauli with 256D embeddings and Optuna-tuned classical models). Both use a stacking ensemble combining RotatE embeddings, classical tree-based models, and a quantum support vector classifier (QSVC). Even when QSVC standalone performance (0.6343 Pauli, 0.7216 ZZ) is below the best classical model (0.7838), the quantum kernel provides predictions sufficiently uncorrelated with classical model errors that the meta-learner achieves a net ensemble gain.

Clinical validation against ClinicalTrials.gov confirms 33% of top novel predictions and identifies one novel drug repurposing hypothesis (Ezetimibe for gout). The newly introduced mechanism-of-action feature module encodes binding target overlap, pharmacologic class, and treatment analogy from the multi-relational KG structure to recalibrate predictions and penalize false positives.

This work establishes a new intersection between two previously disjoint research tracks — quantum kernel methods (applied to molecular features) and KG-based drug repurposing (applied with classical methods). We release the full pipeline, configuration, and documentation to support reproduction, extension to additional Hetionet relation types, and deployment on quantum hardware.

---

## 8. References

### Primary Citations

1. Himmelstein, D.S. et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife*, 6, e26726. [Hetionet]
2. Mayers, M. et al. (2023; updated 2024). Drug repurposing using consilience of knowledge graph completion methods. *bioRxiv*, 10.1101/2023.05.12.540594. [KG link prediction benchmarks including RotatE]
3. QKDTI (2025). Quantum kernel-based drug-target interaction prediction. *Scientific Reports*, 10.1038/s41598-025-07303-z. [Quantum kernels for DTI]
4. Kruger, D.M. et al. (2023). Quantum machine learning framework for virtual screening. *Machine Learning: Science and Technology*, 10.1088/2632-2153/acb900. [QSVC for drug screening]
5. Deep Learning-Based Drug Repurposing Using KG Embeddings and GraphRAG (2025). *bioRxiv*, 10.64898/2025.12.08.693009. [KG + LLM drug repurposing]
6. Large-Scale Quantum Computing Framework Enhances Drug Discovery (2026). *bioRxiv*, 10.64898/2026.02.09.704961. [Quantum hardware for drug discovery]
7. Hybrid Classical-Quantum Pipeline for Real World Drug Discovery (2024). *bioRxiv*, 10.1101/2024.01.08.574600. [Hybrid quantum drug discovery]

### Code and Data

- Repository: [hybrid-qml-kg-poc](https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc) (Quantum Global Group).
- Hetionet: [het.io](https://het.io/).
- Dashboard: [Hugging Face Space – QGG-HYBRID-PROJECT](https://huggingface.co/spaces/rocRevyAreGoals15/QGG-HYBRID-PROJECT).

### Software

- PyKEEN: knowledge graph embedding training.
- Qiskit: quantum circuits, feature maps, QSVC, Aer/cuStateVec, IBM Quantum.
- scikit-learn: classical models, stacking, GridSearchCV, metrics.

---

*Document version: 2026-04. Incorporates MoA feature module, clinical trial validation, and literature positioning.*
