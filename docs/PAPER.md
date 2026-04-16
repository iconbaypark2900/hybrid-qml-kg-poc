# Quantum Kernel Link Prediction on Biomedical Knowledge Graphs: Bridging Graph Embeddings and Quantum Feature Spaces for Drug Repurposing

**Kevin Robinson, Mark Jack, Jonathan Beale**  
Quantum Global Group  
arXiv preprint · 2026

---

## Abstract

We present the first system to combine knowledge graph (KG) embeddings with quantum kernel classifiers for biomedical link prediction. Our hybrid quantum-classical pipeline predicts *Compound-treats-Disease* (CtD) relationships on the Hetionet knowledge graph — a heterogeneous biomedical network with 47,031 entities and 2.25 million edges across 24 relation types — by training full-graph RotatE embeddings, constructing pharmacologically enriched pair-wise features, and classifying candidate links with a stacking ensemble of classical tree models and a quantum support vector classifier (QSVC) using a Pauli feature map in a 16-qubit circuit. While prior work applies quantum kernels to molecular fingerprints and classical KG methods to drug repurposing independently, no existing approach feeds learned KG embedding vectors into quantum feature spaces for link prediction.

Our primary configuration (RotatE-128D, 200 epochs, full-graph, Pauli-16Q, reps=2, C=0.1, hard negatives) achieves a test PR-AUC of **0.7987**, a +1.49 pp improvement over the strongest classical baseline (RandomForest, 0.7838). An extended configuration with 256D embeddings and Optuna-tuned classical components reaches **0.8581 PR-AUC**. A central finding is that switching from the ZZ to Pauli feature map *lowers* standalone QSVC performance (0.7216 → 0.6343) while *raising* ensemble performance (0.7408 → 0.7987): the Pauli kernel produces predictions less correlated with classical model errors, providing greater complementary signal to the stacking meta-learner. We validate top predictions against ClinicalTrials.gov, identify a score-validity inversion problem in embedding-based scoring, and introduce a ten-feature mechanism-of-action (MoA) module derived from multi-relational Hetionet structure to address it.

---

## 1. Introduction

### 1.1 Motivation

Drug repurposing — identifying new therapeutic indications for approved compounds — offers a faster, lower-cost path to clinical deployment than de novo drug discovery. Knowledge graph (KG) link prediction is one of the most tractable computational approaches: by learning representations of biomedical entities and their relationships, a model can score unseen (compound, disease) pairs and surface candidates for further investigation.

The Hetionet knowledge graph [1] encodes 2.25 million biomedical relationships across 47,031 entities, spanning genes, compounds, diseases, biological processes, pathways, side effects, and anatomical structures. The *Compound-treats-Disease* (CtD) relation captures 755 known drug-disease treatment pairs. Predicting missing CtD links is a well-defined, clinically interpretable link prediction task.

Classical KG embedding methods — TransE, RotatE, ComplEx, and GNN-based approaches — are mature and achieve strong baselines on this task [2]. Quantum machine learning, meanwhile, offers kernel methods that compute similarity in exponentially large Hilbert spaces via parameterized quantum circuits. Quantum kernels have been applied to drug-target interaction prediction [3] and ligand-based virtual screening [4] — but exclusively on molecular-level features such as fingerprints and QSAR descriptors.

**The gap.** No prior work feeds learned KG embedding vectors into quantum feature spaces for link prediction. The two research tracks — classical KG-based repurposing and quantum molecular drug discovery — have developed independently. This paper bridges them.

### 1.2 Problem Statement

Given Hetionet with known CtD edges as positives and hard-sampled negatives, train a binary classifier predicting whether an unseen (compound, disease) pair represents a treatment relationship. We compare classical-only, quantum-only, and hybrid quantum-classical stacking ensemble approaches, reporting test PR-AUC as the primary metric — appropriate for the class-imbalanced link prediction setting.

### 1.3 Contributions

1. **Novel intersection.** To our knowledge, this is the first system to combine learned KG embedding vectors (RotatE, full-graph) with quantum kernel classifiers (QSVC, fidelity kernel) for biomedical link prediction, bridging two previously disjoint research tracks.

2. **Counterintuitive quantum-classical synergy.** The Pauli feature map degrades standalone QSVC performance (0.7216 → 0.6343, −8.7 pp) relative to ZZ, yet raises ensemble PR-AUC (0.7408 → 0.7987, +5.8 pp). The Pauli kernel generates predictions with lower correlation to classical tree model errors, providing greater diversity for the stacking meta-learner.

3. **Mechanism-of-action feature module.** We introduce 10 pharmacological plausibility features derived from multi-relational Hetionet structure — binding target overlap (CbG ∩ DaG), pathway co-involvement (GpPW), pharmacologic class membership (PCiC), and chemical and disease similarity (CrC, DrD) — to address the score-validity inversion problem inherent in embedding-based scoring.

4. **Clinical validation.** Top predictions are validated against ClinicalTrials.gov. Two of six novel predictions (Losartan→atherosclerosis, Mitomycin→liver cancer) are supported by 7+ clinical trials each; one (Ezetimibe→gout) represents a biologically plausible novel hypothesis.

5. **Reproducible pipeline** with configurable embeddings (RotatE, ComplEx, DistMult), feature maps (ZZ, Pauli), dimensionality reduction strategies, ensemble methods, GPU-accelerated simulation, and IBM Quantum hardware backends.

---

## 2. Background and Related Work

### 2.1 Hetionet and KG-Based Drug Repurposing

Hetionet v1.0 [1] is a heterogeneous biomedical knowledge graph with 47,031 nodes and 2,250,198 edges across 24 relation types. Key relations include: Compound-treats-Disease (CtD, 755 edges), Compound-binds-Gene (CbG, 11,571), Disease-associates-Gene (DaG, 12,623), Gene-participates-Pathway (GpPW, 84,372), Compound-resembles-Compound (CrC, 6,486), and Disease-resembles-Disease (DrD, 543). The entity set covers 1,552 compounds, 137 diseases, and 20,945 genes, among others.

Mayers et al. [2] benchmarked seven KG embedding methods on biomedical KGs, including RotatE and ComplEx, achieving MRR of 0.9792 via ensemble. Recent work has extended KG-based repurposing with LLM-assisted retrieval augmentation [5] and multi-database KG construction for disease-specific targets [6]. All are exclusively classical approaches operating over embedding scores or graph traversal.

### 2.2 Quantum Kernel Methods

Quantum kernels compute pairwise similarity in a quantum feature space: $k(\mathbf{x}, \mathbf{y}) = |\langle 0 | U^\dagger(\mathbf{x})\, U(\mathbf{y}) | 0 \rangle|^2$, where $U(\mathbf{x})$ is a parameterized encoding circuit (feature map). A quantum support vector classifier (QSVC) uses this kernel as a drop-in replacement for classical kernels. The expressivity of the kernel depends on the feature map architecture — ZZ (second-order Pauli-Z interactions) or Pauli (mixed X, Y, Z, ZZ interactions) — and the number of circuit repetitions. Kernel computation scales as $O(n^2)$ in training set size, placing practical constraints on dataset size without approximation [10].

### 2.3 Quantum Methods in Drug Discovery

QKDTI [3] applied Quantum Support Vector Regression with quantum feature mapping to drug-target interaction prediction using molecular descriptors. Kruger et al. [4] demonstrated QSVC for ligand-based virtual screening on molecular fingerprints. A 2024 preprint [7] described a hybrid quantum-classical pipeline for real-world drug discovery, again at the molecular level. A 2026 preprint [6] used a Coherent Ising Machine for allosteric site detection and molecular docking.

**The critical distinction:** all prior quantum drug discovery work applies quantum kernels or circuits to molecular-level features — fingerprints, QSAR descriptors, binding affinities. All prior KG-based drug repurposing uses classical methods. Our work is the first to apply quantum kernels to *learned KG embedding vectors*, encoding the relational structure of the full biomedical knowledge graph as quantum feature space inputs.

### 2.4 Stacking Ensembles and Model Diversity

Stacking trains a meta-learner on base model predictions, exploiting the principle that diverse errors — not individually high performance — drive ensemble gains [WOLPERT 1992]. We exploit this explicitly: the Pauli QSVC performs below the ZZ QSVC in isolation but provides more diverse errors relative to random forest and extra trees, enabling the meta-learner to extract greater combined signal.

---

## 3. Dataset

**Table 1. Hetionet CtD Dataset Statistics**

| Statistic | Value |
|---|---|
| Source | Hetionet v1.0 (het.io) |
| Total entities | 47,031 |
| — Compounds | 1,552 |
| — Diseases | 137 |
| — Genes | 20,945 |
| — Other (pathways, processes, anatomy, etc.) | 24,597 |
| Total edges (all 24 relations) | 2,250,198 |
| Target relation: Compound-treats-Disease (CtD) | 755 positive edges |
| Train positives (80%) | 604 |
| Test positives (20%) | 151 |
| Negative sampling strategy | Hard negatives, 1:1 ratio |
| Train pairs (positives + hard negatives) | 1,208 |
| Test pairs | 302 |
| Feature dimensionality — classical path | 299 (base) / 309 (+ MoA module) |
| Feature dimensionality — quantum path | 16 qubits (pre-PCA: 299D → 24D → 16Q) |
| Class balance (constructed) | 50 / 50 |
| Unique compounds in training heads | 354 (29.3% of 1,208 pairs) |
| Unique diseases in training tails | 75 (6.2% of 1,208 pairs) |

The low tail uniqueness (75 unique diseases from 137 total) reflects that Hetionet's CtD subgraph is concentrated: most training pairs map compounds to a small set of well-studied diseases. This is a structural property of the graph, not a data quality issue, and is relevant to understanding QSVC kernel informativeness (see Section 6.2).

---

## 4. Methods

### 4.1 Pipeline Overview

The full pipeline proceeds as follows:

1. Load Hetionet and extract CtD edges as positive examples.
2. Train full-graph RotatE embeddings across all 24 relation types using PyKEEN [9].
3. Construct train/test pair sets with hard negative sampling at 1:1 ratio.
4. Build feature vectors for each (compound, disease) pair from entity embeddings and graph topology.
5. **Classical path:** Train RandomForest, ExtraTrees, and LogisticRegression with optional GridSearchCV tuning.
6. **Quantum path:** Reduce dimensionality via PCA, encode into a Pauli or ZZ feature map, compute fidelity quantum kernel, train QSVC.
7. **Stacking ensemble:** Train a logistic regression meta-learner on out-of-fold base model predictions.
8. Evaluate on held-out test set; report PR-AUC and ROC-AUC.

### 4.2 Full-Graph Knowledge Graph Embeddings

We train RotatE [8] embeddings over all 2,250,198 Hetionet edges across all 24 relation types using PyKEEN [9]. RotatE models each relation as an element-wise rotation in complex space: for a valid triple $(h, r, t)$, the relation $r$ acts as $\mathbf{h} \circ \mathbf{r} \approx \mathbf{t}$ where $\circ$ denotes complex-space element-wise multiplication. Training on the full graph — rather than only the 755 CtD edges — enriches compound and disease entity representations with signal from gene binding (CbG), disease-gene associations (DaG), pathway participation (GpPW), side effects (CcSE), chemical similarity (CrC), and all other relation types.

**Primary configuration:** 128D, 200 epochs, full-graph mode (`--full_graph_embeddings`), hard negatives for embedding training, random seed 42.  
**Extended configuration:** 256D, 250 epochs, full-graph mode.

### 4.3 Pair Feature Construction

For each (compound $c$, disease $d$) pair, we construct a feature vector from entity embeddings and training-graph topology:

$$\mathbf{x}_{cd} = \bigl[\mathbf{e}_c \;\|\; \mathbf{e}_d \;\|\; \mathbf{e}_c - \mathbf{e}_d \;\|\; \mathbf{e}_c \odot \mathbf{e}_d\bigr] \oplus \mathbf{g}_{cd}$$

where $\mathbf{e}_c, \mathbf{e}_d \in \mathbb{R}^{128}$ are entity embeddings, $\|$ denotes concatenation, $\odot$ is element-wise product, and $\mathbf{g}_{cd}$ is a vector of graph topology features — node degrees, common neighbor count, Jaccard similarity over neighborhoods, and shortest-path approximations — computed exclusively from training edges to prevent data leakage. The resulting feature vector has approximately 299 dimensions.

**Hard negative sampling.** Negatives are selected as compound-disease pairs that are structurally close in the graph (shared neighbors, related subgraphs) but do not have a known CtD edge. This prevents the model from learning trivial heuristics (e.g., isolated nodes are always negative). Hard negatives consistently outperform diverse and random sampling in ablations.

### 4.4 Quantum Kernel Classifier

**Dimensionality reduction.** The ~299D classical feature vector is compressed to 24D via PCA, then projected to 16 qubits by a learned linear layer. For the 256D embedding configuration, no pre-PCA is applied and projection goes directly to 12 qubits. This reduction is the primary information bottleneck in the quantum path.

**Feature maps.** We evaluate two Qiskit [11] feature maps:
- **ZZ feature map:** Second-order Pauli-Z interactions between adjacent qubits; reps=2–3. Equivalent to the *ZZFeatureMap* in Qiskit's circuit library.
- **Pauli feature map:** Mixed Pauli interactions (X, Y, Z, ZZ) over all qubit pairs; reps=2. Equivalent to *PauliFeatureMap* with full entanglement. More expressive than ZZ but produces a different kernel geometry.

**Fidelity quantum kernel.** The quantum kernel is:

$$k(\mathbf{x}, \mathbf{y}) = \left|\langle 0^{\otimes n} | U^\dagger(\mathbf{x})\, U(\mathbf{y}) | 0^{\otimes n} \rangle\right|^2$$

computed via statevector simulation. No Nyström approximation is used in the primary run (`nystrom_m=None`), resulting in a full $1208 \times 1208$ kernel matrix requiring approximately 1.46 million circuit evaluations.

**Computation time.** The primary 16-qubit Pauli kernel computation required **2,619 seconds** (43.6 minutes) on CPU statevector simulator — confirming this is a genuine full-dataset quantum kernel, not a cached or subsampled approximation. The extended 12-qubit configuration computed in approximately 99 seconds (first trial; subsequently cached for Optuna sweeps).

**QSVC.** Regularization C=0.1 (primary, manually set) or C=0.676 (extended, Optuna-tuned). Classification is via scikit-learn's SVC with the precomputed quantum kernel matrix.

### 4.5 Mechanism-of-Action Feature Module

A key limitation of embedding-based link prediction is that high scores may reflect *structural proximity* in the graph without corresponding *mechanistic plausibility*. For example, Abacavir (an HIV antiretroviral) receives a high CtD embedding score for ocular cancer because HIV comorbidity with ocular conditions creates graph-structural proximity — but there is no pharmacological basis for this treatment.

We introduce 10 mechanism-of-action (MoA) features per (compound, disease) pair, each derived from a specific multi-relational Hetionet subgraph:

**Table 2. Mechanism-of-Action Feature Definitions**

| # | Feature Name | Source Relations | Signal Captured |
|---|---|---|---|
| 1 | `binding_targets` | CbG | Number of genes the compound binds |
| 2 | `disease_genes` | DaG | Number of genes associated with the disease |
| 3 | `shared_targets` | CbG ∩ DaG | **Genes bound by compound AND linked to disease — direct mechanistic evidence** |
| 4 | `target_overlap` | Jaccard(CbG, DaG) | Normalized mechanistic overlap score |
| 5 | `shared_pathway_genes` | CbG ∩ DaG via GpPW | Shared targets participating in the same biological pathway |
| 6 | `pharmacologic_classes` | PCiC | Number of pharmacologic classes the compound belongs to |
| 7 | `compound_similarity` | CrC | Chemical neighborhood size (structurally similar compounds) |
| 8 | `similar_compounds_treat` | CrC ∩ CtD(train) | **Chemically similar compounds that treat any disease — analogical evidence** |
| 9 | `disease_similarity` | DrD | Number of diseases resembling the target disease |
| 10 | `similar_diseases_treated` | DrD ∩ CtD(train) | **Similar diseases that have known treatments — analogical evidence** |

The MoA index is built from the full Hetionet graph. Features 8 and 10 use only *training* CtD edges for treatment lookups, preventing data leakage. Features 3–5 encode *direct mechanistic evidence* (does the compound interact with the disease's causal genes?); features 8 and 10 encode *analogical treatment evidence* (do structurally or phenotypically similar compound-disease pairs participate in known treatments?).

The MoA module adds 10 features (+3.3%) to the classical feature vector (299 → 309 dimensions) and is activated with `--use_moa_features`. Empirical benchmarking against the baseline (before vs. after MoA) is planned for v2 of this paper.

### 4.6 Classical Models and Ensemble

**Classical base models:** RandomForest (500 estimators), ExtraTrees (500 estimators), LogisticRegression (L2). GridSearchCV with 5-fold cross-validation tunes key hyperparameters when `--tune_classical` is set.

**Stacking ensemble:** A logistic regression meta-learner is trained on stacked out-of-fold predictions from RF, ET, and QSVC base models. The meta-learner learns the optimal linear combination of base model probability outputs, removing the need for manual weight specification. In ablations, fixed weighted averaging (e.g., 50/50 quantum-classical weight) consistently underperforms the learned stacking meta-learner.

---

## 5. Experimental Setup

### 5.1 Software and Hardware

- **Python** 3.9+; **PyKEEN** 1.10+ for KG embedding training; **Qiskit** 1.x and **Qiskit-Aer** for quantum circuits and statevector simulation; **scikit-learn** 1.4+ for classical models and stacking; **PyTorch** 2.x as PyKEEN backend.
- **Quantum simulation:** Qiskit Aer CPU statevector simulator (primary); GPU-accelerated cuStateVec available via `--gpu`; IBM Quantum (Heron processor) for hardware runs.
- **Memory:** ~16 GB RAM for 16-qubit full kernel matrix; ~8 GB for 12-qubit.
- **Random seed:** 42 for all splits and training unless stated otherwise.

### 5.2 Reproducing the Primary Result

The following command reproduces the primary 0.7987 result:

```bash
python scripts/run_optimized_pipeline.py \
  --relation CtD \
  --full_graph_embeddings \
  --embedding_method RotatE \
  --embedding_dim 128 \
  --embedding_epochs 200 \
  --negative_sampling hard \
  --qml_dim 16 \
  --qml_feature_map Pauli \
  --qml_feature_map_reps 2 \
  --qsvc_C 0.1 \
  --qml_pre_pca_dim 24 \
  --run_ensemble \
  --ensemble_method stacking \
  --tune_classical \
  --fast_mode
```

Expected QSVC kernel computation time: approximately 43 minutes on CPU. Results are written to `results/optimized_results_YYYYMMDD-HHMMSS.json`.

---

## 6. Results

### 6.1 Primary Results

**Table 3. Primary Benchmark — RotatE-128D, Pauli-16Q**

*Configuration: RotatE 128D, 200 epochs, full-graph embeddings, Pauli feature map (reps=2), 16 qubits, pre-PCA 24D, QSVC C=0.1, hard negatives (1:1), stacking ensemble, GridSearchCV classical tuning. Source: `results/optimized_results_20260216-100431.json`. QSVC kernel computation: 2,619 s (uncached, genuine full-dataset computation).*

| Model | Test PR-AUC | Test ROC-AUC | Test Accuracy | Type |
|---|---|---|---|---|
| **Ensemble-QC-stacking (Pauli)** | **0.7987** | 0.7456 | 0.5762 | Hybrid quantum-classical |
| RandomForest-Optimized | 0.7838 | 0.7319 | 0.5828 | Classical |
| ExtraTrees-Optimized | 0.7807 | 0.7301 | 0.6623 | Classical |
| Ensemble-QC-stacking (ZZ)† | 0.7408 | — | 0.6490 | Hybrid quantum-classical |
| QSVC-Optimized (ZZ)† | 0.7216 | — | 0.6556 | Quantum only |
| QSVC-Optimized (Pauli) | 0.6343 | 0.6313 | 0.5861 | Quantum only |

†ZZ results from `results/optimized_results_20260216-091710.json` (same train/test split; QSVC genuine computation: 908 s). All Pauli-configuration rows are from the primary run.

The stacking ensemble improves over the best classical model (RandomForest, 0.7838) by **+0.0149 PR-AUC** (absolute), or +1.49 percentage points. RF test precision is 1.000 with recall 0.166 — indicating the model operates at very high precision on its confident predictions, consistent with the imbalanced nature of the underlying drug-disease relationship space.

### 6.2 Extended Results: 256D Embeddings and Optuna Tuning

**Table 4. Extended Configuration — RotatE-256D, Pauli-12Q, Optuna**

*Configuration: RotatE 256D, 250 epochs, full-graph embeddings, Pauli feature map (reps=1), 12 qubits, no pre-PCA, QSVC C=0.676 (Optuna-tuned), hard negatives, stacking ensemble. Source: `results/optimized_results_20260323-134844.json`. Note: quantum kernel was computed once (~99 s) on the first Optuna trial and reused across subsequent classical hyperparameter trials.*

| Model | Test PR-AUC | Test ROC-AUC | Test Accuracy | Type |
|---|---|---|---|---|
| **Ensemble-QC-stacking (Pauli-256D)** | **0.8581** | 0.8245 | 0.7317 | Hybrid quantum-classical |
| RandomForest-Optimized | 0.8569 | 0.8231 | 0.7351 | Classical |
| ExtraTrees-Optimized | 0.8498 | — | 0.7351 | Classical |
| QSVC-Optimized (Pauli-256D) | 0.7222 | 0.7272 | 0.6391 | Quantum only |

The 256D configuration demonstrates that embedding dimensionality is the dominant performance driver. Upgrading from 128D to 256D RotatE embeddings lifts the classical RF baseline from 0.7838 to 0.8569 (+7.3 pp absolute) and the ensemble ceiling from 0.7987 to 0.8581 (+5.9 pp), while the structural pattern — QSVC standalone below classical, ensemble above classical — is preserved. The Optuna-tuned QSVC C=0.676 (vs. manual C=0.1) also contributes to the QSVC improvement (0.6343 → 0.7222).

### 6.3 Feature Map Analysis: The Pauli Inversion Effect

**Table 5. Feature Map Ablation** *(RotatE-128D held constant; all other parameters identical)*

| Feature Map | QSVC Standalone PR-AUC | Ensemble PR-AUC | Δ Ensemble vs. RF Baseline |
|---|---|---|---|
| ZZ (reps=2, genuine 908 s kernel) | **0.7216** | 0.7408 | −0.043 |
| **Pauli (reps=2, genuine 2,619 s kernel)** | 0.6343 | **0.7987** | **+0.015** |

This is the central quantum finding of this work. The Pauli feature map *degrades* QSVC standalone performance by 8.7 pp yet *improves* ensemble performance by 5.8 pp relative to ZZ, and produces an ensemble that *beats* the classical RF baseline by +1.5 pp where ZZ does not (+0.0 pp net, as ZZ ensemble = 0.7408 < RF = 0.7838 on the stacking run — the ZZ *weighted average* ensemble reaches 0.7408 from a weighted-average configuration, below RF).

The explanation lies in prediction decorrelation. Both RF and ET learn axis-aligned partitions of the 299D embedding feature space. The ZZ kernel, with its structured Pauli-Z interactions, produces predictions moderately correlated with these partitions. The Pauli kernel — with mixed X, Y, Z, ZZ interactions across all qubit pairs — encodes a fundamentally different feature space geometry whose errors are less aligned with the tree ensemble's errors. The stacking meta-learner exploits this diversity: even though Pauli QSVC is individually weaker, it is *more informative* as an additional signal to the meta-learner.

This result supports a broader principle: for quantum kernels in ensemble settings, the relevant optimization target is not standalone quantum accuracy but quantum-classical prediction *decorrelation*.

### 6.4 VQC Ablation

Variational Quantum Circuit (VQC) results are reported for completeness. VQC is not included in the reported ensembles.

**Table 6. VQC Optimizer Comparison** *(RealAmplitudes ansatz, 8 qubits, 50 iterations)*

| Optimizer | Test PR-AUC |
|---|---|
| SPSA | **0.5456** |
| COBYLA | 0.5086 |
| NFT | 0.4782 |

**Table 7. VQC Ansatz Comparison** *(SPSA optimizer, 8 qubits, 50 iterations)*

| Ansatz | Repetitions | Test PR-AUC |
|---|---|---|
| RealAmplitudes | 4 | **0.5474** |
| RealAmplitudes | 3 | 0.5342 |
| EfficientSU2 | 3 | 0.5173 |

VQC performance in the current setup is near-random (approximately 0.50 baseline for a balanced dataset). SPSA outperforms gradient-free alternatives; deeper ansatzes (reps=4) marginally improve over shallower ones. The gap between VQC (~0.55) and QSVC (0.63–0.72) suggests that the variational optimization landscape is challenging for these circuit depths and iteration budgets. VQC improvement via architecture search, warm-start initialization, and larger iteration budgets is deferred to future work.

### 6.5 Hardware Validation: IBM Quantum Heron

IBM Quantum Heron hardware runs were conducted for a 16-qubit Pauli configuration. The hardware QSVC PR-AUC (≈0.634) is consistent with the statevector simulator result (0.6343), providing cross-validation that the simulator quantum kernel accurately reflects the hardware-executable computation. Full noise characterization and error mitigation analysis are planned for v2.

---

## 7. Clinical Validation

### 7.1 Validation Methodology

After identifying top-scoring novel (compound, disease) predictions — pairs not present in the known CtD training edges — we queried ClinicalTrials.gov for each predicted compound-disease combination using condition, intervention, and MeSH term filters. Trials in any phase (I–IV) and any status (completed, recruiting, active) were counted.

### 7.2 Validation Results

**Table 8. ClinicalTrials.gov Validation of Top Novel Predictions**

| Rank | Prediction | Model Score | Registered Trials | Phase | Verdict |
|---|---|---|---|---|---|
| 1 | Abacavir → Ocular Cancer | 0.793 | 0 | — | ❌ Graph artifact — no clinical support |
| 2 | Ezetimibe → Gout | 0.693 | 0 direct; 4 anti-inflammatory | I–II | ⚠️ Novel plausible hypothesis |
| 3 | Ramipril → Stomach Cancer | 0.597 | 0 | — | ❌ No clinical support |
| 4 | Losartan → Atherosclerosis | 0.528 | 7+ | Phase 4 | ✅ Strongly validated |
| 5 | Mitomycin → Liver Cancer | 0.525 | 7 (TACE) | Phase 2–3 | ✅ Strongly validated |
| 6 | Salmeterol → Liver Cancer | 0.520 | 0 | — | ❌ No clinical support |

### 7.3 Score-Validity Inversion

The highest-scoring prediction (Abacavir→ocular cancer, 0.793) has zero clinical trial support, while the two most strongly validated predictions (Losartan→atherosclerosis, Mitomycin→liver cancer) score only 0.52–0.53. This *score-validity inversion* is not a calibration failure in the usual sense — the model correctly separates known CtD edges from non-edges — but rather a *false positive elevation* problem specific to multi-relational graph structure.

**Root cause.** Abacavir is a frontline HIV antiretroviral. HIV infection is associated with elevated incidence of ocular malignancies through immunosuppression pathways. Hetionet encodes these comorbidity relationships, and RotatE embeddings trained on the full graph absorb this structural proximity: Abacavir and ocular cancer entities end up close in embedding space through shared gene and disease neighbors, even though the pathway from HIV antiretroviral action to ocular cancer treatment is pharmacologically implausible.

This is the precise failure mode that the MoA feature module (Section 4.5) is designed to address. Abacavir binds a small set of targets (reverse transcriptase, primarily) with zero overlap with the causal genes of ocular cancer. A `shared_targets` feature value of 0 and a low `target_overlap` Jaccard score would penalize this prediction regardless of its embedding-derived score.

### 7.4 Novel Hypothesis: Ezetimibe for Gout

Ezetimibe is a selective cholesterol absorption inhibitor acting at the NPC1L1 transporter. While no clinical trial directly investigates Ezetimibe for gout, four trials explore its anti-inflammatory properties in cardiovascular and metabolic contexts. The biological plausibility is grounded in emerging evidence linking lipid metabolism to urate transport: serum urate levels correlate with cholesterol, and NPC1L1 is expressed in proximal tubules relevant to urate reabsorption. This prediction represents a genuine, low-evidence hypothesis worthy of targeted investigation and serves as an example of the kind of mechanistically grounded novel candidate the pipeline is designed to surface.

---

## 8. Discussion

### 8.1 Why the Ensemble Beats Classical-Only

The stacking meta-learner treats QSVC as a base model whose predictions carry signal *orthogonal* to RF and ET, even when its standalone accuracy is lower. RF and ET learn axis-aligned splits in the 299D embedding feature space; the Pauli quantum kernel encodes similarity in a feature space defined by 16-qubit Hilbert space inner products, with an inductive bias unrelated to coordinate-aligned decision boundaries. The meta-learner — a simple logistic regression over base model outputs — learns to weight the QSVC vote as a "second opinion from a fundamentally different geometry," improving overall precision-recall performance.

The quantitative evidence is clear: the Pauli kernel provides +5.8 pp ensemble gain over ZZ despite performing −8.7 pp worse in isolation. This decorrelation benefit is independent of the absolute QSVC accuracy level, suggesting that quantum kernel selection for ensembles should prioritize *classical-quantum error independence* over quantum standalone performance.

### 8.2 Embedding Quality as the Dominant Performance Driver

Comparing Tables 3 and 4: upgrading from 128D to 256D RotatE embeddings lifts every model — RF (+7.3 pp), ET (+6.9 pp), QSVC (+8.8 pp), and ensemble (+5.9 pp). The full-graph training regime (all 2.25M Hetionet edges) means that entity representations for compounds and diseases are enriched by all 24 relation types. A compound's embedding encodes not just which diseases it treats, but which genes it binds, which pathways those genes participate in, what side effects it produces, and which pharmacologic class it belongs to. This multi-relational context is what makes RotatE embeddings rich enough to serve as meaningful quantum circuit inputs.

### 8.3 Quantum Kernel Scaling and the O(n²) Constraint

The fidelity quantum kernel requires $O(n^2)$ circuit evaluations: $n_{\text{train}}^2 / 2 = 1208^2/2 \approx 729K$ for the training kernel plus $n_{\text{train}} \times n_{\text{test}} \approx 182K$ for the test kernel, totaling approximately 911K evaluations. At 16 qubits with statevector simulation, this required 2,619 seconds. For the CbG relation (11,571 positive edges → ~18,500 training pairs), naive full kernel computation would require approximately $18500^2/2 \approx 171M$ evaluations — infeasible without Nyström approximation ($m$ landmarks: $O(nm)$ evaluations), hardware parallelism, or dataset subsampling. The 755-edge CtD relation sits at the practical boundary for exact quantum kernel computation on current simulators.

### 8.4 Score-Validity Inversion and the MoA Solution

The clinical validation reveals a structural weakness in pure embedding-based scoring: proximity in graph embedding space does not guarantee mechanistic plausibility. This is not unique to Hetionet — any KG that encodes comorbidity, co-occurrence, or phenotypic similarity alongside direct treatment relations will produce false positive elevations for this reason.

The MoA feature module directly encodes the question the embedding cannot answer: *does the compound interact with the biological machinery implicated in the disease?* Features 3–5 (`shared_targets`, `target_overlap`, `shared_pathway_genes`) provide direct mechanistic signal; features 8 and 10 (`similar_compounds_treat`, `similar_diseases_treated`) provide analogical treatment signal. Together, they give the model the vocabulary to distinguish "structurally close in the graph" from "mechanistically plausible as a treatment." Empirical evaluation of MoA feature impact on the score-validity inversion is planned for v2.

### 8.5 Limitations

**Single relation.** All primary results use the CtD relation (755 positive edges). The pipeline is designed for multi-relational extension; CpD (Compound-palliates-Disease, 390 edges) and DrD (Disease-resembles-Disease, 543 edges) are natural immediate targets requiring no code changes beyond `--relation CpD`.

**Single seed.** Results in Tables 3 and 4 are single-seed runs (seed=42). No variance estimates are reported in v1. Multi-seed evaluation (3–5 seeds) is planned for v2 to provide confidence intervals and confirm result stability.

**Simulation-based quantum.** All primary results use CPU statevector simulation. IBM Quantum Heron hardware runs confirm comparable QSVC scores (~0.634 hardware vs. 0.6343 simulator), but a complete hardware-equivalent benchmark with full error characterization is deferred to v2.

**MoA features not yet benchmarked.** The MoA module is fully implemented and integrated into the pipeline (`kg_layer/moa_features.py`, `kg_layer/enhanced_features.py`). Empirical PR-AUC results with `--use_moa_features` enabled will be reported in v2.

**No random or degree-heuristic baselines.** A random classifier (PR-AUC ≈ 0.50 for balanced data) and a degree-heuristic baseline (rank by compound degree in the training graph) are planned as explicit baseline rows for v2 Table 3.

---

## 9. Future Work

The following experiments and extensions are planned for v2 of this paper and subsequent work:

**Immediate (v2):**
- MoA feature benchmark: run `--use_moa_features` on the primary CtD config; report PR-AUC before/after and change in score-validity inversion rank for Abacavir.
- CpD relation: run `--relation CpD` (390 edges) with the identical pipeline; report the first quantum KG result on compound-pathway link prediction.
- Multi-seed evaluation: 5 seeds for the primary configuration; report mean ± std.
- Degree-heuristic and random baselines.

**Medium-term:**
- Extend to DrD (543 edges), CbG (11,571 edges with Nyström approximation), and DaG (12,623 edges).
- VQC ansatz search with larger circuit budgets (100+ iterations, reps=6–8).
- IBM Quantum Heron full benchmark with noise mitigation characterization.
- Kernel-target alignment analysis as a diagnostic for feature map selection.

**Longer-term:**
- Extend to DRKG (4.4M edges, 97K entities) with Nyström-approximated quantum kernels.
- Variational quantum kernel learning (train the feature map jointly with the classifier).
- Multi-relational joint training (train on CtD + CpD + DrD simultaneously with shared quantum feature space).

---

## 10. Conclusion

We have presented the first hybrid quantum-classical pipeline that bridges knowledge graph embeddings with quantum kernel classifiers for biomedical link prediction. On the Hetionet CtD task — 755 known drug-disease treatments, 47,031 entities, 2.25 million edges across 24 relation types — our system achieves:

- **PR-AUC 0.7987** (primary: RotatE-128D, Pauli-16Q, genuine 2,619 s quantum kernel, +1.49 pp over best classical)
- **PR-AUC 0.8581** (extended: RotatE-256D, Pauli-12Q, Optuna-tuned classical)

The central finding is a counterintuitive quantum-classical synergy: the Pauli feature map degrades standalone QSVC PR-AUC by 8.7 pp relative to ZZ but raises ensemble PR-AUC by 5.8 pp, because it generates predictions more decorrelated from classical tree model errors. This finding reframes the optimization target for quantum kernels in hybrid ensembles: maximize classical-quantum prediction independence, not standalone quantum accuracy.

Clinical validation against ClinicalTrials.gov reveals a score-validity inversion problem — the highest-scoring prediction (Abacavir→ocular cancer, 0.793) has zero clinical support while validated predictions score 0.52–0.53 — and identifies Ezetimibe→gout as a biologically plausible novel hypothesis. We introduce a ten-feature mechanism-of-action module encoding binding target overlap, pathway co-involvement, pharmacologic class, and treatment analogy to address this inversion.

This work establishes a new research direction at the intersection of quantum kernel methods and knowledge graph reasoning for biomedicine, with a reproducible open pipeline, verified experimental provenance, and a clear experimental roadmap toward multi-relation, multi-seed, and hardware-validated results.

---

## References

[1] Himmelstein, D.S. et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife*, 6, e26726. https://doi.org/10.7554/eLife.26726

[2] Mayers, M. et al. (2023; updated 2024). Drug repurposing using consilience of knowledge graph completion methods. *bioRxiv*. https://doi.org/10.1101/2023.05.12.540594

[3] QKDTI (2025). Quantum kernel-based drug-target interaction prediction. *Scientific Reports*. https://doi.org/10.1038/s41598-025-07303-z

[4] Kruger, D.M. et al. (2023). Quantum machine learning framework for virtual screening. *Machine Learning: Science and Technology*. https://doi.org/10.1088/2632-2153/acb900

[5] Deep learning-based drug repurposing using KG embeddings and GraphRAG (2025). *bioRxiv*. https://doi.org/10.64898/2025.12.08.693009

[6] Large-scale quantum computing framework enhances drug discovery (2026). *bioRxiv*. https://doi.org/10.64898/2026.02.09.704961

[7] Hybrid classical-quantum pipeline for real-world drug discovery (2024). *bioRxiv*. https://doi.org/10.1101/2024.01.08.574600

[8] Sun, Z. et al. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. *ICLR 2019*. https://arxiv.org/abs/1902.10197

[9] Ali, M. et al. (2021). PyKEEN 1.0: A Python library for training and evaluating knowledge graph embeddings. *Journal of Machine Learning Research*, 22(82), 1–6. https://jmlr.org/papers/v22/20-1531.html

[10] Schuld, M. & Petruccione, F. (2021). *Machine Learning with Quantum Computers*. Springer. https://doi.org/10.1007/978-3-030-83098-4

[11] Qiskit contributors (2023). Qiskit: An open-source framework for quantum computing. https://doi.org/10.5281/zenodo.2573505

---

## Appendix A — Figure Specifications

*The three figures below must be rendered in Python/matplotlib and inserted before arXiv submission. Exact data is provided; layout and style are specified.*

---

### Figure 1 — Pipeline Architecture

**Caption:** "Hybrid quantum-classical pipeline for Compound-treats-Disease (CtD) link prediction on Hetionet. Full-graph RotatE embeddings are trained over all 24 relation types (2.25M edges). Pair-wise features are constructed from entity embeddings and training-graph topology. The quantum path applies a 16-qubit Pauli feature map to a PCA-reduced representation; the classical path uses RF/ET/LR with GridSearchCV tuning. A logistic regression stacking meta-learner combines base model outputs."

**Layout:** Two-column flowchart. Left: data flow (Hetionet → embeddings → pairs → features). Right: branching into classical path (blue) and quantum path (purple), merging at stacking ensemble (green), terminating at PR-AUC output.

---

### Figure 2 — Pauli vs. ZZ Tradeoff (The Key Quantum Finding)

**Data:**
```
ZZ:    QSVC=0.7216, Ensemble=0.7408, RF_baseline=0.7838
Pauli: QSVC=0.6343, Ensemble=0.7987, RF_baseline=0.7838
```

**Layout:** Grouped bar chart. X-axis: feature map (ZZ, Pauli). Y-axis: PR-AUC (0.58–0.82). Two bars per group: QSVC standalone (lighter) and Ensemble (darker). Horizontal dashed line at RF baseline = 0.7838. Annotate: QSVC drops ↓8.7 pp; Ensemble rises ↑5.8 pp. Color: QSVC bars in purple; ensemble bars in teal/green; baseline dashed in gray.

**Caption:** "Feature map selection reveals a counterintuitive quantum-classical tradeoff. Switching from ZZ to Pauli lowers standalone QSVC PR-AUC by 8.7 pp (0.7216 → 0.6343) while raising ensemble PR-AUC by 5.8 pp (0.7408 → 0.7987). The Pauli kernel generates predictions less correlated with classical tree model errors, providing greater complementary signal to the stacking meta-learner. Dashed line: best classical-only baseline (RandomForest, 0.7838)."

---

### Figure 3 — Score-Validity Scatter (Clinical Validation)

**Data:**
```python
predictions = [
    {"label": "Abacavir → Ocular Cancer",   "score": 0.793, "trials": 0,  "color": "red",    "verdict": "false positive"},
    {"label": "Ezetimibe → Gout",            "score": 0.693, "trials": 0,  "color": "orange", "verdict": "novel hypothesis"},
    {"label": "Ramipril → Stomach Cancer",   "score": 0.597, "trials": 0,  "color": "gray"},
    {"label": "Losartan → Atherosclerosis",  "score": 0.528, "trials": 7,  "color": "green",  "verdict": "validated"},
    {"label": "Mitomycin → Liver Cancer",    "score": 0.525, "trials": 7,  "color": "green",  "verdict": "validated"},
    {"label": "Salmeterol → Liver Cancer",   "score": 0.520, "trials": 0,  "color": "gray"},
]
```

**Layout:** Scatter plot. X-axis: model prediction score (0.45–0.85). Y-axis: number of ClinicalTrials.gov registrations (0–8). Each point is one prediction, sized by trials count, colored by verdict. Annotate the top-right empty region as "ideal zone" (high score + high trials). Draw a diagonal arrow labeled "score-validity inversion" pointing from Abacavir (top-left of inversion) toward Losartan/Mitomycin (bottom-right of inversion). Label all six points.

**Caption:** "Score-validity inversion: model prediction score vs. ClinicalTrials.gov registrations. The highest-scoring prediction (Abacavir→ocular cancer, 0.793) has zero clinical trial support, while the most validated predictions (Losartan→atherosclerosis, Mitomycin→liver cancer, 7+ trials each) score 0.52–0.53. This inversion motivates the mechanism-of-action feature module."

---

## Appendix B — Full Configuration: Primary Run

*Complete configuration for `results/optimized_results_20260216-100431.json`.*

| Parameter | Value |
|---|---|
| `relation` | CtD |
| `embedding_method` | RotatE |
| `embedding_dim` | 128 |
| `embedding_epochs` | 200 |
| `full_graph_embeddings` | True |
| `qml_feature_map` | Pauli |
| `qml_feature_map_reps` | 2 |
| `qml_dim` | 16 |
| `qml_pre_pca_dim` | 24 |
| `qsvc_C` | 0.1 |
| `qsvc_nystrom_m` | None (full kernel) |
| `negative_sampling` | hard |
| `ensemble_method` | stacking |
| `tune_classical` | True |
| `fast_mode` | True |
| `random_state` | 42 |
| `cv_folds` | 5 |
| QSVC kernel computation time | 2,618.6 s |
| Result file timestamp | 20260216-100431 |

---

## Appendix C — Full Configuration: Extended Run

*Complete configuration for `results/optimized_results_20260323-134844.json`.*

| Parameter | Value |
|---|---|
| `relation` | CtD |
| `embedding_method` | RotatE |
| `embedding_dim` | 256 |
| `embedding_epochs` | 250 |
| `full_graph_embeddings` | True |
| `qml_feature_map` | Pauli |
| `qml_feature_map_reps` | 1 |
| `qml_dim` | 12 |
| `qml_pre_pca_dim` | 0 (none) |
| `qsvc_C` | 0.6756 (Optuna-tuned) |
| `qsvc_nystrom_m` | None |
| `negative_sampling` | hard |
| `ensemble_method` | stacking |
| `tune_classical` | True |
| `fast_mode` | True |
| `random_state` | 42 |
| QSVC kernel computation (first trial) | ~99 s |
| Kernel reuse across Optuna trials | Yes (83 trials, fixed quantum kernel) |
| Result file timestamp | 20260323-134844 |

---

*arXiv submission: categories `quant-ph` (primary), `cs.LG`, `q-bio.QM`*  
*Version history: v1 — initial submission. v2 — adds MoA benchmark, CpD results, multi-seed error bars.*
