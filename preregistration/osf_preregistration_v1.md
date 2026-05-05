# Pre-Registration: Quantum-Kernel Methods for Biomedical Link Prediction on Hetionet, with Hardware-Validated QSVC and Multiple Baselines

**Submitted to:** Open Science Framework (OSF)
**Pre-registration type:** Retroactive standard pre-registration (computational study with disclosed pre-existing data)
**Pre-registration date:** Anticipated Q3 2026, before final analysis runs
**Anticipated manuscript submission:** Q1 2027
**Target venue:** *Quantum Machine Intelligence* (primary), *npj Quantum Information* (backup), *Bioinformatics* (tertiary)

---

## Authors

Jonathan Beale¹ *(corresponding author, lead author)*
Mark A. Jack²
Kevin Robinson³
Abdulrehman Elsayed⁴

¹ Quantum Global Group
² Florida Agricultural and Mechanical University, Department of Physics
³ EdAdvance / QuantumCT
⁴ [Affiliation to be confirmed]

ORCIDs: to be added before submission.

## Critical disclosure: retroactive pre-registration

This pre-registration is retroactive. The project was active before this document was filed. Specifically:

- **Decisions made before pre-registration:** Knowledge graph choice (Hetionet v1.0); pair-feature representation via full-graph **RotatE** node embeddings (128D, 200 epochs); pair-feature operations (concat + diff + Hadamard); pre-PCA reduction (24D) into a **PauliFeatureMap** (reps=2) on 16 qubits; QSVC implementation with `FidelityQuantumKernel`; three classical baselines tuned via GridSearchCV (Logistic Regression, Random Forest, Extra Trees); **stacking ensemble** combining QSVC and the tuned classical models; **hard negative sampling**; OGB-style temporal splits with leakage audit; primary metric (PR-AUC).
- **Preliminary data already observed:** PR-AUC **0.7987** on the Hetionet Compound-treats-Disease (CtD) benchmark from the stacking ensemble; PR-AUC 0.7216 from QSVC alone; PR-AUC 0.7838 from RandomForest-Optimized; PR-AUC 0.7807 from ExtraTrees-Optimized; PR-AUC 0.7408 from a ZZ-feature-map stacking ensemble. Variant-by-variant detail is in `README.md`.
- **Decisions formalized at pre-registration (not blinded):** Final choice of four edge types for evaluation; addition of two new baselines (R-GCN, TransE); commitment to hardware experiment scope; statistical analysis plan (paired bootstrap with conjunction-across-baselines decision rule).
- **Decisions still blinded:** Hardware experiment results (have not yet run); Pauli Path ZNE extrapolation results; R-GCN and TransE baseline results (have not yet been implemented); all sensitivity analyses including the ZZ-feature-map sensitivity comparison.

H1 (primary methodology hypothesis) is **data-aware confirmation**, not blinded hypothesis test. The hypothesis was constructed to formalize the methodology's apparent strengths based on existing QSVC and three-classical-baseline data. H1b (ensemble headline) is similarly data-aware. H2 and H3 are blinded with respect to the hardware experiments which have not yet run.

Reviewers must treat H1 and H1b accordingly. The methodology contribution is the QSVC-on-Hetionet methodology demonstration with the full baseline panel, plus the hybrid quantum-classical stacking ensemble; it is not a blinded hypothesis test. The contribution stands or falls on whether the methodology produces credible results against strong baselines including the not-yet-implemented R-GCN and TransE, not on whether H1/H1b are "confirmed" in a blinded sense.

This disclosure is the honest version of retroactive pre-registration. Reviewers who demand fully blinded pre-registration will reject this work; that is acknowledged. The authors believe that documenting in-flight decisions is more useful than fabricating a clean pre-registration narrative.

## CRediT author contributions (provisional)

- **Jonathan Beale:** conceptualization, methodology, software, formal analysis, investigation, writing — original draft, writing — review and editing, project administration.
- **Mark A. Jack:** methodology review (specifically QSVC kernel design and hardware-experiment design), writing — review and editing.
- **Kevin Robinson:** writing — review and editing, validation (specifically reviewing methodology accessibility for non-QML-specialist readers).
- **Abdulrehman Elsayed:** writing — review and editing.

---

## 1. Study identification

### 1.1 Working title

> Hybrid Quantum-Classical Kernel Methods for Biomedical Link Prediction on Hetionet: A Methodology Study with Hardware-Validated QSVC, a Stacking Ensemble, and Comprehensive Baseline Comparison

### 1.2 Research question

Can quantum support vector classification (QSVC) with Pauli-feature-map quantum kernels — alone, and as a component of a stacking ensemble with tuned classical learners — outperform a comprehensive panel of classical baselines (Logistic Regression, Random Forest, Extra Trees, R-GCN, TransE) on biomedical link prediction over Hetionet's drug-repurposing edge subgraph, evaluated with OGB-style protocols and validated on real quantum hardware (IBM Torino with Pauli Path ZNE error mitigation)?

### 1.3 Hypotheses

**H1 (Primary, data-aware confirmation):** On the Hetionet drug-repurposing benchmark using four edge types (Compound-treats-Disease, Compound-resembles-Compound, Compound-binds-Gene, Disease-associates-Gene), QSVC with PauliFeatureMap (reps=2) on 16-qubit pre-PCA-reduced RotatE pair features outperforms each of the five classical baselines (LogReg, Random Forest, Extra Trees, R-GCN, TransE) on PR-AUC. Decision rule: 95% paired bootstrap CI on per-fold PR-AUC differences excludes zero in the favorable direction for all five baselines simultaneously.

**H1b (Primary, data-aware confirmation, headline):** The hybrid quantum-classical *stacking ensemble* of QSVC + tuned classical models outperforms each individual model in the panel — including the best-performing classical baseline — on PR-AUC. Decision rule as in H1 but applied to ensemble vs. each individual model.

**H2 (Secondary, blinded):** QSVC hardware-evaluated kernels on IBM Torino with Pauli Path ZNE produce PR-AUC within 5 percentage points of QSVC simulator-evaluated kernels at the same problem sizes (10, 15, 20 graph feature dimensions). Decision rule: 95% bootstrap CI on PR-AUC difference (hardware-ZNE − simulator) includes zero or favors hardware within 5 percentage points.

**H3 (Tertiary, blinded):** QSVC kernel computation cost on IBM Torino with Pauli Path ZNE scales sub-quadratically in problem size on the 10/15/20 problem-size grid. Decision rule: linear regression on log(compute time) vs log(problem size) yields slope coefficient with 95% CI upper bound below 2.0.

### 1.4 Falsifiability and outcome neutrality

Authors commit to publishing a complete report regardless of which hypotheses are supported.

- If H1 fails on any baseline: paper reports as "QSVC competitive with [supported] baselines, methodology limitations against [failing] baseline." Weaker but still publishable. R-GCN failure is the most plausible scenario; if R-GCN ties or beats QSVC the paper frames as "QSVC competitive with KG-aware methods, outperforms shallow classical methods."
- If H1b fails (ensemble does not strictly improve over the best individual): the headline becomes "competitive ensemble with QSVC contributing comparable signal," and the paper foregrounds H1 alone instead.
- If H2 fails: hardware results far from simulator. Paper discusses likely causes (noise, ZNE limitations). H1 still stands as simulator-validated methodology result.
- If H3 fails: scaling super-quadratic. Forward-looking applicability weakened.
- If all four fail: paper publishable as null-result methodology study with discussion of why hybrid quantum-classical kernels do not provide advantage on this task.

---

## 2. Background and motivation

Biomedical knowledge graphs encode rich relational information about diseases, drugs, genes, and their interactions. Hetionet (Himmelstein et al., 2017) is a widely-studied biomedical knowledge graph with ~47K nodes, ~2.25M edges, 11 metanode types, and 24 edge types. Link prediction on Hetionet has direct applications to drug repurposing, hypothesis generation, and biomedical literature mining.

Classical link prediction methods on Hetionet include shallow embedding methods (TransE, RotatE), graph neural networks (R-GCN, CompGCN), and metapath-based feature engineering followed by standard classifiers. Performance on the drug-repurposing subset is well-characterized in the literature.

Quantum kernel methods, particularly QSVC (Havlíček et al., 2019), have been proposed as quantum-machine-learning approaches with theoretical advantages on certain task structures. Knowledge-graph link prediction has been suggested as a domain where quantum-kernel methods may be competitive with classical methods due to the rich combinatorial structure of relational data. **Hybrid quantum-classical ensembles**, in which a QSVC contributes alongside tuned classical learners under a meta-learner, are a less-studied configuration whose practical contribution this study quantifies.

Published QML-on-knowledge-graphs work is sparse. A systematic methodology study comparing QSVC and a hybrid stacking ensemble with comprehensive classical baselines on standardized biomedical KG link prediction, with hardware-validation experiments, has not been published as of pre-registration date.

---

## 3. Study scope

### 3.1 Knowledge graph

Hetionet v1.0. Snapshot SHA-256 hashes and access date recorded in [`docs/reproducibility/hetionet_snapshot.md`](../docs/reproducibility/hetionet_snapshot.md) — regenerable via `python scripts/record_hetionet_hash.py`. Loader: `kg_layer/kg_loader.py:extract_task_edges` and `kg_layer/kg_loader.py:prepare_link_prediction_dataset`.

### 3.2 Edge types for evaluation

Four edge types, forming the drug-repurposing inference subgraph:
- Compound-treats-Disease (CtD)
- Compound-resembles-Compound (CrC)
- Compound-binds-Gene (CbG)
- Disease-associates-Gene (DaG)

The headline evaluation focuses on **CtD**; CrC, CbG, DaG are reported as breadth.
Other 20 Hetionet edge types are out of scope.

### 3.3 Out-of-scope task variants

- Node classification, graph clustering, anomaly detection
- Other knowledge graphs (DRKG, OpenBioLink, BiolinkBERT)
- Real-time inference deployment
- Wet-lab biological validation of predicted links
- Production drug-repurposing recommendations

---

## 4. Pair feature representation

Already-executed component. Pair features are constructed by combining **full-graph RotatE node embeddings** of the head and tail entities, not by metapath path-counting.

Pipeline:

1. **Embedding training.** RotatE embeddings are trained on **all 24 Hetionet relations** (full-graph) with embedding dimension 128 over 200 epochs via PyKEEN (`kg_layer/advanced_embeddings.py`). Diversity diagnostics in `experiments/embedding_diversity_report.py`.
2. **Pair feature construction.** For a candidate (head, tail) pair, three pair-feature operations are concatenated: element-wise concatenation, head − tail, and head ⊙ tail (Hadamard product). See `kg_layer/enhanced_features.py:EnhancedFeatureBuilder`.
3. **Pre-PCA reduction.** The pair feature is projected to **24 dimensions** before quantum kernel evaluation. PCA is fit on training only and applied to validation/test.
4. **Quantum encoding dimension.** The pre-PCA features are encoded into a **16-qubit** quantum kernel via the feature map in §5. Pre-PCA dim 24 → final kernel dim 16 (the additional reduction lives in `quantum_layer/qml_encoder.py:encode_features_quantum`).

For hardware experiments (§5.4), the same pipeline applies with feature dimensions 10/15/20 in place of 16, calibrated to remain within ZNE-mitigatable noise on current hardware.

---

## 5. Quantum kernel methodology

### 5.1 Quantum kernel feature map (PRIMARY)

**PauliFeatureMap** with reps=2 (Qiskit Machine Learning). 16 qubits. Quantum kernel matrix entries computed via `FidelityQuantumKernel` (`quantum_layer/qml_model.py:QMLLinkPredictor._prepare_quantum_kernel`).

The PauliFeatureMap is primary because it produces the headline PR-AUC 0.7987 in the stacking ensemble. The originally drafted preregistration treated ZZFeatureMap as primary; the §12 appendix documents the change and its rationale.

### 5.2 Quantum kernel feature map (SUPPLEMENTARY SENSITIVITY)

**ZZFeatureMap** with depth 2. Reported in supplementary materials as a sensitivity analysis demonstrating that the methodology is not catastrophically dependent on feature-map choice. Two configurations only, not a full sweep.

### 5.3 QSVC training and inference

Resulting kernel matrix passed to scikit-learn's `SVC` for SVM training. Hyperparameter C tuned on validation split: {0.05, 0.1, 1.0, 10.0}; best-on-validation value is **C = 0.1** (locked in `utils/preregistered_constants.py`).

### 5.4 Hybrid quantum-classical stacking ensemble (HEADLINE)

A **stacking ensemble** combines QSVC predictions with three GridSearchCV-tuned classical learners (Logistic Regression, Random Forest, Extra Trees). The meta-learner is a logistic regression on out-of-fold base-learner probabilities. This is the configuration that produces PR-AUC 0.7987 on Hetionet CtD and is the headline reported in the manuscript abstract. See `quantum_layer/quantum_classical_ensemble.py`.

### 5.5 Variational classifier comparison

VQC with hardware-efficient ansatzes (RealAmplitudes, EfficientSU2, TwoLocal) and SPSA optimizer (`quantum_layer/qml_model.py`). Reported as supplementary, not primary. Best observed PR-AUC is 0.5474 (RealAmplitudes reps=4); this is reported alongside QSVC for completeness.

### 5.6 Hardware experiments (forward-looking)

IBM Torino (Heron r1, 133 qubits) primary backend. IBM Brisbane (Eagle r3, 127 qubits) backup. Backend routing in `quantum_layer/quantum_executor.py`.

Experiment scope: 3 problem sizes (10, 15, 20 dimensions) × 3 backend snapshots (different calibration cycles) × 100 evaluation samples per condition = 900 hardware-evaluated samples.

Each sample: full QSVC kernel matrix entry computed via real quantum circuits with shot count calibrated to per-circuit precision target.

### 5.7 Pauli Path ZNE error mitigation

For each hardware kernel matrix entry:
1. Compute at noise scaling 1× (raw)
2. Compute at noise scaling 3× (fold-based circuit folding)
3. Compute at noise scaling 5× (fold-based circuit folding)
4. Extrapolate to zero-noise via linear regression
5. Extrapolate to zero-noise via Richardson extrapolation
6. Report both extrapolations and raw unmitigated result

---

## 6. Baselines

### 6.1 Already-executed baselines

**Logistic Regression** with L2 regularization. Hyperparameter C tuned on validation split.

**Random Forest.** Hyperparameters (n_estimators, max_depth, min_samples_split) tuned via GridSearchCV.

**Extra Trees.** Hyperparameters tuned via GridSearchCV.

All three implemented competently with full hyperparameter tuning. Same train/val/test splits as QSVC. See `classical_baseline/train_baseline.py` and `scripts/run_optimized_pipeline.py`.

(SVM with RBF kernel was an earlier baseline; it is superseded in the headline panel by Random Forest and Extra Trees, which dominate it. SVM-RBF is reported in supplementary materials for completeness.)

### 6.2 Forward-looking baselines

**R-GCN (Relational Graph Convolutional Network).** PyTorch Geometric implementation. Hyperparameters: hidden dim ∈ {64, 128, 256}, layers ∈ {2, 3}, lr ∈ {0.001, 0.01}, dropout ∈ {0.1, 0.3}, num_bases ∈ {None, 30}. Tuning budget 50 trials. Best-on-validation selected.

**TransE.** PyKEEN implementation. Hyperparameters: embedding dim ∈ {50, 100, 200}, lr ∈ {0.0001, 0.001, 0.01}, margin ∈ {1.0, 2.0, 5.0}, epochs ∈ {100, 300, 500} with early stopping, negative sampling ratio ∈ {1, 5, 10}. Tuning budget 50 trials. Best-on-validation selected.

### 6.3 Why these baselines

Three classical baselines (LogReg, RF, ET) are already implemented and provide a methodology floor. Adding R-GCN (KG-aware deep learning) and TransE (KG embedding) ensures the baseline panel includes modern KG-aware methods that reviewers will expect to see.

Other potential baselines (RotatE-as-classifier, ComplEx, CompGCN, BiolinkBERT) are not in scope. RotatE is used here only as a *feature extractor*, not as a classifier; this distinction is explicit in the manuscript.

---

## 7. Evaluation infrastructure

### 7.1 Train/val/test splits

OGB-style temporal split with leakage audit. Specifically:
- 70% training edges, 15% validation, 15% test (deterministic seed `SPLIT_SEED = 20251015`)
- Negative samples: 1:1 ratio with positive edges, generated via **hard negative sampling** (closest non-positive pairs in embedding space — see `kg_layer/kg_loader.py:prepare_link_prediction_dataset`). Hard sampling was selected over random sampling based on observed PR-AUC improvements; this is documented as a pre-existing decision.
- Leakage audit: ensured no node-pair appears in both training and test (excluding the original positive instance). Audit was conducted retroactively after an issue was identified. Corrected splits are the basis for all forward-looking work; the manuscript reports only post-audit results.

### 7.2 Test set sealing

Test partition is referenced for evaluation only at two pre-registered unsealing points; access is documented in the reproducibility appendix:
- Unsealing 1: at conclusion of all baseline implementations and simulator QSVC + ensemble evaluation, before hardware experiments. Computes simulator-only test results.
- Unsealing 2: at conclusion of hardware experiments. Computes hardware-validated test results.

Each unsealing event is logged with methodology version (git commit), configuration evaluated, and reason. The audit log is permanent. (A lightweight sealed-dataset wrapper may be added for enforcement; the convention itself is the commitment.)

### 7.3 Metrics

**Primary:** PR-AUC (precision-recall area under curve, computed via `sklearn.metrics.average_precision_score`). Standard for imbalanced biomedical link prediction.

**Secondary:** ROC-AUC. Standard for general link prediction.

**Hardware-specific:** Compute time per kernel matrix entry, including all noise-scaling levels for ZNE.

---

## 8. Statistical analysis plan

### 8.1 Primary analysis (H1 and H1b)

Per-fold PR-AUC differences (QSVC − each baseline for H1; ensemble − each baseline for H1b) on the test partition. 5-fold cross-validation.

Paired bootstrap with **10,000 resamples** produces 95% CI on the mean per-fold PR-AUC difference for each baseline. Implementation: `utils/bootstrap_ci.paired_bootstrap_pr_auc_difference` and `utils/bootstrap_ci.conjunction_across_baselines` — in-tree, pure NumPy + scikit-learn, no external statistical dependency.

**Decision rule:** H1 / H1b supported iff 95% CI excludes zero AND lies in the favorable direction for ALL FIVE baselines simultaneously.

The conjunction-across-baselines requirement is intentional. If the ensemble beats LogReg, RF, ET but loses to R-GCN: H1b not supported in its strong form, paper frames as "competitive with KG-aware methods, outperforms shallow classical methods."

### 8.2 Secondary analysis (H2)

Per-condition PR-AUC differences (QSVC-hardware-ZNE − QSVC-simulator) at each of the 3 problem sizes × 3 backend snapshots = 9 conditions.

Bootstrap with 10,000 resamples produces 95% CI on the mean per-condition PR-AUC difference.

**Decision rule:** H2 supported iff 95% CI lies within ±5 percentage points of zero (both bounds within the band) OR favors hardware (lower bound > -5).

### 8.3 Tertiary analysis (H3)

Linear regression on log(per-kernel-entry compute time) vs log(problem size) across the 3 problem sizes × 3 backend snapshots × 100 samples per condition.

**Decision rule:** H3 supported iff slope 95% CI upper bound < 2.0.

### 8.4 Bootstrap procedure

10,000 resamples, deterministic seed (`BOOTSTRAP_SEED = 20260504`, locked in `utils/preregistered_constants.py`). The paired-bootstrap primitives live in-tree at `utils/bootstrap_ci.py`. See §12.1 for the methodology mapping from the initial scaffold draft, including why no external statistics package is taken as a dependency.

### 8.5 Multiple comparison correction

Within H1: conjunction-across-baselines is the correction. No additional Bonferroni applied.
Across H1/H1b/H2/H3: no correction; conceptually distinct.

### 8.6 Sensitivity analyses pre-committed

- **Feature-map choice (PauliFeatureMap vs ZZFeatureMap)** — a primary sensitivity. ZZ is the supplementary comparison; PauliFeatureMap (reps=2) is primary.
- ZNE extrapolation method (linear vs Richardson reported separately)
- Backend (Torino primary vs Brisbane backup)
- Backend snapshot variability (different calibration cycles)
- Two-baseline comparison (QSVC vs R-GCN-only and QSVC vs TransE-only) reported separately to surface which baselines are decisive
- **Ensemble configuration** — stacking (primary) vs weighted-average (sensitivity); manual `ensemble_quantum_weight` settings reported as having no additional effect once stacking learns weights

---

## 9. Reproducibility commitments

### 9.1 Code

Open-source release at manuscript submission to public GitHub repository:
- All baseline implementations (LogReg, RF, ET, R-GCN, TransE)
- QSVC kernel implementation with PauliFeatureMap (primary) and ZZFeatureMap (supplementary)
- Stacking ensemble implementation
- VQC implementation
- Pauli Path ZNE implementation
- Evaluation pipeline including OGB-style protocol
- Statistical analysis utilities (`utils/bootstrap_ci.py`)
- Pre-registered constants (`utils/preregistered_constants.py`)
- Test suite including synthetic-KG fixtures

License: MIT (per existing repository license).

### 9.2 Data

Hetionet v1.0 publicly available. Snapshot SHA-256 hashes and access date recorded in [`docs/reproducibility/hetionet_snapshot.md`](../docs/reproducibility/hetionet_snapshot.md).

Train/val/test splits derived from Hetionet via deterministic procedure (`SPLIT_SEED = 20251015`). Splits regenerable bit-identically.

### 9.3 Hardware results

IBM job IDs for all hardware runs recorded in reproducibility appendix. Hardware results reproducible only on IBM Quantum hardware with appropriate access.

### 9.4 Computational environment

Code commit hash, environment hash (SHA-256 of `pip freeze`), OS fingerprint recorded for every experiment run.

### 9.5 Reproducibility check

Before submission: clean-room reproduction of all manuscript numerical results from released code. Bit-identical match for non-quantum results; within-noise-tolerance match for quantum hardware.

---

## 10. Timeline

| Phase | Calendar quarter | Milestone |
|---|---|---|
| Retroactive scaffolding (this) | Q2 2026 | Charter and pre-registration locked |
| R-GCN and TransE baselines implemented | Q2-Q3 2026 | New baselines added |
| Hardware experiments | Q2-Q3 2026 | QSVC + ZNE on IBM Torino |
| **OSF pre-registration submission** | **Q3 2026** | Before final analysis runs |
| Statistical analysis and sensitivity analyses | Q3-Q4 2026 | All bootstrap CIs computed via `utils/bootstrap_ci.py` |
| Manuscript draft complete | Q4 2026 | Methodology paper for *Quantum Machine Intelligence* |
| **Manuscript submission** | **Q1 2027** | Anticipated |
| Reviewer revisions | Q2-Q3 2027 | Standard cycle |

If by Q4 2026 the methodology results have not coalesced into a coherent narrative, project pauses for reassessment.

---

## 11. Funding and conflicts of interest

**Funding:** Internal QGG. No external grants for this study.

**Conflicts of interest:** Authors are affiliated with Quantum Global Group, which provides quantum-computing consulting services. The methodology paper is not directly tied to a productized offering. Authors disclose this conflict explicitly.

**Data and code availability:** Open-source release at manuscript submission. Hetionet v1.0 publicly accessible. Hardware run IDs recorded.

**Customer data:** No customer or patient data used. All data from public benchmarks.

---

## 12. Appendix: changes from initial study design

This section logs deviations from any initial draft of this preregistration. The preregistration is locked at OSF submission (anticipated Q3 2026); modifications afterward require an explicit amendment recorded in this section.

### 12.1 Methodology mapping (initial scaffold → reconciled v1)

The initial scaffold of this preregistration (drafted from the `qgg-hybrid-qml-kg-biomedical` research-rigor template) committed to a different primary configuration than the project actually executed. The reconciliation:

| Item | Initial scaffold draft | Reconciled v1 (this document) |
|---|---|---|
| Primary feature map | ZZFeatureMap, depth 2 | **PauliFeatureMap, reps=2** (with ZZ as the supplementary sensitivity) |
| Pair feature representation | Metapath path-count features (1024D, PCA→10/15/20) | **RotatE 128D node embeddings** combined via concat + diff + Hadamard, pre-PCA reduced to 24D, then encoded into 16 qubits |
| Negative sampling | 1:1 random, stratified by edge type | **Hard negative sampling** (1:1) |
| Headline classifier | QSVC alone (H1 only) | **Stacking ensemble** of QSVC + tuned classical learners (H1b headline), with QSVC alone retained as H1 |
| Embedding training | None (metapaths are the features) | **Full-graph RotatE**, 128D, 200 epochs (PyKEEN) |
| Statistical-analysis dependency | External `qgg_shared.statistics` (QGG monorepo, not on PyPI) | In-tree `utils/bootstrap_ci.py` — pure NumPy + scikit-learn, no extra dep |

The reconciliation was made because the initial scaffold draft did not match what the project had already executed. Per the retroactive-disclosure framing, the right move is to document what was *actually* run rather than re-run experiments to match a draft. The honest disclosure block in §"Critical disclosure" identifies all data-aware decisions that informed this reconciliation.

### 12.2 Open editorial decision (resolved)

H1 (QSVC alone) is retained as a methodology hypothesis even though the headline number comes from H1b (the stacking ensemble). H1 is the cleaner methodology claim — it isolates the quantum-kernel contribution from any classical-ensemble effect. H1b is the fuller picture and the headline. Both are reported.

### 12.3 Subsequent amendments

[None at preregistration submission.]

---

*This pre-registration is retroactive. Decisions made before pre-registration are documented in §"Critical disclosure" and §"Already-executed components" throughout. Hypotheses formulated with knowledge of preliminary data are flagged. Hypotheses blinded with respect to remaining experiments are flagged. Reviewers must consider the retroactive disclosure when evaluating methodology rigor.*
