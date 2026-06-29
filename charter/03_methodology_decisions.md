# 03 — Methodology Decisions (Retroactive)

**Project:** hybrid-qml-kg-poc
**Gate:** 3 of 5 (Methodology)
**Decision date:** 2026-05-02
**Status:** Locked. Backward-looking documentation of executed methodology + forward-looking commitment for remaining work.

> **2026-06 reproducibility note:** Current reproducible headline ensemble PR-AUC is **0.7805** (256D RotatE + MoA, seed 42) with 5-seed mean **0.7398 ± 0.038** (`results/multiseed/TABLE3.md`). References to **0.7987** below describe the legacy 128D `fast_mode` protocol.

---

## Methodology overview

The project applies QSVC with quantum-kernel feature maps — combined with tuned classical learners under a stacking ensemble — to biomedical link prediction on Hetionet. Classical baselines (LogReg, RF, ET, R-GCN, TransE) provide reference performance. Hardware experiments on IBM Torino with Pauli Path ZNE error mitigation provide hardware-validation evidence.

| Component | Choice | Status |
|---|---|---|
| Knowledge graph | Hetionet v1.0 | Already executed |
| Edge types | CtD (headline) + CrC, CbG, DaG | Locked at Gate 2 |
| Pair feature representation | RotatE 128D node embeddings, full-graph (200 epochs); concat + diff + Hadamard pair ops; pre-PCA → 24D | Already executed |
| Quantum encoding dimension | 16 qubits | Already executed |
| Quantum kernel feature map (PRIMARY) | **PauliFeatureMap, reps=2** | Already executed |
| Quantum kernel feature map (SUPPLEMENTARY SENSITIVITY) | ZZFeatureMap, depth 2 | Already executed |
| Quantum kernel implementation | `FidelityQuantumKernel` (Qiskit Machine Learning) | Already executed |
| Quantum classifier | QSVC (sklearn `SVC` on the precomputed quantum kernel matrix) | Already executed |
| QSVC regularization | C = 0.1 (best-on-validation) | Already executed |
| **Headline classifier** | **Stacking ensemble of QSVC + tuned LR/RF/ET** (PR-AUC 0.7987) | Already executed |
| Variational classifier comparison | VQC with RealAmplitudes / EfficientSU2 / TwoLocal; SPSA optimizer | Already executed |
| Classical baseline 1 | Logistic Regression with L2 regularization (GridSearchCV-tuned) | Already executed |
| Classical baseline 2 | Random Forest (GridSearchCV-tuned) | Already executed |
| Classical baseline 3 | Extra Trees (GridSearchCV-tuned) | Already executed |
| Classical baseline 4 | R-GCN (Relational Graph Convolutional Network) | Forward-looking — not yet implemented |
| Classical baseline 5 | TransE knowledge-graph embedding | Forward-looking — not yet implemented |
| Negative sampling | Hard negative sampling, 1:1 ratio | Already executed |
| Evaluation protocol | OGB-style with corrected leakage protocol | Already executed (post-audit) |
| Primary metric | PR-AUC (`average_precision_score`) | Already executed |
| Secondary metric | ROC-AUC | Already executed |
| Hardware backend (primary) | IBM Torino (Heron r1) | Forward-looking |
| Hardware backend (backup) | IBM Brisbane (Eagle r3) | Forward-looking |
| Error mitigation | Pauli Path ZNE (linear + Richardson) | Forward-looking |
| Statistical analysis | Paired bootstrap, 10,000 resamples (`utils/bootstrap_ci.py`) | Forward-looking |

## Quantum kernel feature map

Already executed: **PauliFeatureMap with reps=2** is primary. The PauliFeatureMap was selected after a sensitivity comparison against ZZFeatureMap depth 2; switching from ZZ to Pauli moved the stacking-ensemble PR-AUC from 0.7408 to 0.7987. Each input feature (16 dims after pre-PCA from 24) is mapped to a single qubit; the Pauli rotation structure plus entanglement creates richer feature interactions than ZZ on this task.

ZZFeatureMap is retained as a supplementary sensitivity comparison rather than dropped, because reviewers will want evidence that the methodology's headline performance is not catastrophically dependent on the feature-map choice. Two configurations only (Pauli reps=2 vs ZZ depth 2), not a full sweep.

## Hybrid quantum-classical stacking ensemble

The headline result is from a **stacking ensemble** combining QSVC and three GridSearchCV-tuned classical learners (Logistic Regression, Random Forest, Extra Trees). The meta-learner is a logistic regression on out-of-fold base-learner probabilities; stacking learns the optimal classical/quantum weights automatically (manual `ensemble_quantum_weight` settings produced no additional effect).

Reported alongside the ensemble: QSVC alone (PR-AUC 0.7216), the best individual classical baseline (RandomForest-Optimized, 0.7838), and Extra Trees (0.7807). Both H1 (QSVC alone vs each baseline) and H1b (ensemble vs each baseline) are pre-registered.

## Forward-looking baseline implementations

### R-GCN

Implementation plan: PyTorch Geometric's RGCNConv. Hyperparameters tuned on validation split:
- Hidden dimension: {64, 128, 256}
- Layers: {2, 3}
- Learning rate: {0.001, 0.01}
- Dropout: {0.1, 0.3}
- Number of bases (relation parameter sharing): {None, 30}

Tuning budget: ~50 trials. Best-on-validation configuration selected.

### TransE

Implementation plan: PyKEEN library. Hyperparameters tuned on validation split:
- Embedding dimension: {50, 100, 200}
- Learning rate: {0.0001, 0.001, 0.01}
- Margin: {1.0, 2.0, 5.0}
- Number of epochs: {100, 300, 500} with early stopping
- Negative sampling ratio: {1, 5, 10}

Tuning budget: ~50 trials. Best-on-validation configuration selected.

Both baselines: implementation reviewed before final results, evaluated on the same train/val/test splits as QSVC and the simpler classical baselines.

## Hardware experiment protocol

### QSVC kernel evaluation on hardware

For each hardware experiment:
1. Construct QSVC kernel matrix entries via real quantum circuits on IBM Torino
2. Apply Pauli Path ZNE with 3 noise scaling levels (1×, 3×, 5×) using fold-based circuit folding
3. Extrapolate to zero-noise via both linear and Richardson methods; report both
4. Use the resulting kernel matrix in classical SVM training/inference
5. Evaluate PR-AUC and ROC-AUC on test split

Problem sizes: 10, 15, 20 graph-feature dimensions. Calibrated to remain within ZNE-mitigatable noise regime on current hardware. (Note that the simulator headline runs at 16 qubits; hardware experiments evaluate the methodology at sizes that are tractable on current devices, not to match the simulator dimension.)

### Sensitivity analyses pre-committed

- **Feature-map choice (PauliFeatureMap primary vs ZZFeatureMap supplementary)** — primary sensitivity
- ZNE extrapolation method (linear vs Richardson)
- Backend (Torino vs Brisbane)
- Backend snapshot (different calibration cycles)
- Ensemble configuration (stacking primary, weighted-average sensitivity)

## Statistical analysis plan

### Decision rules

**H1 (QSVC-alone):** Paired bootstrap on per-fold PR-AUC differences (QSVC − each baseline). 10,000 resamples. 95% CI must exclude zero in the favorable direction for all five baselines simultaneously.

**H1b (stacking-ensemble headline):** Same as H1 but applied to ensemble − each baseline. Same decision rule.

**H2 (Secondary):** Bootstrap on per-condition PR-AUC differences (QSVC-hardware-ZNE − QSVC-simulator). 95% CI must include zero or lie within 5 percentage points favoring hardware.

**H3 (Tertiary):** Linear regression on log(hardware compute time) vs log(problem size). 95% CI on slope must have upper bound below 2.0.

Bootstrap implementation: `utils/bootstrap_ci.paired_bootstrap_pr_auc_difference` and `utils/bootstrap_ci.conjunction_across_baselines` (this repository, no external dependency). Seed and parameters locked in `utils/preregistered_constants.py` (`BOOTSTRAP_SEED = 20260504`, `BOOTSTRAP_N_RESAMPLES = 10_000`, `BOOTSTRAP_CONFIDENCE = 0.95`).

### Multiple comparison correction

Within H1 and H1b: five baselines tested simultaneously via conjunction-across-baselines decision rule. The conjunction is more conservative than any standard correction would impose (Bonferroni at α = 0.05/5 = 0.01 per baseline would require 99% CIs; the conjunction of 95% CIs is approximately the same significance threshold but with the additional requirement of unanimity).

Across H1 / H1b / H2 / H3: no Bonferroni — conceptually distinct claims at different layers of the methodology.

### Failure-mode publishability

If H1 fails on any baseline (say, R-GCN ties or beats QSVC alone): the paper reports as "QSVC competitive with KG-aware methods, outperforms shallow classical methods." Weaker but still publishable.

If H1b fails on any baseline: the ensemble headline becomes "competitive with [supported] methods, methodology limitations against [failing] method." Paper foregrounds H1 (QSVC alone) as the methodology claim instead.

If H2 fails: hardware results are too far from simulator results. Paper reports honestly with discussion of likely causes (noise dominates, ZNE insufficient for this problem class). H1/H1b still stand but the hardware-validation narrative is undermined.

If H3 fails: scaling is super-quadratic. Paper reports honestly. Forward-looking applicability is weakened but methodology contribution at current scales is unaffected.

If all four fail: paper still publishable as null-result methodology study. Reviewers value honest negative results in QML.

## Honest framing of what the project is and isn't

**The project demonstrates:**
- QSVC with PauliFeatureMap kernels can be competitive with classical baselines on Hetionet biomedical link prediction
- A hybrid quantum-classical stacking ensemble produces a meaningful headline PR-AUC (0.7987 on CtD)
- The methodology runs end-to-end on real quantum hardware with ZNE error mitigation (pending hardware experiments)
- Hardware-evaluated QSVC produces results within reasonable bounds of simulator results (pending)

**The project does NOT demonstrate:**
- Quantum advantage at scales where classical baselines fail
- Production-ready drug-repurposing predictions
- Generalization to other knowledge graphs without re-tuning
- That QSVC is the best quantum method for this task (other quantum kernels not systematically compared beyond Pauli vs ZZ)

This framing matches the anti-decoration discipline of sibling QGG research projects. The methodology paper says what it shows and acknowledges what it doesn't.

## Risks identified at this gate

- **Pre-existing PR-AUC 0.7987 (ensemble) and 0.7216 (QSVC alone) results could shift after R-GCN and TransE are added.** R-GCN in particular is a strong KG-aware baseline. Possible the QSVC-alone advantage shrinks or disappears against R-GCN; the ensemble margin may also narrow. Pre-registered to publish either way.

- **Hardware experiments may produce PR-AUC well below simulator.** Realistic on current hardware noise levels. H2 failure handled honestly per failure-mode publishability.

- **ZNE extrapolation can produce nonsensical values when noise is high.** Pre-registered to report both linear and Richardson; pre-registered to report unmitigated raw results alongside.

- **Reviewer pushback on retroactive scaffolding.** Some reviewers may consider the project insufficiently rigorous because key decisions were made before pre-registration. Mitigation: explicit disclosure in pre-registration, transparent reporting of what was pre-registered vs. what was data-aware.

- **PyTorch Geometric R-GCN implementation may have known issues with Hetionet's specific edge type structure.** Mitigation: implementation reviewed before final results; alternative R-GCN implementation (DGL) ready as fallback.

- **Stacking ensemble headline depends on having strong classical base learners.** If R-GCN or TransE prove dramatically stronger than the current LR/RF/ET base set, the ensemble structure may need re-tuning to incorporate them. Pre-registered to report whichever ensemble configuration the OGB-style validation favors.

## What's deferred to Gate 4 (pre-registration)

Gate 4 commits the formal OSF pre-registration. The pre-reg document `preregistration/osf_preregistration_v1.md` explicitly distinguishes:

- Decisions already made before pre-registration (most of the methodology)
- Decisions formalized at pre-registration (additional baselines, hardware experiment scope, statistical analysis plan)
- Hypotheses formulated with knowledge of preliminary data (H1, H1b)
- Hypotheses blinded with respect to remaining experiments (H2, H3)

This is the honest version of retroactive pre-registration. It will not satisfy reviewers who demand fully blinded pre-registration. It will satisfy reviewers who accept the realities of in-flight work being formalized.
