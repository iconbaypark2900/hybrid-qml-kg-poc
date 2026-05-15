# 02 — Scope Decisions (Retroactive)

**Project:** hybrid-qml-kg-poc
**Gate:** 2 of 5 (Scope)
**Decision date:** 2026-05-02
**Status:** Locked. Forward-looking scope decisions; backward-looking executed scope is documented in §"Already Executed".

---

## Already executed (descriptive, not decisional)

These decisions were made during project execution before scaffolding. They cannot be pre-registered retrospectively. They are documented here for transparency.

| Element | Choice already made | When |
|---|---|---|
| Knowledge graph | Hetionet (v1.0, ~47K nodes, ~2.25M edges, 11 metanode types, 24 edge types) | Project start |
| Headline edge type | Compound-treats-Disease (CtD) | Project start |
| Pair feature representation | Full-graph **RotatE 128D node embeddings** combined via concat + diff + Hadamard, pre-PCA reduced to 24D | Mid-project |
| Quantum kernel feature map | **PauliFeatureMap, reps=2** (primary); ZZFeatureMap depth 2 (sensitivity) | Mid-project |
| Quantum kernel implementation | `FidelityQuantumKernel` (Qiskit Machine Learning) on 16 qubits | Project start |
| QSVC regularization | C = 0.1 (best-on-validation in {0.05, 0.1, 1.0, 10.0}) | Mid-project |
| Headline classifier | **Stacking ensemble** of QSVC + tuned LR/RF/ET | Mid-project |
| Variational classifier comparison | VQC with RealAmplitudes, EfficientSU2, TwoLocal ansatzes; SPSA optimizer | Mid-project |
| Classical baselines selected | Logistic Regression, Random Forest, Extra Trees (GridSearchCV-tuned) | Project start |
| Negative sampling | **Hard negative sampling**, 1:1 ratio (diverse-negative variants explored, rejected) | Mid-project |
| Evaluation framework | OGB-style link prediction protocol | Mid-project (post-leakage-audit) |
| Train/val/test split | 70/15/15 with leakage audit | Mid-project (corrected) |
| Headline performance | **PR-AUC 0.7987** on Hetionet CtD from stacking ensemble | Pre-scaffolding |
| Hardware backends accessed | IBM Brisbane, IBM Torino | Pre-scaffolding |
| Optuna hyperparameter search | 30-trial ensemble objective, 20-trial QSVC and classical objectives | Pre-scaffolding |

These elements are *not* re-opened by retroactive scaffolding. They are what they are.

## Forward-looking scope decisions

### Edge types for final evaluation

**Decision: focus on the four drug-repurposing-relevant edge types.** Specifically: Compound-treats-Disease (CtD, headline), Compound-resembles-Compound (CrC), Compound-binds-Gene (CbG), Disease-associates-Gene (DaG). These four edge types form the core drug-repurposing inference graph and have the strongest published interest from pharma audiences.

The full Hetionet has 24 edge types. Including all 24 in the final evaluation dilutes the pharma-relevance narrative and inflates compute cost without clear methodological gain.

Rejected alternative: full 24-edge-type evaluation. Adds 5-10× compute, weakens narrative, doesn't materially strengthen the methodology claim.

### Additional baselines

**Decision: add two baselines for the final paper.** Going beyond LogReg/RF/ET:

- **Graph Neural Network baseline** (specifically R-GCN). Knowledge-graph-native classical method. Strong baseline that QML methodology papers are increasingly expected to beat.
- **Knowledge graph embedding baseline** (specifically TransE). Standard in OGB-style biomedical link prediction benchmarks. Reviewers expect to see at least one KG embedding method.

These two were considered earlier but not implemented as classifiers. Going forward they are committed scope. (Note: RotatE is already used in the project, but as a *feature extractor*, not as a classifier; this distinction is explicit in the manuscript.)

Rejected alternative: skip KG-aware baselines because the stacking ensemble already beats LogReg/RF/ET. This was the earlier *de facto* position. Reversed because reviewer pushback would be severe.

### Headline classifier

**Decision: the hybrid quantum-classical stacking ensemble is the headline.** PR-AUC 0.7987 is the headline number; QSVC alone (PR-AUC 0.7216) is reported alongside as the isolated quantum-kernel contribution. Both are subjects of pre-registered hypotheses (H1b for ensemble, H1 for QSVC alone). See `preregistration/osf_preregistration_v1.md` §1.3.

Rejected alternative: report QSVC alone as the headline. Loses the headline number; understates what the methodology actually delivers.

### VQC reporting

**Decision: VQC reported as supplementary, not primary.** Based on existing data, VQC remains near random (best PR-AUC 0.5474 with RealAmplitudes reps=4, SPSA). The methodology contribution is hybrid-QSVC-on-knowledge-graphs; VQC is a secondary comparison demonstrating that the choice of QSVC over VQC is itself an informed methodological choice, not a default.

VQC results in supplementary materials with full reporting (no selective omission). Manuscript main text focuses on QSVC + ensemble vs classical baselines.

### Pauli Path ZNE error mitigation

**Decision: implement Pauli Path ZNE for the QSVC hardware experiments only.** The Google Quantum AI methodology applies to QSVC kernel evaluation circuits. Implementation scope:

- Linear and Richardson extrapolation, both reported
- Fold-based circuit folding (3 noise levels: 1×, 3×, 5×)
- Per-circuit calibration data recorded for sensitivity analysis
- Compared against unmitigated raw results

ZNE for VQC is not in scope. Compute cost too high for the supplementary-status of VQC results.

### Hardware experiment scope

**Decision: 3 problem sizes × 3 backend snapshots × 100 evaluation samples per condition.** Total: 900 hardware-evaluated samples for QSVC. Problem sizes: 10, 15, 20 graph-feature dimensions (calibrated to fit on IBM Torino with reasonable error rates after ZNE). Backends: IBM Torino primary, IBM Brisbane backup, with snapshots taken across calibration cycles to capture noise variability.

Rejected alternative: single problem size at single backend. Insufficient for the hardware-validation narrative reviewers expect.

Rejected alternative: 5+ problem sizes. Compute cost prohibitive given Torino queue contention and other QGG projects sharing access.

## Hypotheses pre-committed (from this point forward)

These are the methodology hypotheses going into the manuscript. They are pre-registered at Gate 4 with the explicit caveat that pre-existing data informed their formulation (i.e., these are not blinded hypotheses; they are formalized commitments based on what the data has already shown).

**H1 (Primary, QSVC-alone):** On the four-edge-type Hetionet drug-repurposing benchmark, QSVC with PauliFeatureMap (reps=2) outperforms each of the five classical baselines (LogReg, RF, ET, R-GCN, TransE) on PR-AUC. Decision rule: 95% paired bootstrap CI on per-fold PR-AUC difference excludes zero in the favorable direction for all five baselines simultaneously.

**H1b (Primary, stacking-ensemble headline):** The hybrid quantum-classical stacking ensemble outperforms each individual model in the panel on PR-AUC. Decision rule as in H1 but applied to ensemble vs. each individual model.

**H2 (Secondary): QSVC hardware-evaluated kernels with ZNE error mitigation produce PR-AUC within 5 percentage points of QSVC simulator-evaluated kernels at the same problem sizes.** Decision rule: 95% bootstrap CI on PR-AUC difference between hardware-ZNE and simulator results includes zero or favors hardware within 5 points.

**H3 (Tertiary): QSVC kernel computation cost on IBM Torino with Pauli Path ZNE scales sub-quadratically in problem size (graph feature dimension) on the 10/15/20 problem-size grid.** Decision rule: linear regression on log(compute time) vs log(problem size) yields slope coefficient with 95% CI upper bound below 2.0.

H1/H1b are primary because they encode the methodology claim. H2 establishes that hardware results are credible (otherwise the methodology is simulator-only). H3 establishes that the methodology has plausible scaling characteristics for forward-looking work.

## Honest disclosure of pre-existing data

The pre-registration document at `preregistration/osf_preregistration_v1.md` states explicitly:

> H1, H1b, H2, H3 were formalized after data informing them was available. Specifically: PR-AUC 0.7987 from the stacking ensemble on Hetionet CtD was observed before this pre-registration. Classical baselines were tuned and evaluated before this pre-registration. The hypothesis structure was constructed to formalize the methodology's apparent strengths rather than to blindly test predictions. Reviewers should treat H1 and H1b as "data-aware confirmation" rather than "blinded hypothesis test." H2 and H3 are blinded with respect to the hardware experiments which have not yet run.

This disclosure weakens H1/H1b's status compared to a greenfield pre-registration. It does not invalidate the methodology contribution — the contribution is the hybrid-QSVC-on-knowledge-graphs methodology demonstration, not a blinded hypothesis test. The disclosure is the honest version of what retroactive scaffolding produces.

## Out-of-scope (explicitly)

- **Biological validation of predicted drug-repurposing edges.** Wet-lab or literature-based confirmation of specific predicted associations is not in scope. Pharma audiences asking "which drugs do you think work for which diseases" get a methodology-paper-and-code response, not validated predictions.

- **Production deployment.** The Streamlit dashboard and FastAPI service exist for reproducibility, not as products.

- **Other knowledge graphs.** Hetionet only. DRKG, OpenBioLink, and other biomedical KGs are not in scope. Methodology generalization to other KGs is mentioned in manuscript future-work section.

- **Other tasks beyond link prediction.** Node classification, graph clustering, etc. are not in scope.

- **Quantum advantage at scales where classical baselines fail.** Not claimed. Methodology paper's framing is competitiveness, not advantage.

- **Real pharma collaboration.** Out of scope as a project deliverable. The methodology paper is the artifact that *enables* such collaboration; the collaboration itself is a separate downstream project.

## Risks identified at this gate

- **The "data-aware confirmation" disclosure may be unacceptable to some reviewers.** Mitigation: the methodology contribution is novel (hybrid-QSVC-on-Hetionet with this specific scope and these specific baselines is not previously published); the pre-existing data does not invalidate that novelty.

- **Adding R-GCN and TransE baselines is real work.** Both require implementation, hyperparameter tuning, and evaluation. Estimated 4-6 weeks. This was the largest forward-looking scope addition; without it the baselines story is too weak.

- **ZNE on hardware can be expensive.** 900 hardware-evaluated samples × 3 noise scaling levels = 2700 hardware circuits per backend snapshot. At IBM Torino's typical queue rates, this is real wall time. Mitigation: budget allocated, alternative backend (Brisbane) ready as fallback.

- **Hetionet's age.** Hetionet was published in 2017. Newer biomedical KGs (DRKG, OpenBioLink) include more recent literature. Reviewers may ask "why not a newer KG." Manuscript will address: Hetionet has the strongest published baseline numbers and the largest body of comparison literature; switching to a newer KG would lose that comparison context.
