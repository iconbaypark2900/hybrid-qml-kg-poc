# 01 — Phase Alignment (Retroactive)

**Project:** hybrid-qml-kg-poc
**Gate:** 1 of 5 (Phase Alignment)
**Decision date:** 2026-05-02 (retroactive scaffolding date)
**Project active since:** Earlier 2025 (exact date approximate; pre-registration absent until now)
**Status:** Locked. Retrospective; further structural changes require charter amendment.

---

## What "retroactive" means here

This project was active before being scaffolded through the QGG research playbook. The charter describes decisions that were made *implicitly* during execution and surfaces them as explicit commitments going forward. This is structurally different from a greenfield charter — Gate 1 here is partly description (what the project actually is) and partly decision (what it commits to from this point forward).

The discomfort of retroactive scaffolding is intentional. Several decisions that would have been pre-registered at greenfield Gate 4 cannot be pre-registered now (training has already occurred on data that should have been blinded; baselines have already been selected; some hyperparameters were tuned on what should have been held-out data). The honest path is to acknowledge these explicitly rather than fabricate a clean pre-registration narrative.

Reviewers of the eventual paper must be told which aspects were pre-registered and which were not. The pre-registration document at Gate 4 (`preregistration/osf_preregistration_v1.md`) distinguishes these explicitly.

## Path positioning

**Path A (research)** with **secondary Path B (consulting) potential** if biomedical/pharma audiences engage warmly.

The project is research-first. The output is a peer-reviewed methodology paper demonstrating that hybrid quantum-classical kernel methods (QSVC alone and as a component of a stacking ensemble) can outperform a comprehensive panel of classical baselines on knowledge-graph link prediction in a specific biomedical regime. Path B is a *possibility* — pharma R&D groups have published interest in quantum-enhanced drug-repurposing methods — but is not the primary motivation.

There is no productized demo planned for this project beyond the Streamlit dashboard / FastAPI service that already exist for reproducibility. Knowledge-graph link prediction tools exist commercially but adding quantum kernels to them is not currently a defensible product positioning at QGG's scale. If the methodology paper succeeds and pharma conversations open, productization scope becomes a separate project.

## Audience

**Primary audience: quantum machine learning research community.** Venues: *Quantum Machine Intelligence*, *Physical Review Applied*, *npj Quantum Information*, *NeurIPS workshop on quantum machine learning*. The methodology contribution is the strongest fit for QML-specific venues that understand both the QSVC machinery and the OGB-style evaluation rigor.

**Secondary audience: biomedical informatics community.** Venues: *Bioinformatics*, *Journal of Biomedical Informatics*. The biomedical link-prediction angle is real but the methodology contribution dominates the framing. A purely biomedical-informatics venue would expect biological-validation experiments (in-silico drug-target validation, literature-based confirmation) that this project does not commit to.

**Tertiary audience: drug-repurposing / pharma R&D.** Real but downstream. The methodology paper is the artifact that opens warm conversations; the conversations themselves are not part of this project's scope.

## Audience temperature

**Cold audience.** No existing reviewer relationships at QML venues. Method-paper publication strategy: rigorous baselines, transparent failure modes, OGB-standard evaluation, public code release.

The headline PR-AUC 0.7987 result on Hetionet biomedical link prediction (Compound-treats-Disease, hybrid stacking ensemble) is real and was achieved before scaffolding. The authors did not have warm-introduction support for that result; the methodology paper builds the credibility that subsequent warm conversations require.

## Scope shape

**Comprehensive scope** (in retrospect — was *de facto* comprehensive without being declared as such). The project has executed:

- QSVC with PauliFeatureMap (reps=2, primary) and ZZFeatureMap (depth 2, supplementary)
- Hybrid quantum-classical **stacking ensemble** — the configuration that produces the headline PR-AUC 0.7987
- VQC variational classifier comparison with multiple ansatz families
- Three classical baselines (Logistic Regression, Random Forest, Extra Trees) implemented competently with GridSearchCV hyperparameter tuning
- Full-graph **RotatE** embedding training (128D, 200 epochs, PyKEEN) plus ComplEx and DistMult comparisons
- Pair-feature construction via concat + diff + Hadamard ops
- Pre-PCA reduction (24D) before quantum kernel evaluation on 16 qubits
- **Hard negative sampling**, with diverse-negative variants explored and rejected
- OGB-standard benchmarking protocol with leakage audit
- IBM Brisbane and IBM Torino backend access established
- Optuna-based hyperparameter search
- A working Streamlit dashboard (six-page narrative) and FastAPI service (`middleware/api.py`)
- Hugging Face Spaces deployment

The scope already extends beyond a focused-then-comprehensive Gate 2 commitment. Retroactive Gate 2 acknowledges this and restructures *forward-looking* commitments around what's already done.

## Timeline

**Backward-looking:** Project execution began earlier 2025. Approximate elapsed effort to date: 8-12 months of part-time work across primary author and ad-hoc collaborators.

**Forward-looking:**

| Phase | Calendar quarter | Milestone |
|---|---|---|
| Retroactive scaffolding (this) | Q2 2026 | Charter and pre-registration locked; what's already done is documented; what's left is committed explicitly |
| R-GCN and TransE baselines | Q2-Q3 2026 | New baselines added per pre-registration §6.2 |
| Hardware experiments | Q2-Q3 2026 | QSVC on IBM Torino with Pauli Path ZNE error mitigation at 10/15/20 qubit problem sizes |
| Statistical hardening | Q3 2026 | Paired bootstrap CIs computed via `utils/bootstrap_ci.py`; sensitivity analyses run; data leakage audit re-verified |
| Manuscript draft complete | Q4 2026 | Methodology paper draft for QML venue |
| **Manuscript submission** | **Q1 2027** | Anticipated submission to *Quantum Machine Intelligence* (primary) |
| Reviewer revisions | Q2-Q3 2027 | Standard revision cycle |

The manuscript timeline is faster than greenfield QGG research projects because much of the methodological work is already done. The remaining work is hardening, hardware validation, and writing — not re-execution of the methodology comparison.

## 12-month success criteria

If the next 12 months produce all of the following, the project succeeded:

1. Methodology paper draft complete with all baselines pre-registered going forward, all sensitivity analyses run, all hardware experiments completed
2. Manuscript submitted to *Quantum Machine Intelligence* or equivalent QML venue
3. Open-source code release with full reproducibility infrastructure (datasets, splits, baseline implementations, evaluation pipeline, synthetic-KG fixture for tests, paired-bootstrap statistical utilities)
4. At least one biomedical-informatics or pharma conversation generated by the methodology artifact (warm introduction or cold response indicating real interest)

Honest threshold for "warm conversation": email exchange where the other party asks substantive methodology questions or proposes a follow-up call. Not "they replied to a cold email."

## Why this project, why now (retroactive answer)

The work was already happening because biomedical link prediction is the application domain where quantum-kernel methods have the strongest theoretical case for advantage on near-term hardware. Knowledge graphs have rich relational structure; kernel methods on graph features have established classical performance; the quantum-kernel literature has identified link prediction as a sensible target.

Why now (for scaffolding): the project has reached a point where the next milestones (hardware experiments, manuscript draft) require pre-registration discipline if the eventual paper is to clear peer review. The data leakage audit conducted earlier in the project surfaced exactly the kind of issue that pre-registration would have caught earlier. Continuing without scaffolding risks accumulating more such issues.

## Co-authors

Tentative author list:

- **Jonathan Beale** — corresponding author, lead author. Methodology, software, evaluation, manuscript.
- **Mark A. Jack** (Florida A&M, Physics) — review of quantum-kernel methodology and hardware-experiment design.
- **Kevin Robinson** (EdAdvance) — review of methodology accessibility for non-QML readers.
- **Abdulrehman Elsayed** — review. *(Affiliation and ORCID: **TBD** — see `preregistration/osf_preregistration_v1.md` authors table; update before OSF / manuscript submission.)*

CRediT roles will be filed at manuscript submission per Gate 4 commitment.

**Venue order and submission calendar** in the charter and pre-registration (**QMI → *npj Quantum Information* → *Bioinformatics***; OSF **Q3 2026**, manuscript **Q1 2027**) are **provisional** until all authors sign off — edit the pre-registration header block in one place when final.

## What this project is not

- It is not a productized SaaS or product offering. The Streamlit dashboard and FastAPI service exist for reproducibility, not as products.
- It is not a biological-validation study. The link-prediction methodology is evaluated on benchmark accuracy metrics (PR-AUC, ROC-AUC), not on biological correctness of predicted links. Biological validation requires wet-lab work or literature-based confirmation neither of which is in scope.
- It is not a quantum-advantage proof at hardware-relevant scales. The headline PR-AUC 0.7987 result is at problem sizes where classical baselines can also be evaluated. Hardware QSVC at scales where classical baselines become infeasible is not currently achievable on near-term hardware regardless of methodology choice. The methodology paper does not claim quantum advantage; it claims hybrid quantum-classical competitiveness with — and in the ensemble case, modest improvement over — strong classical baselines on a specific task.

## Risks identified at this gate (retroactive)

- **Pre-registration cannot cover decisions already made.** Some baselines were chosen, some hyperparameters were tuned, some splits were defined before scaffolding. The pre-registration document at `preregistration/osf_preregistration_v1.md` identifies these explicitly. Reviewers will see the distinction. This weakens but does not invalidate the methodology contribution.

- **Data leakage audit was retrospective.** The audit caught and corrected a leakage issue. The corrected protocol is now the basis for forward-looking work. But the fact that leakage existed at all weakens the project's reliability narrative. Pre-registration surfaces this explicitly rather than hiding it.

- **Concentration risk on IBM Torino.** Multiple QGG projects depend on IBM Torino. Brisbane backup pre-registered for each. True hardware diversification deferred.

- **QML venue calibration uncertainty.** Reviewer expectations at *Quantum Machine Intelligence* are still settling as the venue is relatively new. Backup venue (*npj Quantum Information*) has stricter expectations particularly around hardware-experiment scale. Manuscript may need restructuring if first venue rejects.

- **Pharma audience temperature unknown.** "Warm conversation" success criterion is genuinely uncertain. May produce zero conversations even if the paper is well-received in QML venues. Acknowledged honestly rather than rationalized.

## What goes into the next gate

Gate 2 (Scope) for this project is constrained by what's already executed. The remaining open scope decisions are:

- Which Hetionet edge types to include in the final evaluation (subset already in use; commit explicitly)
- Whether to add additional baselines beyond LogReg, RF, ET (commit one way or the other — committed: yes, R-GCN + TransE)
- Whether VQC results are reported as primary or as supplementary (current data: supplementary)
- Whether the headline is QSVC-alone or the stacking ensemble (committed: ensemble headline with QSVC alone reported alongside)
- Pauli Path ZNE error mitigation: commit to scope and target precision
- Hardware experiment scope: specific number of QSVC runs at specific problem sizes

These decisions are locked at Gate 2.
