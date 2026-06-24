# Master Work Plan — Hybrid QML-KG Drug Repurposing Platform

**Project:** Hybrid Quantum Machine Learning + Knowledge Graph for Biomedical Drug Repurposing  
**Lead:** Jonathan Beale  
**Collaborators:** Jack, Robinson, Elsayed  
**Target:** arXiv v2 → *Quantum Machine Intelligence* submission (Q1 2027)  
**OSF bundle deadline:** Q3 2026  
**Document last updated:** 2026-06-24  

---

## How to use this document

This document is the single source of truth for what needs to be done, in what
order, and why. Work through the tiers in sequence. Do not start Tier 2 until
Tier 1 is complete. Items within a tier can be parallelized if compute permits.

Completion criteria for each item are marked with **Done when:** lines. Nothing
is done until those conditions are met.

---

## Tier 1 — Blocks arXiv v2 Submission

These items are explicitly committed to in the current paper text or are
required to produce a compilable, reviewable PDF. None have result files yet.
Estimated total effort: **2–4 days compute + 1 day writing**.

---

### 1.1 MoA Feature Benchmark

**Why:** The paper introduces a 10-feature mechanism-of-action module in §4.5
and §7.3 as the fix for the score-validity inversion problem (Abacavir → ocular
cancer scores 0.793 with zero clinical support). The module is fully implemented
and the flag exists, but no PR-AUC result has ever been produced with it.

**Command:**

```bash
python scripts/run_optimized_pipeline.py \
  --relation CtD --full_graph_embeddings \
  --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24 \
  --run_ensemble --ensemble_method stacking --tune_classical --fast_mode \
  --use_moa_features \
  --results_dir results/moa_benchmark
```

**What to record:**
- PR-AUC before vs after MoA features (baseline is 0.7987)
- Change in rank of Abacavir → ocular cancer (expected: drops significantly)
- Change in rank of Losartan → atherosclerosis (expected: rises)
- Feature importance of MoA features vs embedding features

**Done when:** `results/moa_benchmark/` contains a result JSON with PR-AUC for
each model, and the delta vs baseline is computed and written to `PAPER.md` §7.3.

---

### 1.2 CpD Relation Run

**Why:** The paper claims multi-relational capability. CpD (Compound-palliates-
Disease, 390 positive edges) is the natural first extension and requires zero code
changes. This result is needed for the multi-relational section of the paper.

**Command:**

```bash
python scripts/run_optimized_pipeline.py \
  --relation CpD --full_graph_embeddings \
  --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
  --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24 \
  --run_ensemble --ensemble_method stacking --tune_classical --fast_mode \
  --results_dir results/cpd_run
```

**What to record:**
- PR-AUC for each model (expect lower absolute numbers than CtD — fewer positives)
- Whether the Pauli inversion effect holds on CpD (QSVC standalone < classical,
  ensemble > classical)
- Top novel CpD predictions not in training set

**Done when:** `results/cpd_run/` contains a result JSON and Table 3 in
`docs/paper.tex` has a CpD section added.

---

### 1.3 Multi-Seed Evaluation (5 Seeds)

**Why:** All current results use `--random_state 42` exclusively. A single seed
could be lucky or unlucky. No variance estimate exists for any headline number
in the paper. Reviewers will ask for this.

**Command:**

```bash
for seed in 42 7 13 99 2026; do
  python scripts/run_optimized_pipeline.py \
    --relation CtD --full_graph_embeddings \
    --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 \
    --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli \
    --qml_feature_map_reps 2 --qsvc_C 0.1 --qml_pre_pca_dim 24 \
    --run_ensemble --ensemble_method stacking --tune_classical --fast_mode \
    --random_state $seed \
    --results_dir results/multiseed/seed_$seed
done
```

**Note on time:** Each seed runs a full QSVC kernel (~43 min on CPU). Use
`--qsvc_nystrom_m 200` if time is a constraint and document it explicitly.
5 seeds with Nyström ≈ 5 × 2 min = 10 min total.

**What to compute from output:**
- Mean ± std PR-AUC for RF, ET, QSVC, and ensemble across 5 seeds
- Confirm 0.7987 is not a lucky outlier
- Update Table 3 in `docs/paper.tex` with error bars: `0.7987 ± σ`

**Done when:** 5 result JSONs exist, mean ± std is computed, and Table 3 is updated.

---

### 1.4 Fix LaTeX Citation Keys

**Why:** The compiled `paper.pdf` currently shows `[?]` for every in-text
citation. The `\cite{key}` labels in the body do not match the `\bibitem{key}`
labels in the bibliography block. The paper cannot be submitted in this state.

**What to do:**

Open `docs/paper.tex`. The body uses these keys:

```
himmelstein2017, mayers2023, qkdti2025, kruger2023, graphrag2025,
cim2026, hybrid2024, rotate2019, pykeen2021, schuld2021, qiskit2023
```

Find `\begin{thebibliography}` (around line 578) and ensure every
`\bibitem{key}` matches exactly one `\cite{key}` in the body — spelling,
capitalization, and year must match exactly.

After fixing, compile:

```bash
cd docs
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
bibtex paper          # if using .bib file
pdflatex -interaction=nonstopmode paper.tex
```

**Done when:** `paper.pdf` compiles with zero `[?]` citations and zero undefined
reference warnings.

---

### 1.5 Render and Commit the 3 Required Figures

**Why:** `paper.tex` references three figures as PDFs in `figures/`. The
`figures/` directory is currently untracked. Without these files, the paper
will not compile cleanly and the PDF will have missing figure boxes.

---

#### Figure 1 — Pipeline Architecture

**File:** `figures/fig1_pipeline.pdf`  
**Layout:** Two-column flowchart.  
- Left column: Hetionet → KG embeddings → compound-disease pairs → feature vector
- Right column: classical path (blue) branching from quantum path (purple),
  merging at stacking ensemble (green)
- Arrows flow top to bottom; label each box with the module name and file

**Tool:** Generate with matplotlib, graphviz, or draw.io. Export as PDF.

---

#### Figure 2 — Pauli vs ZZ Feature Map Tradeoff

**File:** `figures/fig2_pauli_zz.pdf`  
**Data (exact values to use):**

| Config | QSVC PR-AUC | Ensemble PR-AUC | RF Baseline |
|--------|------------|----------------|-------------|
| ZZ featuremap | 0.7216 | 0.7408 | 0.7838 |
| Pauli featuremap | 0.6343 | 0.7987 | 0.7838 |

**Layout:** Grouped bar chart. Horizontal dashed line at 0.7838 (RF baseline).
X-axis: ZZ / Pauli. Bars per group: QSVC (purple), Ensemble (green).
Annotate the Pauli group: "QSVC < baseline, Ensemble > baseline".

---

#### Figure 3 — Score-Validity Scatter

**File:** `figures/fig3_clinical.pdf`  
**Data:** 6 manually validated (compound, disease) pairs with model score
and ClinicalTrials.gov trial count. These 6 pairs are specified in
`PAPER.md` Appendix A (the score-validity inversion scatter spec).  
**Layout:** Scatter plot. X-axis: model score. Y-axis: trial count.
Diagonal arrow annotated "score-validity inversion". Abacavir → ocular
cancer should appear high score / zero trials (top left of arrow).

**Script to generate Figure 3 data:** `scripts/compute_kta_zz_vs_pauli_subset.py`
can be adapted; or write `figures/fig3_clinical.py` directly from the
six known data points.

**Done when:** All three PDFs exist in `figures/`, `paper.tex` compiles
without missing figure errors, and figures are committed to git.

---

### 1.6 Add Missing Baselines to Table 3

**Why:** Without a random baseline (expected PR-AUC ≈ 0.50) and a
degree-heuristic baseline, the paper cannot establish whether 0.7987 is
meaningful. Reviewers will immediately ask for these.

**Random baseline:** Add as a row labeled `Random (expected)` with PR-AUC = 0.50.
No code needed — this is a constant for a balanced test set.

**Degree-heuristic baseline:** Rank all (compound, disease) test pairs by
the product of compound degree × disease degree in the training graph.

```python
# Run this after loading the pipeline's training graph
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score

# compound_degree[c] = number of training edges involving compound c
# disease_degree[d] = number of training edges involving disease d
scores = [compound_degree[c] * disease_degree[d] for c, d in test_pairs]
degree_pr_auc = average_precision_score(test_labels, scores)
print(f"Degree-heuristic PR-AUC: {degree_pr_auc:.4f}")
```

Add the computed value as a row labeled `Degree-heuristic` in Table 3 of
both `PAPER.md` and `docs/paper.tex`.

**Done when:** Table 3 has `Random (expected)` and `Degree-heuristic` rows,
with the degree-heuristic value computed from the actual test split.

---

## Tier 2 — Strengthens the Quantum Claim

These items do not block submission but will be explicitly challenged by
reviewers at any quantum-ML venue. Complete before submitting to
*Quantum Machine Intelligence*.

---

### 2.1 Fair Quantum vs Classical Ablation (A/B/C/D)

**Why:** The current comparison is QSVC on 16-dim PCA features vs RF on
512-dim full features. This gives classical models ~32× more information.
The only scientifically fair comparison is B vs C (both receive exactly
16-dimensional input).

**Four conditions:**

| Condition | Description | Flags |
|-----------|-------------|-------|
| A | Classical, full 512-dim | `--classical_only` |
| B | Classical, restricted 16-dim | `--classical_only --restrict_classical_to_qml_dim --qml_dim 16` |
| C | Quantum only, 16-dim | `--quantum_only --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2` |
| D | Stacking ensemble (A + C) | `--run_ensemble --ensemble_method stacking` |

Use `./scripts/run_campaign3_ablation.sh` — dry-run by default, pass
`EXECUTE=1` to run. Add `RUN_FAST_MODE=1` for faster iteration.

**Key result needed:** B vs C PR-AUC comparison. If QSVC (C) outperforms
classical SVM on 16-dim (B), that is the paper's strongest quantum finding.

**Done when:** Four result directories exist, a table of (Condition, Model,
Input Dim, PR-AUC) is added to paper §6.3, and the B vs C delta is stated
explicitly in the text.

---

### 2.2 5-Fold Cross-Validation for Quantum + Ensemble

**Why:** The existing 5-fold CV run (Run 2, 2026-03-23) skipped QSVC and
ensemble because full kernel computation per fold takes 43 minutes. No
honest variance estimate exists for the quantum result.

**Solution — Nyström approximation for CV:**

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2 \
  --qsvc_C 0.1 --qml_pre_pca_dim 24 \
  --qsvc_nystrom_m 200 \
  --run_ensemble --ensemble_method stacking --tune_classical \
  --use_cv_evaluation --cv_folds 5 \
  --results_dir results/cv_quantum
```

Use `./scripts/run_cv_feasibility_smoke.sh` for dry-run. `EXECUTE=1` to run.
`--qsvc_nystrom_m 200` reduces per-fold time from 43 min to ~2 min.

**What to report:** RF, ET, QSVC (Nyström), Ensemble — mean ± std PR-AUC
across 5 folds. Document that Nyström was used.

**Done when:** CV result JSON exists with per-fold scores, mean ± std is
computed, and §6.1 of the paper is updated.

---

### 2.3 Nyström Accuracy vs Speed Sweep

**Why:** The `--qsvc_nystrom_m` flag exists but no systematic sweep of
`m` values has been run. Without this, the choice of m=200 (used in CV)
is arbitrary and cannot be justified in the paper.

**Command (or use `./scripts/run_nystrom_sweep.sh` with `EXECUTE=1`):**

```bash
for m in 50 100 200 400 800; do
  python scripts/run_optimized_pipeline.py --relation CtD \
    --full_graph_embeddings --embedding_method RotatE \
    --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
    --quantum_only --qml_dim 16 --qml_feature_map Pauli \
    --qml_feature_map_reps 2 --qsvc_C 0.1 \
    --qsvc_nystrom_m $m \
    --results_dir results/nystrom_sweep/m$m
done
```

Full kernel reference: m=None → PR-AUC=0.6343, time=2,619s.

**Done when:** Table of (m, PR-AUC, kernel_time) exists, the accuracy/speed
knee is identified, and the chosen m for all subsequent Nyström runs is
documented and justified in the paper.

---

### 2.4 Noisy Simulator Benchmark

**Why:** All primary results use noiseless statevector simulation. Quantum
advantage claims from noiseless simulation may not hold on real hardware.
The noisy simulator result fills the gap between ideal and Heron.

**Command (or use `./scripts/run_noisy_sim_smoke.sh` with `EXECUTE=1`):**

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --quantum_only --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --quantum_config_path config/quantum_config_noisy.yaml \
  --results_dir results/noisy_sim
```

**Expected outcome:** PR-AUC lower than 0.6343 (ideal). Document the
degradation as evidence of noise sensitivity. This motivates the error
mitigation work in §2.5.

**Done when:** `results/noisy_sim/` exists, noisy PR-AUC is reported, and
the delta vs ideal is added to paper §6.5.

---

### 2.5 Noise Mitigation Ablation

**Why:** `quantum_layer/quantum_error_mitigation.py` and
`quantum_layer/advanced_error_mitigation.py` exist but have never been
evaluated against any result. Without this, noise mitigation is a code
artifact with no reported effect.

**Four-condition ablation:**

1. Noisy sim, no mitigation → baseline (from §2.4)
2. Noisy sim + ZNE (zero-noise extrapolation)
3. Noisy sim + readout mitigation
4. Noisy sim + ZNE + readout mitigation

**Expected result:** Each technique should recover 2–5 pp of the ideal
PR-AUC. Add a noise mitigation ablation table to paper §6.5.

**Done when:** Four noisy result JSONs exist and a mitigation table is
added to the paper.

---

### 2.6 Kernel-Target Alignment (KTA) Analysis

**Why:** The paper proposes KTA as a diagnostic for feature map selection
(§6.3). `quantum_layer/quantum_kernel_alignment.py` exists but has never
been used. KTA pre-screens take ~2 seconds vs 43 minutes for a full kernel.

**Step 1 — Export training subset:**

```bash
python scripts/export_kta_xy_npz.py --relation CtD --max_entities 200 \
  --embedding_dim 64 --qml_dim 16 --subset 100 \
  --out results/kta_train_subset.npz
```

**Step 2 — Compare ZZ vs Pauli KTA:**

```bash
python scripts/compute_kta_zz_vs_pauli_subset.py \
  --npz results/kta_train_subset.npz --qml_dim 16 --n_samples 100
```

**What to report:** KTA scores for ZZ and Pauli. Higher KTA predicts better
SVM performance. Publish values alongside the Feature Map Ablation table
(`\label{tab:ablation}` in `docs/paper.tex`).

**Done when:** KTA values for ZZ and Pauli are computed and appear in the
Feature Map Ablation table in the paper.

---

### 2.7 Score-Validity Inversion — Quantitative Analysis

**Why:** The score-validity inversion is described qualitatively in §7.3
(Abacavir scores high with zero clinical support). No Spearman correlation
between model score and trial count has been computed across the full test
set.

**What to compute:**

```python
import scipy.stats as stats
rho, p = stats.spearmanr(scores, trial_counts)
print(f"Spearman ρ = {rho:.3f}, p = {p:.4f}")
```

Use `scripts/compute_score_validity_inversion_metrics.py`:

```bash
python scripts/compute_score_validity_inversion_metrics.py \
  --fig3-published --top-k 6
```

A negative Spearman ρ confirms the inversion. Report in §7.3 alongside
Figure 3.

**Done when:** Spearman ρ and p-value are computed and stated in §7.3.

---

## Tier 3 — Frontend Parity with Streamlit

The Next.js frontend (`frontend/`) is ~40% complete. Streamlit is still
the daily driver. These items are required before Streamlit can be retired.

---

### 3.1 MoA Explanation Panel in `/predict`

**Why:** This is the highest-value frontend feature. When a user scores a
(compound, disease) pair, there is no breakdown of why the model scored it
that way.

**What to build:**
- After scoring, show the 10 MoA feature values (binding targets, shared
  targets, pathway genes, etc.)
- Feature importance bar chart from RF/ET
- Mechanistic support indicator: green if `shared_targets > 0`, amber if
  `similar_compounds_treat > 0`, red if all zero

**API needed:** Add `GET /predict/explain` to `middleware/api.py` that
returns MoA feature values alongside the score. The MoA feature vector is
already computed in `kg_layer/moa_features.py` — expose it.

**Done when:** `/predict` shows the explanation panel with real MoA values
for any scored pair.

---

### 3.2 Pipeline Job Trigger UI at `/simulation/new`

**Why:** The Streamlit dashboard has a form to launch pipeline runs. The
Next.js `/simulation` page shows existing jobs but has no way to start a
new one. This is required before Streamlit can be retired.

**What to build:**
- Form at `/simulation/new` exposing all pipeline flags as inputs
  (embedding method, dim, epochs, relation, quantum settings)
- `POST /jobs` endpoint in `middleware/api.py` (verify whether it exists;
  if not, implement it as a subprocess-launch endpoint)
- Job status polling with auto-refresh on `/simulation`
- Link from each job in the list to a result detail view

**Done when:** A user can launch a full pipeline run from the Next.js UI,
see it appear in the job list, and click through to its results.

---

### 3.3 Benchmark Registry / Run Comparison UI at `/experiments/compare`

**Why:** `results/benchmark_registry.jsonl` tracks all pipeline runs, but
there is no UI to browse or compare them without pandas.

**What to build:**
- Table of all registered runs with columns: timestamp, relation, embedding
  method/dim, QSVC config, PR-AUC, ROC-AUC
- Diff view: select two runs, highlight config differences and metric deltas
- Filter by relation, embedding method, and execution mode

**API:** Extend `GET /experiments` to return all registry entries (currently
returns only the latest run).

**Done when:** `/experiments/compare` shows the full run history and the
diff view works.

---

### 3.4 Experiment History Chart at `/experiments`

**What to build:**
- Line chart showing PR-AUC over time across all runs
- Separate series for classical-only, QSVC-only, and ensemble
- Tooltip with config details on hover

**Library:** Add `recharts` to `frontend/package.json`. The project already
has Tailwind CSS; no other UI framework change is needed.

**Done when:** `/experiments` shows a time-series chart with real data from
the benchmark registry.

---

### 3.5 Complete or Remove `/quantum` Placeholder Sections

**Current state:** `/quantum` renders "coming soon" placeholder cards for
most of its content.

**Options:**
- **Implement:** Add circuit depth display, feature map visualization (Qiskit
  SVG), backend status display, and execution mode toggle
- **Remove from nav:** If not implementing now, remove the placeholder entries
  from the sidebar so users do not encounter dead ends

**Done when:** Either the page has real content or it is removed from nav.

---

### 3.6 Audit `/analysis/drug-delivery` and `/analysis/next-steps`

**What to do:** Read both files. If they are stubs, either implement them
or remove them from the sidebar navigation. Placeholder routes visible in
the sidebar erode user trust.

**Done when:** Both routes either have real content or are removed from nav.

---

### 3.7 Fix `/visualization` Live API Gaps

**Known risks:**
- Entity lookup may fail for names not in the embedding index (no error
  handling for missing embeddings)
- KG browser requires `GET /kg/graph` — verify it exists in `middleware/api.py`
- Pair scoring duplicates `/predict` — separate concerns or deduplicate

**What to do:**
1. Audit `GET /kg/graph` and `GET /kg/entity/{name}` in `middleware/api.py`
2. Add embedding-not-found error state with helpful suggestions
3. Add entity autocomplete from the known compound/disease lists

**Done when:** `/visualization` works against a live API with real data and
handles missing-entity errors gracefully.

---

### 3.8 Resolve `/molecular-design` Duplication

**Current state:** `/molecular-design` wraps the same `PredictForm` component
as `/predict`. It is a duplicate route.

**Decision:** Either differentiate it (structure-based input vs name-based
input) or remove it and redirect to `/predict`.

**Done when:** The route either has distinct functionality or is removed.

---

## Tier 4 — Platform Extensions

These are longer-term items that convert this from a research proof-of-concept
into a production-grade drug repurposing platform. Complete after arXiv
submission.

---

### 4.1 ClinicalTrials.gov Live Query Integration

**What to build:** A module (`kg_layer/clinical_trials_lookup.py`) that
automatically annotates top predictions with trial count, phase, and status
by querying the ClinicalTrials.gov API v2.

**Integration points:**
- Call after scoring top-N predictions in `run_optimized_pipeline.py`
- Expose via `GET /predict/validate` in `middleware/api.py`
- Display in the prediction results panel in the Next.js UI
- Add `trial_count` field to benchmark registry JSON per prediction

Cache results to avoid repeated API calls for the same pairs.

**Done when:** A prediction result includes a `trials` field automatically
populated from ClinicalTrials.gov for the top 25 candidates.

---

### 4.2 Multi-Relational Joint Training

**Three options, in increasing complexity:**

- **Option A (no code changes):** Run CtD, CpD, and DrD experiments with
  shared full-graph embeddings. Train separate classifiers per relation.
  Start here — document results in the paper.

- **Option B (multi-task stacking):** Use CpD and DrD predictions as
  additional input features for the CtD classifier. Requires a new
  multi-task training loop.

- **Option C (relation-aware quantum kernel):** Encode the relation type
  as part of the quantum feature map input — a 2-bit one-hot appended to
  the compound-disease embedding pair. Relevant file:
  `quantum_layer/quantum_enhanced_embeddings.py`.

**Start with Option A. Document Options B and C as research directions.**

---

### 4.3 GNN Baselines

**Why:** The paper compares against RotatE, ComplEx, and classical ML but
not against GNN-based link predictors. A GNN baseline is now expected by
reviewers at quantum-ML and bioinformatics venues.

**What to implement:**
- **R-GCN** (Relational Graph Convolutional Network): supports Hetionet's
  24 heterogeneous relation types natively
- **CompGCN**: composition-based message passing for multi-relational graphs

Both are available through PyKEEN. Use the same training interface as
RotatE. `kg_layer/gnn_baselines.py` already exists — check its state.

**Done when:** GNN PR-AUC results appear in a new Table 2 before the
primary Table 3.

---

### 4.4 DRKG Extension

**What DRKG is:** Drug Repurposing Knowledge Graph — 4.4M edges, 97K
entities across 107 relation types (integrates DrugBank, STRING, Hetionet,
and others). More current than Hetionet (last updated 2017).

**What the extension requires:**
1. `DRKGLoader` sibling class to `HetionetLoader` in `kg_layer/kg_loader.py`
2. GPU embedding training on DGX (use `./scripts/run_full_embedding_dgx.sh`
   as a template for `run_drkg_embedding.sh`)
3. Nyström is mandatory (Nyström m=400–800 from §2.3 sweep)
4. DRKG-compatible feature constructor in `kg_layer/enhanced_features.py`

**Done when:** A full CtD experiment on DRKG embeddings produces a PR-AUC
comparable to or better than the Hetionet result.

---

### 4.5 VQC Architecture Improvement

**Current state:** VQC results are near-random (~0.54–0.55 PR-AUC). The
minimal search used 50 iterations and reps=3–4.

**Three things to try:**

1. **Increase iteration budget to 200** (`--qml_max_iter 200`). Use
   `./scripts/run_vqc_scaling_smoke.sh` with `EXECUTE=1`.

2. **Layerwise training:** Train reps=2 to convergence, then add layers.
   This avoids barren plateaus (vanishing gradients at initialization).

3. **Warm-start from QSVC weights:** Use support vectors from the trained
   QSVC to initialize VQC parameter values.

**Relevant files:** `quantum_layer/quantum_variational_feature_selection.py`,
`quantum_layer/quantum_circuit_optimization.py`.

**Done when:** VQC achieves PR-AUC > 0.60 on CtD, or all three approaches
are documented as having failed to lift it above random.

---

### 4.6 Variational Quantum Kernel Learning (VQKL)

**What this is:** Parameterize the encoding circuit `U(x; θ)` and jointly
train `θ` to maximize kernel-target alignment while training the SVM.
The result is a kernel whose geometry is tuned to the biomedical link
prediction task's label structure.

**Building blocks already in repo:**
- `quantum_layer/quantum_kernel_engineering.py` — `AdaptiveQuantumKernel`
- `quantum_layer/quantum_kernel_alignment.py` — `kernel_target_alignment`
- Pipeline flags: `--optimize_feature_map_reps`, `--use_kernel_alignment`

Full VQKL (joint θ + SVM) is future work on top of these primitives.
Implement only after the KTA diagnostic (§2.6) is producing clean results.

---

### 4.7 IBM Quantum Heron Full Benchmark

**Current state:** A Heron hardware run exists confirming QSVC standalone
PR-AUC ~0.634, consistent with the simulator. Missing: ensemble/stacking
on hardware, and a scripted ZNE readout write-up.

**Minimum viable Heron benchmark:**

```bash
python scripts/train_on_heron.py \
  --relation CtD --max_entities 200 \
  --embedding_dim 128 --qubits 8 \
  --model_type QSVC --feature_map Pauli --feature_map_reps 1 \
  --backend ibm_torino \
  --negative_sampling degree_corrupt \
  --results_dir results/heron
```

Use 8 qubits (not 16) on hardware to limit circuit depth and decoherence.
Report PR-AUC with and without ZNE readout mitigation.

---

### 4.8 Inference API Robustness

**Problems to fix:**

1. **Missing entity handling:** If a compound or disease is not in the
   embedding index, the API silently returns a meaningless score. Add an
   explicit check and return a helpful error with entity suggestions.

2. **Batch endpoint:** Add `POST /predict/batch` accepting a list of
   (compound, disease) pairs. Enable full-graph candidate ranking of
   all ~212K pairs in a single call.

3. **Feature cache:** Pre-compute and cache graph topology features at
   startup so per-prediction latency is O(1) instead of O(graph traversal).

---

## Tier 5 — arXiv Submission Checklist

Complete all Tier 1 items before starting this checklist. Complete Tier 2
before submitting to *Quantum Machine Intelligence*.

```
[ ] 1.1 MoA benchmark result in results/ and Table 3 updated
[ ] 1.2 CpD result in results/ and Table 3 updated
[ ] 1.3 5-seed results computed, mean ± std in Table 3
[x] 1.4 paper.tex compiles with zero [?] citations
[x] 1.5 figures/fig1_pipeline.pdf committed
[x] 1.5 figures/fig2_pauli_zz.pdf committed
[x] 1.5 figures/fig3_clinical.pdf committed
[x] 1.5 pdflatex passes with no missing figure warnings
[x] 1.6 Degree-heuristic baseline computed and in Table 3
[x] 1.6 Random baseline row (PR-AUC = 0.50) added to Table 3
[ ] 2.1 Ablation A/B/C/D results in results/ablation_*/
[ ] 2.1 B vs C delta stated explicitly in paper §6.3
[ ] 2.3 Nyström sweep complete, m chosen and justified
[ ] 2.4 Noisy simulator result in results/noisy_sim/
[ ] 2.6 KTA values for ZZ and Pauli in Feature Map Ablation table
[ ] 2.7 Spearman ρ computed and stated in §7.3
[ ] paper.tex: §8.5 Limitations updated to remove resolved items
[ ] OSF bundle prepared (scripts/prepare_osf_bundle.sh)
[ ] arXiv categories: quant-ph, cs.LG, q-bio.QM
[ ] Submit
```

---

## Dependency Map

```
Tier 1 (paper experiments)
    ├── 1.1 MoA run          → feeds Table 3 §7.3
    ├── 1.2 CpD run          → feeds Table 3 multi-relational section
    ├── 1.3 Multi-seed       → feeds Table 3 error bars; depends on Nyström (optional)
    ├── 1.4 Citation fix     → no dependencies
    ├── 1.5 Figures          → depends on 1.3 (for final Pauli/ZZ numbers)
    └── 1.6 Baselines        → depends on having the test split loaded

Tier 2 (quantum claim)
    ├── 2.1 Ablation A/B/C/D → depends on pipeline running cleanly
    ├── 2.2 CV quantum       → depends on 2.3 (Nyström sweep) to choose m
    ├── 2.3 Nyström sweep    → no dependencies; should run first in Tier 2
    ├── 2.4 Noisy sim        → no dependencies
    ├── 2.5 Mitigation       → depends on 2.4
    ├── 2.6 KTA analysis     → no dependencies
    └── 2.7 Spearman ρ       → depends on top-N predictions being available

Tier 3 (frontend)
    ├── 3.1 MoA panel        → depends on GET /predict/explain API
    ├── 3.2 Job trigger      → depends on POST /jobs API
    ├── 3.3 Registry UI      → depends on GET /experiments returning full history
    └── 3.4 History chart    → depends on registry UI data

Tier 4 (platform)
    ├── 4.1 ClinicalTrials   → can start anytime
    ├── 4.2 Multi-relational → depends on 1.2 (CpD) being done as a baseline
    ├── 4.3 GNN baselines    → depends on PyKEEN being configured
    ├── 4.4 DRKG             → depends on 2.3 (Nyström) for feasibility
    ├── 4.5 VQC improvement  → independent
    ├── 4.6 VQKL             → depends on 2.6 (KTA) being clean
    ├── 4.7 Heron benchmark  → requires IBM Quantum account + backend access
    └── 4.8 API robustness   → independent; do alongside frontend work
```

---

## Quick Start — Next Three Actions

If starting fresh from this document, do these three things first:

1. **Start 1.3 (multi-seed) running in the background** on the DGX or
   overnight on CPU. This is the longest compute item and blocks Table 3.

2. **Fix 1.4 (citation keys)** — 30 minutes in a text editor, no compute,
   immediately unblocks a compilable PDF.

3. **Run 1.1 (MoA benchmark)** while 1.3 is running — 1 hour, produces a
   concrete new result for the paper.

Everything else follows after those three complete.
