# Scientific Gaps — Methodological Completeness

**Status:** Partially addressed  
**Blocks:** Publication credibility and quantum advantage claim

These are gaps in the experimental design that weaken the scientific
conclusions, independent of code completeness.

---

## 1. Ablation Matrix — Fair Quantum vs Classical Comparison

**Current state:** The reported comparison is QSVC on 16-dim PCA features vs
RF/ET on 512-dim full features. This is not a fair quantum comparison because
the classical models have ~32× more input information.

**What the runbook defines (Campaign 3):**

| Condition | Description | Flags |
|-----------|-------------|-------|
| A | Classical models, full 512-dim features | `--classical_only` |
| B | Classical models, restricted to 16-dim (matches quantum) | `--classical_only --restrict_classical_to_qml_dim --qml_dim 16` |
| C | Quantum only, 16-dim | `--quantum_only --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2` |
| D | Stacking ensemble of A + C | `--run_ensemble --ensemble_method stacking` |

**The critical comparison is B vs C:** both receive exactly 16-dimensional
input. If QSVC (C) outperforms a classical SVM on 16-dim (B), that is a
meaningful quantum result.

**Commands:**

```bash
# Condition A — full classical (512-dim)
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --classical_only --results_dir results/ablation_A

# Condition B — classical on 16D
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --classical_only --restrict_classical_to_qml_dim --qml_dim 16 \
  --results_dir results/ablation_B

# Condition C — quantum only
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --quantum_only --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --results_dir results/ablation_C

# Condition D — stacking ensemble
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2 \
  --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
  --results_dir results/ablation_D
```

**Scaffolding (discovery / dry-run / guarded exec):**

- `scripts/run_campaign3_ablation.sh` prints the same four conditions safely by default (`EXECUTE=0`). Examples: `./scripts/run_campaign3_ablation.sh B C`; `EXECUTE=1 RUN_FAST_MODE=1 ./scripts/run_campaign3_ablation.sh B`. Full parity with the table keeps RotatE 128 × 200 epochs (still hours-class per condition even with `--fast_mode`).
- **Iteration:** There is no repository-wide `CHEAP_MODE`; use `RUN_FAST_MODE=1` (forwards `--fast_mode` into `run_optimized_pipeline.py`) on wrappers in §§1–3 for quicker local iterations.

**Expected result table (to be added to paper §6):**

| Condition | Model | Input Dim | PR-AUC |
|-----------|-------|-----------|--------|
| A | RF-Optimized | 512 | TBD |
| B | RF-Optimized | 16 | TBD |
| C | QSVC-Pauli | 16 | ~0.634 |
| D | Stacking (A+C) | 512+16 | ~0.799 |

---

## 2. 5-Fold CV for Quantum + Ensemble Models

**Current state:** The 5-fold CV run (Run 2, 2026-03-23) only evaluated
classical models. QSVC and ensemble were skipped because full kernel computation
is 43 minutes per fold × 5 folds = ~3.5 hours.

**Practical solution — Nyström approximation for CV:**

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

**Feasibility wrapper:** `scripts/run_cv_feasibility_smoke.sh` echoes the invocation above plus `--quantum_config_path config/quantum_config_ideal.yaml` by default for local/statevector stacks. Same ergonomics as Campaign 3: dry-run unless `EXECUTE=1`; add `RUN_FAST_MODE=1` for iteration; override `RESULTS_DIR` as needed.

`scripts/run_optimized_pipeline.py` already supports `--use_cv_evaluation`, `--cv_folds`, and `--qsvc_nystrom_m` (confirmed in argparse / CV branches).

`--qsvc_nystrom_m 200` reduces per-fold kernel computation from 43 minutes to
approximately 2 minutes, making 5-fold CV feasible. Document that Nyström was
used and report the approximation quality separately.

**What to report:**
- RF, ET, QSVC (Nyström), Ensemble: mean ± std PR-AUC across 5 folds
- This is the first honest quantification of variance in the quantum result

---

## 3. Noisy Simulator Benchmark

**Current state:** All primary results use noiseless statevector simulation.
The benchmark spec defines three tiers (ideal / noisy / hardware) that must
not be mixed. No noisy simulation result exists.

**Why it matters:** Quantum advantage claims made from noiseless simulation
may not transfer to real hardware. The noisy simulator fills the gap between
simulation and Heron.

**Config:** `config/quantum_config_noisy.yaml` exists with depolarizing noise
model, ZNE, and readout mitigation.

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --quantum_only --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --quantum_config_path config/quantum_config_noisy.yaml \
  --results_dir results/noisy_sim
```

**Expected outcome:** PR-AUC will be lower than the ideal-simulator result
(0.6343). Document the degradation as evidence of noise sensitivity.

**Feasibility wrapper:** `scripts/run_noisy_sim_smoke.sh` mirrors §§1–2 ergonomics: dry-run by default (`EXECUTE=0`); pass `EXECUTE=1` to run with `--quantum_config_path config/quantum_config_noisy.yaml`. Optional `RUN_FAST_MODE=1` (same semantics as Campaign 3 / CV wrappers).

---

## 4. Hardware Heron Full Benchmark

**Current state:** A Heron hardware run exists that confirms QSVC standalone
PR-AUC ~0.634, consistent with the simulator. **Still missing for a “full” hardware story:**
- No ensemble/stacking benchmark on IBM hardware (QSVC+VQC path only in `scripts/train_on_heron.py`).
- No dedicated noise-characterization or ZNE readout write-up beyond high-level consistency checks.

**What exists today:** `scripts/train_on_heron.py` is a QSVC/VQC hardware CLI with `--dry_run`, token/back-end pre-flight, and cost awareness (see `--help`). It does **not** replace `run_optimized_pipeline.py` ensembles.

**Residual gap (“stub” relative to paper parity):** **Ensemble-on-Heron and scripted ZNE reporting** tied to the main stacking pipeline remain **manual / future work**; for reproducible mitigation evidence, prioritize noisy simulator tiers (§3).

**Minimum viable Heron benchmark:**

```bash
python scripts/train_on_heron.py \
  --relation CtD \
  --max_entities 200 \
  --embedding_dim 128 \
  --qubits 8 \
  --model_type QSVC \
  --feature_map Pauli \
  --feature_map_reps 1 \
  --backend ibm_torino \
  --negative_sampling degree_corrupt \
  --results_dir results/heron
```

Use 8 qubits (not 16) on hardware to limit circuit depth and decoherence.
Report hardware PR-AUC with and without ZNE readout mitigation.

---

## 5. Score-Validity Inversion — Quantitative Analysis

**Current state:** The score-validity inversion is described qualitatively
(Abacavir scores high, Losartan scores low despite more clinical support).
No quantitative analysis of the inversion rate across all test predictions
has been done.

**What to compute:**

```python
# Load top-N novel predictions (not in training CtD)
# For each: (a) model score, (b) ClinicalTrials.gov trial count (manual or API)
# Compute: Spearman correlation between model score and trial count
# Compute: fraction of top-10 predictions that have ≥1 registered trial
# Compute: fraction of top-10 predictions that have 0 trials

import scipy.stats as stats
rho, p = stats.spearmanr(scores, trial_counts)
print(f"Spearman ρ = {rho:.3f}, p = {p:.4f}")
```

A negative Spearman ρ confirms the inversion (statistical power scales with \(N\);
the illustrative scatter uses six manually validated points). Report this value alongside
the scatter plot (**Figure 3** / `figures/fig3_clinical.py`) and the clinical table
(**`tab:clinical` in `docs/paper.tex`**; analogous block in **`docs/paper_qGG_full_2026.tex`**).

**CLI bridge (curated Fig. 3 pairs or your own CSV):**

```bash
# Reproduces ρ on the six panel points synced with RESULTS_EVIDENCE / fig3_clinical.py
python scripts/compute_score_validity_inversion_metrics.py --fig3-published --top-k 6

# Paper-grade row list: CSV with columns score_col / trial_col (e.g. join pipeline export + CT.gov)
python scripts/compute_score_validity_inversion_metrics.py \
  --scores-csv results/clinical_scores_trials.csv --score-col ensemble_score --trial-col trial_count
```

---

## 6. Kernel-Target Alignment (KTA) Analysis

**Current state:** `quantum_layer/quantum_kernel_alignment.py` exists but has
not been used to evaluate or select feature maps. The current workflow runs
full 43-minute kernels to compare ZZ vs Pauli.

**What KTA enables:** Pre-screening feature map candidates in seconds before
committing to a full kernel computation. KTA measures how well the kernel
matrix aligns with the label gram matrix: a higher KTA score predicts better
SVM performance.

**What to run:**

1. **Binary `X`,`y` for a subset:** `scripts/export_kta_xy_npz.py` writes `np.savez_compressed` with keys `X` (n × `qml_dim`) and `y` ({0,1}) from the lightweight path `HetionetEmbedder.reduce_to_qml_dim` + `prepare_link_features_qml` (same family as `scripts/train_on_heron.py` / `scripts/e2e_smoke.py`). For paper-grade tensors that mirror the main pipeline’s `AdvancedQMLFeatureEngineer` / `--qml_pre_pca_dim` stack, reconcile or re-export after that path is surfaced; this exporter targets **seconds-scale KTA pre-screens**.
2. **ZZ vs Pauli KTA CLI:** `scripts/compute_kta_zz_vs_pauli_subset.py` consumes `--npz` or synthetic data.

```bash
python scripts/export_kta_xy_npz.py --relation CtD --max_entities 200 \
  --embedding_dim 64 --qml_dim 16 --subset 100 \
  --out results/kta_train_subset.npz
python scripts/compute_kta_zz_vs_pauli_subset.py \
  --npz results/kta_train_subset.npz --qml_dim 16 --n_samples 100
```

Equivalent call in-process (for notebooks):

```python
from quantum_layer.quantum_kernel_alignment import kernel_target_alignment

kta_zz    = kernel_target_alignment(K_zz, y_train)
kta_pauli = kernel_target_alignment(K_pauli, y_train)
print(f"ZZ KTA: {kta_zz:.4f}")
print(f"Pauli KTA: {kta_pauli:.4f}")
```

Publish KTA values alongside the Feature Map Ablation table (`\label{tab:ablation}` in `docs/paper.tex`; same role in `docs/paper_qGG_full_2026.tex` under subsection *Feature Map Analysis: The Pauli Inversion Effect*, `sec:results_feature_map`) rather than burying them only in run logs.

---

## Summary — what each gap fixes

| Gap | Paper section affected | Scientific impact |
|-----|----------------------|-------------------|
| Ablation matrix (A/B/C/D) | §6.3 | Makes quantum comparison fair |
| CV for quantum + ensemble | §6.1 | Adds variance estimate to 0.7987 |
| Noisy simulator benchmark | §6.5 | Fills ideal→hardware gap |
| Heron full benchmark | §6.5 | Real hardware validation |
| Score-validity inversion quantification | §7.3 | Adds Spearman ρ to Figure 3 |
| KTA analysis | §6.3 | Explains why Pauli outperforms ZZ in ensemble |
