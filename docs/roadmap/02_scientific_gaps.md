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

---

## 4. Hardware Heron Full Benchmark

**Current state:** A Heron hardware run exists that confirms QSVC standalone
PR-AUC ~0.634, consistent with the simulator. However:
- No ensemble has been run on hardware
- No noise characterization or ZNE mitigation has been applied and reported
- `scripts/train_on_heron.py` is still incomplete (see `docs/CHANGES_NEEDED.md` §2)

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

A negative Spearman ρ confirms the inversion. Report this value alongside
the scatter plot (Figure 3) in the paper.

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

```python
from quantum_layer.quantum_kernel_alignment import kernel_target_alignment

# Compute KTA for ZZ and Pauli on a 100-sample subset
# If Pauli KTA > ZZ KTA, Pauli will likely outperform ZZ in ensemble context
kta_zz    = kernel_target_alignment(K_zz, y_train)
kta_pauli = kernel_target_alignment(K_pauli, y_train)
print(f"ZZ KTA: {kta_zz:.4f}")
print(f"Pauli KTA: {kta_pauli:.4f}")
```

Add KTA values to the feature map comparison table (Table 5) in the paper.

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
