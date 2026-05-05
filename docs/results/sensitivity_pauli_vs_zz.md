# Sensitivity analysis — PauliFeatureMap vs ZZFeatureMap (supplementary)

**Status:** Draft — uses existing data from the experiment log; will be validated against the §8 paired-bootstrap CIs once the headline GPU run lands.
**Preregistration anchor:** [`preregistration/osf_preregistration_v1.md`](../../preregistration/osf_preregistration_v1.md) §5.1 (primary feature map = PauliFeatureMap reps=2), §5.2 (ZZFeatureMap depth 2 = supplementary sensitivity), §8.6 ("Feature-map choice (PauliFeatureMap primary vs ZZFeatureMap supplementary)").

This supplementary section answers the obvious reviewer question: *does the headline depend on the choice of feature map?*

## Setup

Both configurations are evaluated under the otherwise-identical headline pipeline:

- Hetionet v1.0 Compound-treats-Disease (CtD) edges as positives
- Hard negative sampling, 1:1 ratio
- Full-graph RotatE 128D node embeddings (200 epochs, PyKEEN)
- Pair features = concat + diff + Hadamard, pre-PCA reduced to 24D, then PCA-reduced to 16D for the quantum kernel
- 16-qubit `FidelityQuantumKernel`
- QSVC with C = 0.1, evaluated on the same train/val/test splits
- Stacking ensemble combining QSVC + tuned LR + tuned RF + tuned ET; meta-learner is a logistic regression on out-of-fold base-learner probabilities

The only varied factor is the quantum feature map:

| Configuration | Feature map | Reps |
|---|---|---|
| Primary (headline) | `PauliFeatureMap` | 2 |
| Supplementary | `ZZFeatureMap` | 2 (depth) |

## Headline numbers (from existing experiment log)

Source: [`README.md`](../../README.md) "Experiment Log."

| Model | PR-AUC (Pauli) | PR-AUC (ZZ) | Δ (Pauli − ZZ) |
|---|---|---|---|
| RandomForest-Optimized | 0.7838 | 0.7838 | 0.0000 |
| ExtraTrees-Optimized | 0.7807 | 0.7807 | 0.0000 |
| QSVC-Optimized | 0.6343 | 0.7216 | −0.0873 |
| **Stacking ensemble** | **0.7987** | **0.7408** | **+0.0579** |

(Classical baselines are unchanged across feature-map configurations as expected — they don't see the quantum kernel.)

## Reading the table

Two effects worth noting:

1. **QSVC alone is *better* under ZZ** (PR-AUC 0.7216 vs 0.6343 with Pauli). The quantum kernel produced by `ZZFeatureMap` is more linearly informative on its own.
2. **The stacking ensemble is *better* under Pauli** (PR-AUC 0.7987 vs 0.7408). The Pauli kernel is more *complementary* to the classical baselines — the meta-learner extracts more lift when QSVC's failure modes are uncorrelated with RF/ET's.

These effects point in opposite directions for the QSVC-alone (H1) and ensemble (H1b) hypotheses. The headline (H1b) ranks Pauli ahead; H1 (QSVC alone) would actually rank ZZ ahead. This is a real outcome of the methodology, not a bug.

## Implication for the headline

Two ways this could land in the manuscript:

- **Conservative reading:** Pauli is primary because the *ensemble* — the deliverable the paper foregrounds — is better with Pauli. ZZ is supplementary precisely because it documents the dual where QSVC alone wins and the ensemble loses; reviewers see both, and the choice is defensible on the operational grounds the headline cares about.
- **Aggressive reading:** Promote ZZ to primary, with the headline becoming H1 (QSVC alone) at PR-AUC 0.7216 vs the strongest classical baseline at 0.7838. This loses the ensemble headline.

The conservative reading is the one the preregistration locks (§5.1 primary = Pauli). This document records why that's defensible and what the dual case looks like.

## What's still pending

- These point estimates lack a paired-bootstrap CI. The §8 statistical analysis plan calls for 95% CIs on per-fold differences. Until [`docs/results/bootstrap_ci_analysis.md`](bootstrap_ci_analysis.md) lands from the headline GPU run on the DGX, the deltas above are unsigned: we don't know whether `+0.0579` (Pauli vs ZZ on the ensemble) would survive a 10,000-resample bootstrap.
- The experiment log was generated on a single seed. The headline run uses 5-fold stratified CV; per-fold variance will widen the bands.
- VQC results (PR-AUC ≤ 0.5474 under several ansatzes) are reported separately as supplementary in the main paper and are not part of this comparison.

## What to do once the GPU run lands

1. Re-run with `--qml_feature_map ZZ --qml_feature_map_reps 2` on the same Hetionet snapshot (SHA-256 in [`docs/reproducibility/hetionet_snapshot.md`](../reproducibility/hetionet_snapshot.md)) and `BOOTSTRAP_SEED = 20260504`.
2. Persist the cached folds as `results/cv_predictions_zz/fold_{0..4}.npz` (separate cache_dir from the Pauli headline run).
3. Compute paired-bootstrap CIs for: ensemble-Pauli vs ensemble-ZZ, QSVC-Pauli vs QSVC-ZZ. Both as additional rows in [`docs/results/bootstrap_ci_analysis.md`](bootstrap_ci_analysis.md) under a new "Sensitivity (§8.6): feature-map choice" section.
4. Replace the point estimates in this doc with `(point, [lo, hi])` tuples and recompute the deltas with CI.

## Files referenced

- [`preregistration/osf_preregistration_v1.md`](../../preregistration/osf_preregistration_v1.md) §5.1, §5.2, §8.6
- [`README.md`](../../README.md) "Experiment Log" — source of the point estimates above
- [`utils/preregistered_constants.py`](../../utils/preregistered_constants.py) `QSVC_FEATURE_MAP_TYPE`, `QSVC_FEATURE_MAP_REPS`, `QSVC_FEATURE_MAP_SENSITIVITY`
