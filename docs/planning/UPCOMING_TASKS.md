# Upcoming Tasks for Hybrid QML-KG Project

This document lists the **remaining** tasks after completion of the high-priority items from `NEXT_TASKS.md`. Best run remains PR-AUC **0.7987**.

---

## Completed (Reference)

- **1.1** Hard pairs extraction in `iterative_learning.py` ✅
- **1.2** Multi-model fusion (`quantum_layer/multi_model_fusion.py`) ✅
- **2** VQC optimization analysis (experiments run; NFT best ~0.5554) ✅

---

## 1. Immediate – Run Experiments (~1–2 hours)

Test improvements that are ready to use:

| Task | Flag / Change | Expected Gain |
|------|---------------|---------------|
| Higher PCA dimension | `--qml_pre_pca_dim 32` | +0.01 to +0.02 |
| Graph features in QML | `--use_graph_features_in_qml` | +0.01 to +0.03 |

**Command:**
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
  --qml_pre_pca_dim 32 --use_graph_features_in_qml --fast_mode
```

---

## 2. Pipeline Integration – Multi-Model Fusion ✅ COMPLETE

**Status:** Integrated into `scripts/run_optimized_pipeline.py` as of 2026-03-23.

Flags: `--run_multimodel_fusion --fusion_method bayesian_averaging`

**Result (fast_mode, single split):** PR-AUC **0.7848** (Ensemble-QC-fusion-bayesian_averaging), +0.0130 over best individual model. BMA weights: RF=0.35, ET=0.42, QSVC=0.23.

See `docs/upcoming-execution/EXPERIMENT_RUNBOOK.md` Run 1 for full details.

---

## 3. Extended Optuna Search (~8–12 hours)

**Action:** Run Optuna with more trials:

```bash
python scripts/optuna_pipeline_search.py --n_trials 50 --objective ensemble
```

**Expected impact:** +0.02 to +0.05 PR-AUC.

---

## 4. Full-Scale Data Run

**Action:** Run without `--max_entities` to use all entities:

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 32
```

**Expected impact:** +0.03 to +0.05. Runtime ~2–3× longer; GPU recommended.

---

## 5. Lower Priority – Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| Multi-task learning | Train on multiple Hetionet relations (e.g., CtD + DaG) |
| Graph neural networks | Replace static embeddings with GNNs |
| Explainability | Interpret why specific links are predicted |
| Quantum error mitigation | Apply for IBM Heron / hardware runs |
| Class imbalance | SMOTE, focal loss, adjusted negative:positive ratios |
| Kernel PCA | Non-linear dimensionality reduction in QML path |

---

## Projected Performance

| State | PR-AUC |
|-------|--------|
| Current best | 0.7987 |
| With quick experiments | 0.81–0.83 |
| Combined (full Optuna + full-scale) | 0.85–0.95 |

---

## References

- `NEXT_TASKS.md` – Original task list (some items completed)
- `IMPLEMENTATION_STATUS_REPORT.md` – Detailed completion status
- `COMMAND_REFERENCE_NEXT_TASKS.md` – Command usage guide
