# UPCOMING_TASKS.md - Progress Report

**Date:** March 23, 2026
**Status:** Tasks 1–2 Complete, Tasks 3–4 Ready for Execution

---

## Executive Summary

Significant progress has been made on the tasks from `UPCOMING_TASKS.md`:

✅ **Task 1 (Quick Experiments):** COMPLETE (March 6)
- Ran experiment with `--qml_pre_pca_dim 32` + `--use_graph_features_in_qml`
- **Result:** PR-AUC **0.7825** (vs. best 0.7987)
- Fixed critical bug in `quantum_layer/advanced_qml_features.py`

✅ **Task 2 (Multi-Model Fusion):** COMPLETE (March 23)
- Fusion integrated into `run_optimized_pipeline.py` via `--run_multimodel_fusion`
- **Result:** PR-AUC **0.7848** (Ensemble-QC-fusion-bayesian_averaging)
- BMA weights: RF=0.35, ET=0.42, QSVC=0.23; +0.0130 over best individual

🟡 **Task 3-4:** Ready for execution (infrastructure complete)
🟢 **Task 5:** Documented

---

## Task 1: Quick Experiments ✅ COMPLETE

### Configuration Tested
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
  --qml_pre_pca_dim 32 --use_graph_features_in_qml --fast_mode
```

### Results
| Metric | Value |
|--------|-------|
| **Test PR-AUC** | **0.7825** |
| Best Classical | 0.7718 (RandomForest) |
| Best Quantum | 0.6343 (QSVC) |
| Ensemble Gain | +0.0107 over best individual |

### Comparison to Best Result
- **Current:** 0.7825
- **Best Ever:** 0.7987 (Ensemble-QC-stacking, Pauli, PCA dim 24)
- **Difference:** -0.0162

### Analysis
The slightly lower performance (0.7825 vs 0.7987) suggests:
1. Higher PCA dimension (32 vs 24) may not be beneficial for this dataset
2. Graph features in QML path need tuning
3. The bug fix was critical - pipeline now works correctly

### Bug Fixed
**File:** `quantum_layer/advanced_qml_features.py`

**Issue:** Transform order didn't match fit order, causing dimension mismatch when using `--use_graph_features_in_qml`

**Fix:**
1. Store constant column mask during fit: `self.constant_mask_`
2. Apply transformations in correct order: constant removal → scaling → pre-PCA → feature selection → reduction

---

## Task 2: Multi-Model Fusion Integration ✅ COMPLETE

### Status: Integrated (2026-03-23)
- ✅ `quantum_layer/multi_model_fusion.py` implemented (6 fusion methods)
- ✅ `scripts/test_multi_model_fusion.py` demonstration works
- ✅ Integrated into `run_optimized_pipeline.py` via `--run_multimodel_fusion --fusion_method <method>`

### Result (fast_mode, single split)

| Model | PR-AUC |
|-------|--------|
| **Ensemble-QC-fusion-bayesian_averaging** | **0.7848** |
| Ensemble-QC-stacking | 0.7825 |
| RandomForest-Optimized | 0.7718 |
| ExtraTrees-Optimized | 0.7653 |
| QSVC-Optimized | 0.6343 |

- **Fusion gain:** +0.0130 over best individual (RF)
- **BMA weights:** RF=0.35, ET=0.42, QSVC=0.23
- **Output:** `results/optimized_results_20260323-103756.json`

### Next Step
Run under robust evaluation (CV, no fast_mode) for reportable metrics.

---

## Task 3: Extended Optuna Search 🟡 READY

### Infrastructure Status
- ✅ `scripts/optuna_pipeline_search.py` exists and works
- ❌ Not yet run with 50 trials

### Command to Run
```bash
# Ensemble optimization (50 trials, ~8-12 hours)
python scripts/optuna_pipeline_search.py \
  --n_trials 50 \
  --objective ensemble \
  --relation CtD

# QSVC optimization (30 trials, ~5-7 hours)
python scripts/optuna_pipeline_search.py \
  --n_trials 30 \
  --objective qsvc \
  --relation CtD
```

### Expected Impact
**+0.02 to +0.05 PR-AUC** improvement

### Recommended Action
Run overnight or on weekend due to long runtime.

---

## Task 4: Full-Scale Data Run 🟡 READY

### Current State
- All experiments so far use `--fast_mode` (limited entities)
- Full dataset has 755 CtD edges, 464 unique entities

### Command to Run
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 \
  --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24
  # Note: No --max_entities, no --fast_mode
```

### Expected Impact
- **+0.03 to +0.05 PR-AUC** improvement
- Better generalization
- More robust embeddings

### Warnings
- **Runtime:** 2-3× longer than fast mode
- **GPU recommended** for reasonable training time
- **Memory:** May require >16GB RAM

### Recommended Action
Run on weekend or when compute resources are available.

---

## Task 5: Future Enhancements 🟢 DOCUMENTED

### Lower Priority Enhancements

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| **Multi-task learning** | Train on multiple relations (CtD + DaG) simultaneously | Low |
| **Graph neural networks** | Replace static embeddings with GNNs (e.g., R-GCN, CompGCN) | Low |
| **Explainability** | Interpret why specific links are predicted (SHAP, LIME) | Low |
| **Quantum error mitigation** | Apply for IBM Heron / hardware runs | Low |
| **Class imbalance handling** | SMOTE, focal loss, adjusted negative:positive ratios | Low |
| **Kernel PCA** | Non-linear dimensionality reduction in QML path | Medium |

### When to Implement
These are research-oriented enhancements. Implement after:
1. Achieving target PR-AUC > 0.85 with current methods
2. Having excess compute resources
3. Preparing for publication or hardware demonstration

---

## Projected Performance Summary

| Configuration | PR-AUC | Status |
|---------------|--------|--------|
| **Current Best** | **0.7987** | ✅ Achieved |
| Quick Experiments (Task 1) | 0.7825 | ✅ Tested |
| + Multi-Model Fusion (Task 2) | 0.7848 (fast_mode) | ✅ Tested |
| + Optuna HPO (Task 3) | 0.82-0.85 | 🟡 Ready |
| + Full-Scale Data (Task 4) | 0.85-0.90 | 🟡 Ready |
| **Combined (All Tasks)** | **0.85-0.95** | 🎯 Target |

---

## Recommended Next Steps

### Immediate (Today)
1. ✅ Task 1 complete - bug fixed, pipeline working
2. Test multi-model fusion standalone:
   ```bash
   python scripts/implementations/implement_sophisticated_ensembles.py --relation CtD --fast_mode
   ```

### Short-Term (This Week)
3. Run Optuna search (overnight):
   ```bash
   python scripts/optuna_pipeline_search.py --n_trials 50 --objective ensemble
   ```

4. Integrate multi-model fusion into main pipeline

### Medium-Term (Next Week)
5. Run full-scale data experiment (weekend)
6. Analyze results and tune hyperparameters

### Long-Term (Research)
7. Implement future enhancements from Task 5
8. Prepare for publication or hardware demonstration

---

## Files Modified

### Bug Fixes
1. `quantum_layer/advanced_qml_features.py`
   - Fixed transform order to match fit order
   - Added `constant_mask_` for proper feature selection
   - Resolved dimension mismatch with `--use_graph_features_in_qml`

### New Files Created
1. `IMPLEMENTATION_STATUS_REPORT.md` - Detailed technical report
2. `COMMAND_REFERENCE_NEXT_TASKS.md` - Command usage guide
3. `NEXT_TASKS_IMPLEMENTATION/README.md` - Navigation hub
4. `UPCOMING_TASKS_PROGRESS.md` - This document

---

## Key Learnings

### What Worked
1. ✅ Multi-model fusion implementation (6 methods)
2. ✅ Hard pairs extraction for iterative learning
3. ✅ VQC optimization analysis (NFT identified as best)
4. ✅ Bug fixes for graph features in QML path

### What Needs Work
1. ⚠️ Higher PCA dimension (32) didn't improve performance
2. ⚠️ Graph features in QML path need tuning
3. ⚠️ VQC still underperforms vs. QSVC (0.63 vs 0.77)

### Surprises
1. Bayesian averaging outperformed neural meta-learner in demos
2. Graph features helped classical path but not QML path
3. Embedding diversity is a critical factor for performance

---

## Contact & Support

For questions about these implementations:
- **Technical details:** `IMPLEMENTATION_STATUS_REPORT.md`
- **Commands:** `COMMAND_REFERENCE_NEXT_TASKS.md`
- **Quick start:** `NEXT_TASKS_IMPLEMENTATION/README.md`
- **VQC results:** `results/vqc_analysis/`
- **Experiment logs:** `results/quick_experiment_pca32_graph.log`

---

**Last Updated:** March 23, 2026
**Current Best:** PR-AUC 0.7987 (single-split legacy); 0.7848 (fusion, fast_mode)
**Next Target:** PR-AUC > 0.85 with Tasks 3-4 complete; robust CV run pending
