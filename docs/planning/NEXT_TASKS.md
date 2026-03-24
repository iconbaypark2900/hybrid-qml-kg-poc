# Next Tasks for Hybrid QML-KG Project

This document outlines the prioritized next tasks based on the current project state. Best run (PR-AUC **0.7987**) has been achieved; these tasks focus on closing gaps, fixing TODOs, and exploring further improvements.

---

## Current Status

- **Best result**: Ensemble-QC-stacking (Pauli) at PR-AUC **0.7987**
- **Target PR-AUC > 0.70**: Achieved
- **Key components**: Full-graph RotatE, hard negatives, QSVC (C=0.1), stacking ensemble, GridSearchCV classical tuning

---

## 1. High Priority – Code TODOs

### 1.1 Iterative learning – extract hard pairs

**File:** `quantum_layer/iterative_learning.py` (line 419)

**Issue:** The iterative learning refinement loop has a TODO: extract `hard_pairs` from `X_train[hard_indices]`. The pipeline currently uses `hard_indices` but does not track source/target entity pairs for embedding refinement.

**Action:** Implement extraction of `(source_id, target_id)` pairs from `X_train[hard_indices]` so `refine_for_hard_examples()` can operate on concrete entity pairs.

---

### 1.2 Multi-model prediction combination

**Reference:** `docs/OPTIMIZATION_QUICKSTART.md` (line 275)

**Issue:** Planned feature to combine predictions from multiple models is not yet implemented.

**Action:** Design and implement a mechanism to combine predictions from multiple models (e.g., weighted averaging, meta-learning) beyond the existing stacking ensemble.

---

## 2. Medium Priority – VQC Improvement

**Current state:** VQC performs near random (best PR-AUC ~0.5474). QSVC reaches 0.7216.

**Actions:**

- Run `experiments/vqc_optimization_analysis.py` with extended iterations:
  ```bash
  python experiments/vqc_optimization_analysis.py --experiment optimizers --max_iter 100
  python experiments/vqc_optimization_analysis.py --experiment ansatzes
  ```
- Explore alternative optimizers (ADAM, L-BFGS-B) and ansatzes
- Consider warm start from classical model or curriculum learning for VQC
- Document findings in `results/vqc_analysis/`

---

## 3. Medium Priority – Performance Tuning

### 3.1 Graph features in QML path

The flag `--use_graph_features_in_qml` is implemented but optional. Run comparative experiments to quantify impact on ensemble and quantum-only performance.

### 3.2 Extended Optuna search

Run more trials to explore the hyperparameter space:

```bash
python scripts/optuna_pipeline_search.py --n_trials 50 --objective ensemble
python scripts/optuna_pipeline_search.py --n_trials 30 --objective qsvc
```

### 3.3 Reduce information loss

The pipeline discards ~95% of information (256D → 12D). Experiment with:

- Kernel PCA instead of or in addition to PCA
- Higher `--qml_pre_pca_dim` (e.g., 32)
- Mutual-information-based feature selection

### 3.4 Scale up data

If compute allows, reduce or remove `--max_entities` limits to improve generalization.

---

## 4. Lower Priority – Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| Multi-task learning | Train on multiple Hetionet relations simultaneously (e.g., CtD + DaG) |
| Attention mechanisms | Learn which relations/entities matter most |
| Graph neural networks | Replace static embeddings with GNNs |
| Active learning | Select most informative samples for labeling |
| Explainability | Interpret why the model predicts specific links |
| Quantum error mitigation | Apply for IBM Heron / hardware runs |
| Class imbalance | SMOTE, focal loss, or adjusted negative:positive ratios |
| Cross-validation | Nested CV, entity-based splits to avoid leakage |

---

## Quick Commands Reference

**Reproduce best result:**
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --optimize_feature_map_reps --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24 --fast_mode
```

**With graph features in QML:**
```bash
# Add --use_graph_features_in_qml to the above command
```

**Optuna HPO:**
```bash
python scripts/optuna_pipeline_search.py --n_trials 30 --objective ensemble
```

---

## References

- `docs/planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md` – Detailed experiment log and recommendations
- `docs/OPTIMIZATION_PLAN.md` – Full optimization roadmap
- `docs/WHY_QUANTUM_UNDERPERFORMS.md` – Root cause analysis
- `docs/overview/IMPLEMENTATION_RECAP.md` – Recent changes and GPU/hardware readiness
