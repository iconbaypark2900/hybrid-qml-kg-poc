# Experiment runbook (phase-specific)

Deltas vs [planning/COMMAND_REFERENCE_NEXT_TASKS.md](../planning/COMMAND_REFERENCE_NEXT_TASKS.md): exact flags for this phase, machine notes, output paths.

*Last updated: 2026-03-23. 1 experiment logged below.*

## Output paths

- Pipeline JSON: `results/optimized_results_<stamp>.json`
- Optuna: `results/optuna/optuna_trials.csv`, `results/optuna/optuna_best.json`

## Recommended evaluation presets

| Preset | When to use | Flags |
|--------|-------------|-------|
| **Quick iteration** | Experiments, ablation, development | `--fast_mode`, single split (default) |
| **Robust evaluation** | Reporting, comparisons, model selection | No `--fast_mode`, `--use_cv_evaluation --cv_folds 5` |
| **Paper-ready** | Final results, reproducibility | Robust + `--run_multimodel_fusion`, full-scale (no `--max_entities`) |

**Rule of thumb:** Use `--fast_mode` only for quick iteration. For any reported metric or comparison, omit `--fast_mode` and add `--use_cv_evaluation` for K-fold CV (mean ± std PR-AUC). See [reference/TEST_COMMANDS.md](../reference/TEST_COMMANDS.md) for full commands.

## Commands (this phase)

| Experiment | Flags | Notes |
|------------|-------|-------|
| **Run 1: Combined (Fusion + Higher PCA + Graph)** | `--relation CtD --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard --qml_dim 16 --qml_feature_map Pauli --qsvc_C 0.1 --run_ensemble --ensemble_method stacking --qml_pre_pca_dim 32 --use_graph_features_in_qml --run_multimodel_fusion --fusion_method bayesian_averaging --fast_mode` | **PR-AUC: 0.7848** (fusion), **0.7825** (stacking), **0.7718** (RF best individual). **Runtime: ~3 min**. BMA weights: RF=0.35, ET=0.42, QSVC=0.23. Ensemble vs best individual: **+0.0130**. JSON: `results/optimized_results_20260323-103756.json` |

## Experiment Results

### Run 1: Combined (Fusion + Higher PCA + Graph Features)

**Date:** 2026-03-23  
**Command:**
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
  --qml_pre_pca_dim 32 --use_graph_features_in_qml \
  --run_multimodel_fusion --fusion_method bayesian_averaging --fast_mode
```

**Results:**

| Model | Type | PR-AUC | Accuracy | Time (s) |
|-------|------|--------|----------|----------|
| **Ensemble-QC-fusion-bayesian_averaging** | ensemble | **0.7848** | 0.6391 | 0.00 |
| Ensemble-QC-stacking | ensemble | 0.7825 | 0.6060 | 0.07 |
| RandomForest-Optimized | classical | 0.7718 | 0.6192 | 0.38 |
| ExtraTrees-Optimized | classical | 0.7653 | 0.6358 | 0.57 |
| QSVC-Optimized | quantum | 0.6343 | 0.5861 | 1.21 |

**Key findings:**
- **Fusion improvement:** +0.0130 over best individual (RF: 0.7718 → Fusion: 0.7848)
- **BMA weights:** RandomForest=0.35, ExtraTrees=0.42, QSVC=0.23
- **Stacking improvement:** +0.0107 over best individual
- **Best classical:** RandomForest-Optimized (0.7718)
- **Best quantum:** QSVC-Optimized (0.6343)
- **Quantum vs Classical gap:** -0.1375 (classical wins)

**Configuration notes:**
- RotatE embeddings: 128D (→256D complex→real), 200 epochs, full-graph context
- Hard negative sampling
- Quantum: 16 qubits, Pauli feature map, PCA 32→16D (62.53% variance)
- Graph features included in QML path (14 features)
- Nyström approximation (m=200) for QSVC kernel

**Separability diagnostics:**
- Raw embedding separation ratio: 0.9944
- Quantum feature separation ratio: 1.0961 (improved after PCA)
- Silhouette score: 0.0553 (highly overlapping)
- Significant quantum features: 9/16 (t-test p<0.05)

**Output files:**
- JSON: `results/optimized_results_20260323-103756.json`
- Predictions: `results/predictions_compare.csv`
- Quantum metrics: `results/quantum_metrics_QSVC_20260323-103756.json`

---

### Run 2: Robust 5-Fold CV Evaluation (Paper-Ready Config, Classical Only)

**Date:** 2026-03-23
**Purpose:** First reportable CV metric. Establishes honest baseline vs legacy single-split 0.7987.
**Command:**
```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24 \
  --run_multimodel_fusion --fusion_method bayesian_averaging \
  --use_cv_evaluation --cv_folds 5
```

**Results (5-fold CV, mean ± std):**

| Model | PR-AUC | ROC-AUC | Accuracy | F1-Score |
|-------|--------|---------|----------|----------|
| **RandomForest** | **0.7451 ± 0.0361** | 0.8240 ± 0.0215 | 0.7934 ± 0.0166 | 0.7986 ± 0.0192 |
| RBF-SVM | 0.4968 ± 0.0450 | 0.4979 ± 0.0246 | 0.5046 ± 0.0108 | 0.5610 ± 0.0769 |
| LogisticRegression | 0.4800 ± 0.0109 | 0.5116 ± 0.0263 | 0.5305 ± 0.0233 | 0.5517 ± 0.0167 |

**Per-fold RandomForest PR-AUCs:** [0.7059, 0.7493, 0.7286, 0.8120, 0.7299]

**Key findings:**
- **CV baseline is 0.7451**, ~0.054 below legacy single-split 0.7987 — expected variance reduction
- RandomForest dominates; LogReg and RBF-SVM near chance on PR-AUC
- **Limitation:** CV mode only evaluated classical models (skipped QSVC, ensemble, fusion)
- Embedding: RotatE 128D, early stopped at epoch 130 (best MRR 0.5954 at epoch 80)

**Output files:**
- CV JSON: `results/cv_results_20260323-122644.json`

---

## Pending experiment campaigns

All runs below produce provenance records in `results/benchmark_registry.jsonl`
via the registry wired in Phase A. Discovery metrics (top-10 hit rate, mean
rank) are now emitted automatically by `compute_metrics`.

### Campaign 1: Extended Optuna search (50+ trials)

**Goal:** Broader hyperparameter sweep to push ensemble PR-AUC beyond 0.80.

**Note:** The default subprocess timeout in `optuna_pipeline_search.py` is 600 s
per trial. For full-graph RotatE (128D, 200 epochs) this may not be enough.
Either lower `--embedding_epochs` during search or raise the timeout in the
script before launching.

```bash
python scripts/optuna_pipeline_search.py \
  --n_trials 50 --objective ensemble \
  --results_dir results/optuna
```

**Expected output:** `results/optuna/optuna_trials.csv`, `results/optuna/optuna_best.json`.

### Campaign 2: Full-scale run (no entity cap)

**Goal:** Embed all 47,031 Hetionet entities; use full positive edge set.

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --run_ensemble --ensemble_method stacking --tune_classical \
  --qml_pre_pca_dim 24 --max_entities 0 \
  --results_dir results/full_scale
```

GPU recommended; see `docs/deployment/DGX_SPARK.md` for the DGX wrapper.

### Campaign 3: Ablation matrix (spec sections 4-5)

Four conditions, each registered separately. Comparing QSVC on 16-dim against
classical on matched 16-dim (condition B vs C) is the scientifically valid
quantum comparison.

```bash
# Condition A — full classical (512-dim features)
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard \
  --classical_only --results_dir results/ablation_A

# Condition B — classical on reduced features (matches quantum 16-dim)
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard \
  --classical_only --restrict_classical_to_qml_dim --qml_dim 16 \
  --results_dir results/ablation_B

# Condition C — quantum only (16-dim)
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard \
  --quantum_only --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --results_dir results/ablation_C

# Condition D — stacking ensemble (A + C combined)
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard \
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2 \
  --qsvc_C 0.1 --run_ensemble --ensemble_method stacking \
  --results_dir results/ablation_D
```

### Campaign 4: Noisy simulator tier

**Goal:** Benchmark QSVC under depolarizing noise. Uses
`config/quantum_config_noisy.yaml` (depolarizing:0.01 + ZNE + readout
mitigation). Registry labels `execution_mode` as `simulator` with
`noise_model` set, so results stay separated from ideal-simulator runs.

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard \
  --quantum_only --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --quantum_config_path config/quantum_config_noisy.yaml \
  --results_dir results/noisy_sim
```

### After all campaigns

Query the registry to compare all runs side-by-side:

```bash
python scripts/benchmark_registry.py --list
```

Or load into a DataFrame:

```python
import pandas as pd
df = pd.read_json("results/benchmark_registry.jsonl", lines=True)
print(df[["run_id", "model", "backend", "metrics"]].to_string())
```
