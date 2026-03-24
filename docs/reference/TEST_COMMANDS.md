# Pipeline Testing Commands

Evaluation uses PR-AUC (Average Precision). See [CV_EVALUATION_GUIDE.md](../CV_EVALUATION_GUIDE.md) for K-fold CV usage and [LEAKAGE_PREVENTION_GUIDE.md](../LEAKAGE_PREVENTION_GUIDE.md) for data-leakage safeguards.

## Evaluation Tiers

| Tier | Purpose | Est. time |
|------|---------|-----------|
| **Smoke** | CI, quick sanity | < 5 min |
| **Quick iteration** | Development, ablation | ~5–15 min |
| **Robust evaluation** | Reporting, comparisons | ~30–60 min |
| **Paper-ready** | Reproducible paper/PR | ~1–2 hrs |

---

## Smoke Test (CI, < 5 min)

Minimal pipeline run for CI; uses `--cheap_mode`:

```bash
python scripts/run_optimized_pipeline.py --relation CtD --cheap_mode
```

Alternative: `python scripts/e2e_smoke.py` (custom minimal path, no full pipeline).

---

## Quick Test (Fast Mode) — Quick iteration tier

Test the pipeline with all fixes applied; single split, fewer models:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --use_feature_selection \
    --fast_mode
```

---

## Full Test (Complete Pipeline) — Quick iteration tier

Run the complete pipeline without fast mode; single split:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --use_feature_selection \
    --embedding_epochs 100
```

---

## Robust Evaluation — Reporting tier

K-fold CV, no fast_mode; more reliable estimates:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --embedding_method RotatE \
    --embedding_dim 128 \
    --embedding_epochs 200 \
    --negative_sampling hard \
    --qml_dim 16 \
    --qml_feature_map Pauli \
    --qsvc_C 0.1 \
    --run_ensemble \
    --ensemble_method stacking \
    --tune_classical \
    --qml_pre_pca_dim 24 \
    --use_cv_evaluation \
    --cv_folds 5
```

---

## Paper-ready Evaluation — Full reproducibility

Robust evaluation + full-scale data + multi-model fusion:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --embedding_method RotatE \
    --embedding_dim 128 \
    --embedding_epochs 200 \
    --negative_sampling hard \
    --qml_dim 16 \
    --qml_feature_map Pauli \
    --qsvc_C 0.1 \
    --run_ensemble \
    --ensemble_method stacking \
    --tune_classical \
    --qml_pre_pca_dim 24 \
    --run_multimodel_fusion \
    --fusion_method bayesian_averaging \
    --use_cv_evaluation \
    --cv_folds 5
```

Omit `--max_entities` (default 0) for full-scale data. Add `--fast_mode` only for quicker iteration, not for final reporting.

## Diagnostic Test

Run diagnostics to check feature quality:

```bash
python scripts/diagnose_features.py \
    --relation CtD \
    --use_cached_embeddings \
    --full_graph_embeddings
```

## Step-by-Step Testing

### 1. Test with Task-Specific Embeddings (Baseline)

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_cached_embeddings \
    --use_feature_selection \
    --fast_mode
```

### 2. Test with Full-Graph Embeddings (Improved)

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --use_feature_selection \
    --fast_mode
```

### 3. Test Without Feature Selection

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --fast_mode
```

### 4. Test Classical Models Only

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --classical_only \
    --use_feature_selection \
    --fast_mode
```

### 5. Test Quantum Models Only

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --skip_quantum=false \
    --use_feature_selection \
    --fast_mode
```

## Expected Results

After running the pipeline, you should see:

1. **Feature Diagnostics Section**:
   - Feature variance analysis showing non-zero variance
   - Feature-to-sample ratio warnings (if applicable)
   - Feature filtering results

2. **Model Performance**:
   - Classical models should NOT get exactly 0.5000 PR-AUC
   - Models should show learning (PR-AUC > 0.5)
   - Quantum models should train successfully

3. **No Errors**:
   - No "zero variance" errors
   - No "missing embeddings" errors
   - Features should be built successfully

## Troubleshooting Commands

### Check Embeddings

```bash
python -c "
import numpy as np
emb = np.load('data/complex_128d_entity_embeddings.npy')
print(f'Embeddings shape: {emb.shape}')
print(f'Unique embeddings: {len(np.unique(emb, axis=0))}/{len(emb)}')
print(f'Mean std per feature: {np.std(emb, axis=0).mean():.6f}')
"
```

### Check Feature Variance (Quick)

```bash
python -c "
import numpy as np
import pandas as pd
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.advanced_embeddings import AdvancedKGEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder

# Load data
df = load_hetionet_edges()
task_edges, entity_to_id, id_to_entity = extract_task_edges(df, relation_type='CtD')
train_df, test_df = prepare_link_prediction_dataset(task_edges, random_state=42)

# Load embeddings
embedder = AdvancedKGEmbedder(embedding_dim=64, method='ComplEx', work_dir='data', random_state=42)
embedder.load_embeddings()
embeddings = embedder.get_all_embeddings()

# Convert IDs
train_df['source'] = train_df['source_id'].map(id_to_entity)
train_df['target'] = train_df['target_id'].map(id_to_entity)

# Build features
builder = EnhancedFeatureBuilder(normalize=False)
X_train, _ = builder.build_features(train_df, embeddings, fit_scaler=False)

# Check variance
feature_std = np.std(X_train, axis=0)
print(f'Features shape: {X_train.shape}')
print(f'Zero variance features: {np.sum(feature_std < 1e-10)}/{len(feature_std)}')
print(f'Mean std: {feature_std.mean():.6f}')
"
```

## Recommended Test Sequence

1. **First**: Run smoke or quick test to verify the pipeline
   ```bash
   python scripts/e2e_smoke.py
   # or: python scripts/run_optimized_pipeline.py --relation CtD --cheap_mode
   ```

2. **Second**: Run quick test with feature selection for iteration
   ```bash
   python scripts/run_optimized_pipeline.py --relation CtD --full_graph_embeddings --use_cached_embeddings --use_feature_selection --fast_mode
   ```

3. **Third**: For reporting, run robust evaluation (K-fold CV) or paper-ready command above

## Key Flags Explained

- `--relation CtD`: Task relation (Compound treats Disease)
- `--full_graph_embeddings`: Train on all relations (better embeddings)
- `--use_cached_embeddings`: Use pre-trained embeddings (faster)
- `--use_feature_selection`: Apply feature selection when ratio > 1.0
- `--fast_mode`: Faster training (fewer epochs, fewer models); omit for robust/paper-ready runs
- `--cheap_mode`: Minimal run for CI (implies fast_mode, caps entities/shots)
- `--use_cv_evaluation`: K-fold CV for more robust PR-AUC estimates (see [CV_EVALUATION_GUIDE](../CV_EVALUATION_GUIDE.md))
- `--cv_folds`: Number of folds when using `--use_cv_evaluation` (default: 5)
- `--run_multimodel_fusion`: Combine RF, ET, QSVC predictions via Bayesian averaging (or `--fusion_method`)
- `--classical_only`: Skip quantum models
- `--skip_quantum`: Skip quantum models (alternative flag)

