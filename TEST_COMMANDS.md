# Pipeline Testing Commands

## Quick Test (Fast Mode)

Test the pipeline with all fixes applied:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --use_feature_selection \
    --fast_mode
```

## Full Test (Complete Pipeline)

Run the complete pipeline without fast mode:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_cached_embeddings \
    --use_feature_selection \
    --embedding_epochs 100
```

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

1. **First**: Run diagnostics to verify embeddings and features
   ```bash
   python scripts/diagnose_features.py --relation CtD --use_cached_embeddings --full_graph_embeddings
   ```

2. **Second**: Run quick test with feature selection
   ```bash
   python scripts/run_optimized_pipeline.py --relation CtD --full_graph_embeddings --use_cached_embeddings --use_feature_selection --fast_mode
   ```

3. **Third**: If successful, run full pipeline
   ```bash
   python scripts/run_optimized_pipeline.py --relation CtD --full_graph_embeddings --use_cached_embeddings --use_feature_selection --embedding_epochs 100
   ```

## Key Flags Explained

- `--relation CtD`: Task relation (Compound treats Disease)
- `--full_graph_embeddings`: Train on all relations (better embeddings)
- `--use_cached_embeddings`: Use pre-trained embeddings (faster)
- `--use_feature_selection`: Apply feature selection when ratio > 1.0
- `--fast_mode`: Faster training (fewer epochs, fewer models)
- `--classical_only`: Skip quantum models
- `--skip_quantum`: Skip quantum models (alternative flag)

