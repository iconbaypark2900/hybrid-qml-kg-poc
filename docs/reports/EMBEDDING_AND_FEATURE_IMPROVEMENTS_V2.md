# Embedding Training and Feature Engineering Improvements V2

## Overview

This document describes the comprehensive improvements implemented to address the root cause: **embeddings trained for link prediction (ranking) don't learn class boundaries needed for classification**.

## Root Cause Analysis

From diagnostics:
- **Raw embedding separation ratio**: 0.9981 (barely separable)
- **Quantum feature separation ratio**: 1.0699 (slightly improved after PCA)
- **Silhouette score**: 0.0813 (highly overlapping)
- **QSVC overfitting**: Train PR-AUC 1.0000, Test PR-AUC 0.5500
- **Kernel separability failure**: Within-class ≈ between-class similarity

**Conclusion**: Embeddings need task-specific fine-tuning for classification, not just link prediction ranking.

## Solutions Implemented

### 1. Task-Specific Embedding Fine-Tuning ⭐ **NEW & RECOMMENDED**

**What it does**: Fine-tunes embeddings using **classification loss** (cross-entropy) on the target CtD task. This directly optimizes embeddings for classification rather than ranking.

**How it works**:
- Takes pre-trained embeddings from PyKEEN (RotatE, ComplEx, etc.)
- Adds a classification head (neural network) that maps concatenated head+tail embeddings to binary predictions
- Trains both the embedding layer and classifier using BCELoss
- Uses early stopping based on validation AUC
- Updates entity embeddings to improve link-level classification performance

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_task_specific_finetuning \
    --task_specific_epochs 100 \
    --task_specific_lr 0.001
```

**Parameters**:
- `--use_task_specific_finetuning`: Enable task-specific fine-tuning (recommended)
- `--task_specific_epochs`: Number of fine-tuning epochs (default: 100)
- `--task_specific_lr`: Learning rate (default: 0.001)

**Expected Impact**:
- Should significantly improve embedding separation ratio (0.9981 → >1.2)
- Should improve classification metrics (ROC-AUC, PR-AUC)
- Should reduce QSVC overfitting by improving input separability

### 2. Contrastive Learning Fine-Tuning ⚡

**What it does**: Fine-tunes embeddings using margin-based loss to maximize separation between positive and negative link pairs.

**How it works**:
- Uses margin-based loss to maximize distance between positive and negative pairs
- Minimizes within-class distances
- Updates entity embeddings to improve link-level separability

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_contrastive_learning \
    --contrastive_margin 1.0 \
    --contrastive_epochs 50
```

**Parameters**:
- `--use_contrastive_learning`: Enable contrastive fine-tuning
- `--contrastive_margin`: Margin for loss (default: 1.0, larger = more separation)
- `--contrastive_epochs`: Number of fine-tuning epochs (default: 50)

### 3. Quantum-Aware Embedding Fine-Tuning 🔬

**What it does**: Fine-tunes embeddings using quantum kernel separability as the objective.

**How it works**:
- Computes quantum kernel matrix on link embeddings
- Maximizes within-class similarity and minimizes between-class similarity
- Optimizes embeddings specifically for quantum models

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_quantum_aware_embeddings \
    --quantum_aware_epochs 100 \
    --qml_dim 12
```

**Parameters**:
- `--use_quantum_aware_embeddings`: Enable quantum-aware fine-tuning
- `--quantum_aware_epochs`: Number of epochs (default: 100)
- Requires `--qml_dim` to match quantum feature map

### 4. Improved Feature Engineering with Domain Knowledge 🎯

**What it does**: Creates interaction features and domain knowledge features guided by RandomForest importances.

**Features Created**:

1. **Interaction Features** (guided by RF importances):
   - Product features (x_i * x_j) for top important features
   - Ratio features (x_i / (x_j + eps)) for relative differences
   - Difference features (|x_i - x_j|) for absolute differences
   - Polynomial features (x_i^2) for non-linear patterns

2. **Class Difference Features**:
   - Distance to positive class centroid
   - Distance to negative class centroid
   - Ratio of distances
   - Absolute difference of distances

3. **Domain Knowledge Features** ⭐ **NEW**:
   - Compound type indicator (1.0 if head entity is Compound, else 0.0)
   - Disease type indicator (1.0 if tail entity is Disease, else 0.0)
   - Compound-Disease interaction indicator
   - Normalized compound numeric IDs (for potential grouping effects)
   - Normalized disease numeric IDs (for potential grouping effects)

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_improved_features \
    --use_domain_features \
    --max_interaction_features 50
```

**Parameters**:
- `--use_improved_features`: Enable improved feature engineering
- `--use_domain_features`: Add domain knowledge features (compound/disease properties)
- `--max_interaction_features`: Maximum interaction features to create (default: 50)

**Expected Impact**:
- Should improve feature separability
- Should help models learn class boundaries
- Domain features provide explicit compound-disease relationship signals

## Recommended Usage

### For Best Results (Combined Approach):

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --embedding_method RotatE \
    --embedding_dim 256 \
    --embedding_epochs 300 \
    --use_task_specific_finetuning \
    --task_specific_epochs 100 \
    --use_improved_features \
    --use_domain_features \
    --max_interaction_features 50 \
    --qml_dim 12 \
    --qml_feature_map ZZ \
    --qml_feature_map_reps 3 \
    --random_state 42
```

### For Quick Testing (Task-Specific Only):

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_task_specific_finetuning \
    --task_specific_epochs 50 \
    --qml_dim 12 \
    --random_state 42
```

## Implementation Details

### Task-Specific Fine-Tuning Architecture

```
Input: Pre-trained embeddings [n_entities, dim]
  ↓
Embedding Layer (learnable) [n_entities, dim]
  ↓
Link Embeddings: concat(head_emb, tail_emb) [n_samples, 2*dim]
  ↓
Classification Head:
  - Linear(2*dim → 128) + ReLU + Dropout(0.2)
  - Linear(128 → 64) + ReLU + Dropout(0.2)
  - Linear(64 → 1) + Sigmoid
  ↓
Output: Binary predictions [n_samples]
```

**Loss**: Binary Cross-Entropy Loss (BCELoss)
**Optimizer**: Adam with weight decay (L2 regularization)
**Early Stopping**: Based on validation AUC (patience: 10 epochs)

### Domain Feature Extraction

Domain features extract information from entity IDs:
- **Hetionet format**: `"Compound::DB00001"` or `"Disease::DOID:1234"`
- **Type extraction**: Splits on `::` to get entity type
- **Numeric ID extraction**: Extracts numeric parts for potential ordering/grouping
- **Normalization**: Numeric IDs are normalized for stability

## Expected Improvements

### Embedding Separability
- **Before**: Separation ratio 0.9981, Silhouette 0.0813
- **After (task-specific)**: Separation ratio >1.2, Silhouette >0.3
- **After (combined)**: Separation ratio >1.3, Silhouette >0.4

### Model Performance
- **Before**: QSVC PR-AUC 0.5500, severe overfitting
- **After (task-specific)**: QSVC PR-AUC >0.65, reduced overfitting
- **After (combined)**: QSVC PR-AUC >0.70, minimal overfitting

### Kernel Separability
- **Before**: Within-class ≈ between-class similarity
- **After**: Within-class > between-class similarity (clear separation)

## Notes

1. **Task-specific fine-tuning is recommended** as it directly optimizes for classification
2. **Domain features** provide explicit signals about compound-disease relationships
3. **All fine-tuning methods** can be combined, but task-specific is most effective
4. **Early stopping** prevents overfitting during fine-tuning
5. **Validation split** (20% of training) is used for early stopping

## Files Modified

- `kg_layer/task_specific_embeddings.py` (NEW): Task-specific fine-tuning implementation
- `kg_layer/improved_feature_engineering.py`: Added domain knowledge features
- `scripts/run_optimized_pipeline.py`: Integrated all fine-tuning methods and domain features

## Next Steps

1. Test task-specific fine-tuning alone
2. Test combined approach (task-specific + improved features + domain features)
3. Compare results with baseline (no fine-tuning)
4. Analyze embedding separability improvements
5. Evaluate QSVC performance improvements

