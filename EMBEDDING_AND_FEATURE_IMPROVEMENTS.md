# Embedding Training and Feature Engineering Improvements

## Overview

This document describes the improvements implemented to address the root cause identified by diagnostics: **raw embeddings lack class-separable information** (separation ratio: 1.0017).

## Root Cause Analysis

Diagnostics revealed:
- **Raw embeddings separation ratio**: 1.0017 (barely separable)
- **Only 1/256 features** have mean difference > 0.01 between classes
- **PCA reduction is NOT the problem** - it actually improves separability
- **Quantum kernel reflects input separability** - can't separate what isn't separable

**Conclusion**: Embeddings trained for link prediction (ranking) don't learn class boundaries needed for classification.

## Solutions Implemented

### 1. Contrastive Learning Fine-Tuning ⚡

**What it does**: Fine-tunes pre-trained embeddings using margin-based loss to maximize separation between positive and negative link pairs.

**How it works**:
- Takes pre-trained embeddings from PyKEEN
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

**Expected Impact**:
- Should increase embedding separation ratio from 1.0017 → >1.1
- Should improve quantum feature separability
- Should improve kernel separability

### 2. Improved Feature Engineering 🎯

**What it does**: Creates interaction features and class difference features guided by RandomForest importances.

**Features Created**:
1. **Interaction Features** (guided by RF importances):
   - Product features: `x_i * x_j` (emphasizes correlations)
   - Ratio features: `x_i / x_j` (emphasizes relative differences)
   - Difference features: `|x_i - x_j|` (emphasizes absolute differences)
   - Polynomial features: `x_i^2` (emphasizes non-linear patterns)

2. **Class Difference Features**:
   - Distance to positive class centroid
   - Distance to negative class centroid
   - Ratio of distances (which class is closer)
   - Absolute difference (separation strength)

**Usage**:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_improved_features \
    --max_interaction_features 50
```

**Parameters**:
- `--use_improved_features`: Enable improved feature engineering
- `--max_interaction_features`: Maximum interaction features to create (default: 50)

**Expected Impact**:
- Should create features that explicitly encode class differences
- Should improve classical model performance
- Should provide better features for quantum models

### 3. Combined Approach (Recommended) 🚀

Use both improvements together for maximum impact:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_contrastive_learning \
    --contrastive_margin 1.5 \
    --contrastive_epochs 100 \
    --use_improved_features \
    --max_interaction_features 100 \
    --use_data_reuploading \
    --qml_feature_map custom_link_prediction
```

## Implementation Details

### Contrastive Learning (`kg_layer/contrastive_embeddings.py`)

- **TripletDataset**: Creates triplets (anchor, positive, negative) for training
- **TripletLoss**: Margin-based loss to maximize positive-negative separation
- **ContrastiveEmbeddingFineTuner**: Fine-tunes entity embeddings using PyTorch

**Key Features**:
- Works with any pre-trained embeddings (ComplEx, RotatE, etc.)
- Preserves embedding structure while improving separability
- Computes contrastive loss before/after for diagnostics

### Improved Feature Engineering (`kg_layer/improved_feature_engineering.py`)

- **ImprovedFeatureEngineer**: Main class for enhanced feature creation
- **RandomForest Guidance**: Uses RF importances to select top features for interactions
- **Class Difference Features**: Explicitly encodes class separability

**Key Features**:
- Automatically trains RF model for feature importance
- Creates interaction features from top-K important features
- Adds class difference features that directly measure separability

## Expected Results

### Before Improvements:
- Embedding separation ratio: 1.0017
- Quantum feature separation ratio: 1.0018
- Silhouette score: 0.0044 (highly overlapping)
- Test PR-AUC: ~0.53

### After Improvements:
- **Contrastive Learning**: Should increase embedding separation ratio to >1.1
- **Improved Features**: Should improve feature separability and classical performance
- **Combined**: Should improve quantum model PR-AUC significantly

## Diagnostics

The pipeline now includes comprehensive separability diagnostics that show:
1. Raw embedding separability (before quantum reduction)
2. Quantum feature separability (after PCA reduction)
3. Information loss analysis (before vs after PCA)
4. Feature variance analysis
5. Statistical significance tests

Run with diagnostics to see improvements:
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_contrastive_learning \
    --use_improved_features
```

## Next Steps

1. **Run with contrastive learning** to improve embedding separability
2. **Run with improved features** to enhance feature engineering
3. **Run combined** for maximum impact
4. **Monitor diagnostics** to verify improvements at each stage
5. **Tune hyperparameters** (margin, epochs, interaction features) based on results

## Troubleshooting

**If contrastive learning fails**:
- Check PyTorch installation: `pip install torch`
- Reduce batch size if memory issues
- Reduce epochs if training is too slow

**If improved features fail**:
- Check if RandomForest can train (enough samples)
- Reduce `max_interaction_features` if too many features
- Check feature variance (may need normalization)

**If no improvement**:
- Check diagnostics to see where separability is lost
- Try increasing contrastive margin
- Try more interaction features
- Consider using full-graph embeddings (`--full_graph_embeddings`)

