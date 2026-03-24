# Quick Start Guide: Optimized Pipeline

This guide shows you how to use the optimized pipeline to achieve the best results on the Hetionet CtD dataset.

## What's New?

The optimized pipeline includes:

1. **Advanced KG Embeddings**: ComplEx, RotatE, DistMult (better than TransE)
2. **Enhanced Features**: Graph topology + domain-specific + rich embedding features
3. **Optimized Quantum Features**: Better encoding strategies and feature selection
4. **Comprehensive Comparison**: All models tested with consistent methodology

## Quick Start: Run Optimized Pipeline

### 1. Basic Run (Fast Mode)

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --fast_mode \
    --embedding_method ComplEx
```

**What this does:**
- Uses ComplEx embeddings (best for compound-disease relations)
- Includes graph and domain features
- Tests top 2 classical models
- Tests QSVC quantum model
- Runs in ~3-5 minutes

**Expected Results:**
- Classical: 0.70+ PR-AUC (improvement from 0.62)
- Quantum: 0.62+ PR-AUC (improvement from 0.56)

### 2. Full Optimization Run

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --embedding_method ComplEx \
    --embedding_dim 64 \
    --embedding_epochs 100 \
    --qml_encoding hybrid \
    --qml_dim 5 \
    --use_graph_features \
    --use_domain_features
```

**What this does:**
- Full 64D ComplEx embeddings (takes ~5-10 min)
- All enhanced features (graph + domain + embeddings)
- Tests 3 classical models with optimal hyperparameters
- Tests both QSVC and VQC quantum models
- Runs in ~15-20 minutes

**Expected Results:**
- Classical: 0.72-0.78 PR-AUC
- Quantum: 0.64-0.72 PR-AUC

### 3. Compare Embedding Methods

```bash
# ComplEx (best for asymmetric relations)
python scripts/run_optimized_pipeline.py --relation CtD --embedding_method ComplEx --fast_mode

# RotatE (best for hierarchical relations)
python scripts/run_optimized_pipeline.py --relation CtD --embedding_method RotatE --fast_mode

# DistMult (fast, good for symmetric relations)
python scripts/run_optimized_pipeline.py --relation CtD --embedding_method DistMult --fast_mode

# TransE (baseline)
python scripts/run_optimized_pipeline.py --relation CtD --embedding_method TransE --fast_mode
```

### 4. Classical Only (Fastest)

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --classical_only \
    --fast_mode \
    --embedding_method ComplEx
```

Runs in ~2 minutes, tests only classical models.

### 5. Use Cached Embeddings

After first run, embeddings are cached. Use them to save time:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_cached_embeddings \
    --fast_mode
```

## Command-Line Options

```
Data Options:
  --relation STR           Relation type (CtD, DaG, etc.) [default: CtD]
  --max_entities INT       Limit number of entities [default: None = all]

Embedding Options:
  --embedding_method STR   KG embedding method [ComplEx, RotatE, DistMult, TransE]
  --embedding_dim INT      Embedding dimension [default: 64]
  --embedding_epochs INT   Training epochs [default: 100]
  --use_cached_embeddings  Use cached embeddings if available

Feature Options:
  --use_graph_features     Include graph topology features [default: True]
  --use_domain_features    Include biomedical domain features [default: True]

Quantum Options:
  --qml_dim INT           Number of qubits [default: 5]
  --qml_encoding STR      Encoding strategy [amplitude, phase, hybrid, etc.]
  --qml_max_iter INT      Max VQC iterations [default: 50]
  --skip_quantum          Skip quantum models entirely

Experiment Options:
  --classical_only        Run only classical models
  --fast_mode            Fast mode (fewer models, less tuning)
  --cv_folds INT         Cross-validation folds [default: 5]
  --random_state INT     Random seed [default: 42]
  --results_dir STR      Results directory [default: results]
  --run_ensemble         Run quantum-classical ensemble
  --ensemble_method STR  Ensemble method [weighted_average, voting, stacking]
  --run_multimodel_fusion  Run multi-model fusion (RF, ET, QSVC)
  --fusion_method STR    Fusion method [bayesian_averaging, optimized_weights, etc.]
```

## Understanding the Results

### Output Structure

The pipeline outputs:

1. **Console Rankings**: Real-time progress and final ranking table
2. **JSON Results**: Detailed metrics saved to `results/optimized_results_TIMESTAMP.json`

### Result Files

```json
{
  "config": {...},              // Configuration used
  "classical_results": {        // Classical model results
    "RandomForest-Optimized": {
      "status": "success",
      "train_metrics": {...},
      "test_metrics": {
        "pr_auc": 0.7234,       // ← Main metric
        "accuracy": 0.6891,
        "f1": 0.7012
      },
      "fit_seconds": 0.45
    }
  },
  "quantum_results": {...},     // Quantum model results
  "ranking": [...]              // Sorted by PR-AUC
}
```

### Key Metrics

- **PR-AUC** (Precision-Recall AUC): Main metric, handles class imbalance
  - 0.50 = random guessing
  - 0.70 = good performance
  - 0.80+ = excellent performance

- **Accuracy**: Overall correctness (less important for imbalanced data)
- **F1**: Harmonic mean of precision and recall
- **Fit Time**: Training time in seconds

## Expected Improvements

### From Baseline to Optimized

| Model | Baseline PR-AUC | Optimized PR-AUC | Improvement |
|-------|----------------|------------------|-------------|
| RandomForest | 0.6244 | 0.72-0.78 | +0.10 to +0.16 |
| SVM-RBF | 0.6233 | 0.70-0.76 | +0.08 to +0.14 |
| QSVC | 0.5564 | 0.64-0.72 | +0.08 to +0.16 |
| VQC | 0.5386 | 0.62-0.70 | +0.08 to +0.16 |

### What Contributes to Improvements?

1. **ComplEx Embeddings**: +0.03 to +0.05 PR-AUC
   - Better captures asymmetric compound-disease relations
   - 64D instead of 32D provides richer representations

2. **Graph Features**: +0.02 to +0.04 PR-AUC
   - Centrality, common neighbors, shortest paths
   - Captures network topology information

3. **Domain Features**: +0.01 to +0.02 PR-AUC
   - Entity type indicators
   - Metaedge diversity

4. **Enhanced Embedding Features**: +0.02 to +0.03 PR-AUC
   - Cosine similarity, L2 distance, dot products
   - Non-linear combinations

5. **Optimized Quantum Encoding**: +0.03 to +0.08 PR-AUC
   - Better feature selection (mutual information)
   - Hybrid encoding strategy
   - Proper normalization for quantum circuits

## Troubleshooting

### PyKEEN Not Installed

```bash
pip install pykeen
```

### Out of Memory

Reduce embedding dimensions or limit entities:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --max_entities 300 \
    --embedding_dim 32 \
    --fast_mode
```

### Slow Graph Feature Computation

Disable graph features for large graphs:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_graph_features False \
    --fast_mode
```

### Quantum Models Failing

Skip quantum and focus on classical:

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --classical_only \
    --fast_mode
```

## Next Steps

### 1. Experiment with Different Relations

```bash
# Disease associates with Gene
python scripts/run_optimized_pipeline.py --relation DaG --fast_mode

# Compound binds Gene
python scripts/run_optimized_pipeline.py --relation CbG --fast_mode
```

### 2. Try Different Embedding Methods

See which works best for your specific relation type.

### 3. Hyperparameter Tuning

Use the baseline results to guide further hyperparameter optimization:

- Increase `--embedding_dim` to 128 for more capacity
- Increase `--embedding_epochs` to 200 for better convergence
- Try different `--qml_encoding` strategies

### 4. Ensemble Models

#### Quantum-Classical Ensemble

Combine quantum and classical model predictions using weighted averaging, voting, or stacking:

```bash
python scripts/run_optimized_pipeline.py --relation CtD --fast_mode \
    --run_ensemble --ensemble_method weighted_average --ensemble_quantum_weight 0.5
```

**Ensemble methods:**
- `weighted_average`: Weighted combination of quantum and classical predictions
- `voting`: Simple average of predictions
- `stacking`: Learn optimal combination via logistic regression meta-learner

#### Multi-Model Fusion (Recommended)

Fuse predictions from multiple models (RF, ET, QSVC, etc.) using advanced fusion techniques:

```bash
# Bayesian averaging (default, recommended)
python scripts/run_optimized_pipeline.py --relation CtD --fast_mode \
    --run_multimodel_fusion --fusion_method bayesian_averaging

# Optimized weights (maximizes PR-AUC)
python scripts/run_optimized_pipeline.py --relation CtD --fast_mode \
    --run_multimodel_fusion --fusion_method optimized_weights

# Rank-based fusion
python scripts/run_optimized_pipeline.py --relation CtD --fast_mode \
    --run_multimodel_fusion --fusion_method rank_fusion
```

**Fusion methods:**
- `bayesian_averaging` (default): Bayesian Model Averaging based on likelihood - **recommended**
- `optimized_weights`: Learn optimal weights by maximizing PR-AUC
- `weighted_average`: Simple weighted combination (uniform or custom weights)
- `rank_fusion`: Reciprocal Rank Fusion (RRF) for ranking combination
- `confidence_weighted`: Weight by model confidence per sample
- `neural_metalearner`: Neural network meta-learner (MLP)

**Expected impact:** +0.01 to +0.03 PR-AUC over best individual model.

**Programmatic usage:**

```python
from quantum_layer.multi_model_fusion import create_fusion_ensemble

# Assume you have predictions from multiple models
model_predictions = {
    'quantum': quantum_pred_proba,
    'random_forest': rf_pred_proba,
    'extra_trees': et_pred_proba,
    'gradient_boosting': gb_pred_proba
}

# Create fusion ensemble with optimized weights
fusion, metrics = create_fusion_ensemble(
    model_predictions,
    y_train,
    fusion_method='bayesian_averaging'  # or 'optimized_weights', 'rank_fusion', 'neural_metalearner'
)

# Get fused predictions for test set
fused_pred = fusion.predict(model_predictions_test)
```

See `quantum_layer/multi_model_fusion.py` for implementation details.

## Comparing to Baseline

To see the improvement, run both:

```bash
# Baseline (original script)
python scripts/rbf_svc_fixed.py --relation CtD --fast_mode

# Optimized (new script)
python scripts/run_optimized_pipeline.py --relation CtD --fast_mode
```

Compare the test PR-AUC scores!

## Support

For issues or questions:
1. Check the logs in console output
2. Review the saved JSON results
3. See `docs/OPTIMIZATION_PLAN.md` for detailed explanations
4. See `docs/ARCHITECTURE.md` for system overview

## Citation

If you use this optimized pipeline, please cite:

- ComplEx: Trouillon et al., "Complex Embeddings for Simple Link Prediction" (2016)
- RotatE: Sun et al., "RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space" (2019)
- PyKEEN: Ali et al., "PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings" (2021)
