# Tutorial: Step-by-Step Guide

## Prerequisites

1. Python 3.8+
2. Virtual environment (recommended)
3. Install dependencies: `pip install -r requirements.txt`

## Quick Start

### 1. Basic Pipeline Run

```bash
# Run complete pipeline with defaults
python scripts/run_pipeline.py

# Customize parameters
python scripts/run_pipeline.py \
    --max_entities 500 \
    --qml_model_type QSVC \
    --qml_optimizer SPSA \
    --qml_max_iter 100
```

### 2. Check Results

Results are saved in `results/`:
- `latest_run.csv`: Latest metrics
- `experiment_history.csv`: All runs
- `qml_vqc_loss_*.json`: Loss curves (if VQC)
- `qml_vqc_loss_*.png`: Loss plots (if matplotlib available)

## Common Workflows

### Workflow 1: Diagnostic Analysis

```bash
# 1. Multi-seed validation
python scripts/multi_seed_experiment.py --seeds 42 123 456 789 1011

# 2. Regularization analysis
python scripts/regularization_path.py --penalty l2

# 3. Learning curves
python scripts/learning_curves.py --model LogisticRegression

# 4. Embedding validation
python scripts/embedding_validation.py
```

### Workflow 2: VQC Optimization

```bash
# 1. Compare optimizers
python scripts/compare_optimizers.py \
    --optimizers COBYLA SPSA NFT \
    --max_iter 200

# 2. Search ansatz architectures
python scripts/ansatz_search.py \
    --ansatzes RealAmplitudes EfficientSU2 TwoLocal \
    --reps_range 1 5

# 3. Hyperparameter grid search
python scripts/hyperparameter_search.py \
    --n_splits 5
```

### Workflow 3: Scaling Study

```bash
# Test different dataset sizes
python scripts/scaling_study.py \
    --entity_sizes 100 300 500 1000 2000

# Measure empirical runtimes
python benchmarking/empirical_scaling.py \
    --entity_counts 100 200 300 500 1000
```

### Workflow 4: Model Comparison

```bash
# Compare classical models
python scripts/model_comparison.py \
    --n_splits 5

# Nested CV for unbiased evaluation
python scripts/nested_cv.py \
    --outer_cv 5 --inner_cv 3
```

## Step-by-Step: Custom Experiment

### Example: Testing Different Feature Encodings

1. **Prepare data:**
```python
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges
from kg_layer.kg_embedder import HetionetEmbedder

df = load_hetionet_edges()
task_edges, _, _ = extract_task_edges(df, relation_type="CtD", max_entities=300)
embedder = HetionetEmbedder(embedding_dim=32, qml_dim=5)
embedder.train_embeddings(task_edges)
embedder.reduce_to_qml_dim()
```

2. **Test different encodings:**
```python
from kg_layer.feature_engineering import make_qml_features

# Test diff encoding
features_diff = make_qml_features(h_emb, t_emb, strategy="diff", qml_dim=5)

# Test hadamard encoding
features_had = make_qml_features(h_emb, t_emb, strategy="hadamard", qml_dim=5)

# Test combined encoding
features_both = make_qml_features(h_emb, t_emb, strategy="diff_prod", qml_dim=5)
```

3. **Train and evaluate:**
```python
from quantum_layer.qml_model import QMLLinkPredictor

qml_config = {
    "model_type": "VQC",
    "num_qubits": 5,
    "optimizer": "COBYLA",
    "max_iter": 50
}

predictor = QMLLinkPredictor(**qml_config)
predictor.fit(X_train, y_train)
score = predictor.score(X_test, y_test)
```

## Troubleshooting

### Issue: PyKEEN Not Available
**Solution:** Install with `pip install pykeen` or use deterministic embeddings (automatic fallback)

### Issue: Quantum Hardware Not Accessible
**Solution:** Pipeline defaults to local simulator. Configure `config/quantum_config.yaml` for hardware.

### Issue: Out of Memory
**Solution:** Reduce `--max_entities` or use smaller `--embedding_dim`

### Issue: VQC Training Slow
**Solution:** 
- Use QSVC instead (`--qml_model_type QSVC`)
- Reduce `--qml_max_iter`
- Use smaller dataset (`--max_entities`)

## Best Practices

1. **Always use random seeds** for reproducibility
2. **Run multi-seed experiments** for statistical validity
3. **Save results** with timestamps for tracking
4. **Use nested CV** for unbiased hyperparameter tuning
5. **Profile code** before optimizing (`benchmarking/profile_pipeline.py`)

## Next Steps

- Read [THEORY.md](THEORY.md) for mathematical foundations
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Read [WHEN_TO_USE_QUANTUM.md](WHEN_TO_USE_QUANTUM.md) for decision guidance

