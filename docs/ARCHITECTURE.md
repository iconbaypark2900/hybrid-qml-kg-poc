# System Architecture

## Overview

The hybrid QML-KG pipeline integrates classical machine learning with quantum machine learning for biomedical link prediction on knowledge graphs.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Layer (kg_layer)                     │
├─────────────────────────────────────────────────────────────┤
│  kg_loader.py          │  kg_embedder.py                    │
│  - Load Hetionet       │  - Train embeddings (TransE)      │
│  - Extract task edges  │  - PCA reduction                   │
│  - Negative sampling   │  - Feature construction            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Classical Baseline (classical_baseline)           │
├─────────────────────────────────────────────────────────────┤
│  train_baseline.py                                          │
│  - Logistic Regression                                      │
│  - SVM (RBF)                                                │
│  - Random Forest                                            │
│  - Cross-validation                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Quantum Layer (quantum_layer)                   │
├─────────────────────────────────────────────────────────────┤
│  qml_model.py         │  qml_trainer.py                     │
│  - QMLLinkPredictor   │  - QMLTrainer                       │
│  - VQC/QSVC          │  - Loss tracking                    │
│  - Optimizers         │  - Metrics                          │
│  - Ansatzes           │                                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Execution Layer                           │
├─────────────────────────────────────────────────────────────┤
│  scripts/run_pipeline.py                                    │
│  - Orchestrates full pipeline                               │
│  - CLI interface                                            │
│  - Results aggregation                                      │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. Knowledge Graph Loading
```
Hetionet TSV → DataFrame → Task-specific edges → Train/Test split
```

### 2. Embedding Generation
```
Triples → TransE Training → Entity Embeddings → PCA Reduction
```

### 3. Feature Construction
```
Entity Embeddings → Pair Features → Classical/Quantum Features
```

### 4. Model Training
```
Features + Labels → Model Training → Evaluation → Metrics
```

## Key Modules

### kg_layer

**kg_loader.py:**
- `load_hetionet_edges()`: Downloads and loads Hetionet data
- `extract_task_edges()`: Filters edges by relation type
- `prepare_link_prediction_dataset()`: Creates train/test splits with negatives
- `get_hard_negatives_*()`: Hard negative mining strategies

**kg_embedder.py:**
- `HetionetEmbedder`: Main embedding class
- `train_embeddings()`: Trains TransE or generates deterministic embeddings
- `prepare_link_features()`: Classical features (4×d dimensions)
- `prepare_link_features_qml()`: Quantum features (d dimensions)
- `get_embedding()`, `get_all_embeddings()`: Embedding access
- `compute_similarity_stats()`, `tsne_visualize()`: Quality validation

### classical_baseline

**train_baseline.py:**
- `ClassicalLinkPredictor`: Wrapper for sklearn models
- `regularization_path()`: Regularization analysis
- `train_rbf_svc_cv()`: RBF-SVC with cross-validation

### quantum_layer

**qml_model.py:**
- `QMLLinkPredictor`: Wrapper for Qiskit ML models
- Supports VQC and QSVC
- Optimizer registry (COBYLA, SPSA, NFT, GradientDescent)
- Ansatz support (RealAmplitudes, EfficientSU2, TwoLocal)
- Loss callback tracking

**qml_trainer.py:**
- `QMLTrainer`: Training and evaluation orchestrator
- Loss history persistence
- Metrics computation and saving

### Scripts

**Pipeline Scripts:**
- `run_pipeline.py`: Main E2E pipeline with CLI flags
- `e2e_smoke.py`: Fast CI smoke test

**Analysis Scripts:**
- `compare_optimizers.py`: Optimizer comparison
- `ansatz_search.py`: Architecture search
- `hyperparameter_search.py`: Grid search with CV
- `model_comparison.py`: Classical model comparison
- `learning_curves.py`: Bias/variance diagnosis
- `embedding_validation.py`: Embedding quality checks
- `scaling_study.py`: Performance vs dataset size
- `multi_seed_experiment.py`: Statistical validation
- `stat_tests.py`: Significance testing

**Benchmarking:**
- `benchmarking/empirical_scaling.py`: Runtime measurement
- `benchmarking/profile_pipeline.py`: Code profiling

## Configuration

### Quantum Configuration (`config/quantum_config.yaml`)
- Backend selection (simulator/hardware)
- Execution mode
- Error mitigation levels
- Shot counts

### Pipeline Defaults
- Relation type: "CtD" (Compound treats Disease)
- Max entities: 300 (proof-of-concept)
- Embedding dimension: 32
- QML dimension: 5 (qubits)

## Results Storage

All results saved to `results/` directory:
- `*_metrics_*.json`: Model metrics
- `*_loss_*.json`: Training loss histories
- `*_predictions_*.csv`: Predictions and scores
- `experiment_history.csv`: Accumulated results

## Extensibility

### Adding New Models
1. Implement model wrapper in appropriate layer
2. Add to `train_and_evaluate()` methods
3. Update CLI flags in `run_pipeline.py`

### Adding New Features
1. Extend `prepare_link_features*()` methods
2. Update feature dimension calculations
3. Test with existing models

### Adding New Experiments
1. Create script in `scripts/`
2. Follow JSON/CSV output conventions
3. Document in `experiments/README.md`

