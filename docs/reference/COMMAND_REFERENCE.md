# Command Reference Guide

Complete guide to all command-line flags and usage examples for the optimized pipeline.

## Quick Start Commands

### Basic Run (Default Settings)
```bash
python scripts/run_optimized_pipeline.py --relation CtD
```

### Quantum-Only (QSVC Only)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only
```

### With Contrastive Learning + Improved Features (Recommended)
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

### Fast Mode (Quick Testing)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --fast_mode \
    --quantum_only
```

---

## All Available Flags

### Data Configuration
```bash
--relation CtD                    # Relation type: CtD, DaG, etc. (default: CtD)
--max_entities 500                # Limit number of entities (default: None = all)
```

### Embedding Configuration
```bash
--embedding_method ComplEx         # Method: TransE, ComplEx, RotatE, DistMult (default: ComplEx)
--embedding_dim 64                # Embedding dimension (default: 64)
--embedding_epochs 100             # Training epochs (default: 100)
--use_cached_embeddings            # Use cached embeddings if available
--full_graph_embeddings            # Train on full Hetionet (all relations)
```

### Feature Configuration
```bash
--use_graph_features               # Include graph features (default: True)
--use_domain_features              # Include domain features (default: True)
--use_improved_features            # Use improved feature engineering
--max_interaction_features 50      # Max interaction features (default: 50)
--use_feature_selection            # Apply feature selection when ratio > 1.0
```

### Quantum Configuration
```bash
--qml_dim 12                      # Number of qubits (default: 12)
--qml_encoding hybrid              # Encoding: amplitude, phase, hybrid, optimized_diff, tensor_product
--qml_feature_map ZZ              # Feature map: ZZ, Z, Pauli, custom_link_prediction (default: ZZ)
--qml_feature_map_reps 3           # Feature map repetitions (default: 3)
--qml_entanglement full            # Entanglement: linear, full, circular (default: full)
--use_data_reuploading             # Use data re-uploading (quantum-native)
--use_variational_feature_map      # Use variational (trainable) feature map
--optimize_feature_map_reps        # Optimize reps using kernel-target alignment
--use_classical_features_in_kernel # Use classical features in quantum kernel
--skip_quantum                     # Skip quantum models
```

### Contrastive Learning (New!)
```bash
--use_contrastive_learning         # Fine-tune embeddings with contrastive learning
--contrastive_margin 1.0           # Margin for triplet loss (default: 1.0)
--contrastive_epochs 50            # Fine-tuning epochs (default: 50)
```

### Negative Sampling
```bash
--negative_sampling random          # Strategy: random, hard, diverse (default: random)
--diversity_weight 0.5             # Weight for diverse sampling (default: 0.5)
```

### Model Selection
```bash
--classical_only                   # Run only classical models
--quantum_only                     # Run only quantum models
--skip_svm_rbf                     # Skip SVM-RBF model
```

### Calibration
```bash
--calibrate_probabilities          # Apply probability calibration
--calibration_method isotonic      # Method: isotonic, sigmoid (default: isotonic)
```

### Evaluation
```bash
--use_cv_evaluation                # Use K-Fold CV instead of single split
--cv_folds 5                       # Number of CV folds (default: 5)
--random_state 42                  # Random seed (default: 42)
--results_dir results              # Results directory (default: results)
--quantum_config_path config/quantum_config.yaml  # Quantum execution config path
```

### Performance
```bash
--fast_mode                        # Fast mode (fewer models, less tuning)
```

---

## Common Usage Patterns

### 1. Quantum Model with Advanced Features
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_dim 12 \
    --qml_feature_map custom_link_prediction \
    --qml_feature_map_reps 3 \
    --qml_entanglement full \
    --use_data_reuploading \
    --use_contrastive_learning \
    --contrastive_margin 1.5 \
    --contrastive_epochs 100 \
    --use_improved_features \
    --max_interaction_features 100
```

### 2. Classical Models Only
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --classical_only \
    --use_improved_features \
    --max_interaction_features 50
```

### 3. Full Pipeline (Classical + Quantum)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --use_contrastive_learning \
    --use_improved_features \
    --qml_dim 12 \
    --use_data_reuploading \
    --qml_feature_map custom_link_prediction
```

### 4. Experiment with Different Feature Maps
```bash
# ZZ Feature Map
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_feature_map ZZ \
    --qml_feature_map_reps 3

# Custom Link Prediction Feature Map
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_feature_map custom_link_prediction \
    --use_data_reuploading

# With Repetition Optimization
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_feature_map ZZ \
    --optimize_feature_map_reps
```

### 5. Full-Graph Embeddings (Richer Context)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --full_graph_embeddings \
    --use_contrastive_learning \
    --use_improved_features
```

### 6. Cross-Validation Evaluation
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --use_cv_evaluation \
    --cv_folds 5 \
    --use_contrastive_learning \
    --use_improved_features
```

### 7. Ideal vs Noisy Simulator Benchmark
```bash
# Run ideal + noisy simulator back-to-back
bash scripts/benchmark_ideal_noisy.sh CtD results --fast_mode --quantum_only

# Generate a comparison table from experiment_history.csv
python benchmarking/ideal_vs_noisy_compare.py --results_dir results
```

### 8. Hard Negative Sampling (More Challenging)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --negative_sampling hard \
    --use_contrastive_learning
```

### 9. Diverse Negative Sampling
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --negative_sampling diverse \
    --diversity_weight 0.7 \
    --use_contrastive_learning
```

### 10. Different Embedding Methods
```bash
# ComplEx (default)
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --embedding_method ComplEx \
    --embedding_dim 64

# RotatE
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --embedding_method RotatE \
    --embedding_dim 64

# TransE
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --embedding_method TransE \
    --embedding_dim 64
```

### 11. Different Quantum Encodings
```bash
# Hybrid (default)
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_encoding hybrid

# Amplitude encoding
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_encoding amplitude

# Phase encoding
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_encoding phase

# Tensor product encoding
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --qml_encoding tensor_product
```

---

## Flag Combinations by Use Case

### Maximum Performance (Recommended)
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --full_graph_embeddings \
    --use_contrastive_learning \
    --contrastive_margin 1.5 \
    --contrastive_epochs 100 \
    --use_improved_features \
    --max_interaction_features 100 \
    --use_data_reuploading \
    --qml_feature_map custom_link_prediction \
    --qml_dim 12 \
    --qml_feature_map_reps 3 \
    --qml_entanglement full
```

### Quick Testing
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --fast_mode \
    --qml_dim 8 \
    --use_cached_embeddings
```

### Debugging/Development
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --quantum_only \
    --max_entities 100 \
    --fast_mode \
    --random_state 42
```

### Production Run
```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --use_contrastive_learning \
    --contrastive_margin 2.0 \
    --contrastive_epochs 150 \
    --use_improved_features \
    --max_interaction_features 150 \
    --use_data_reuploading \
    --qml_feature_map custom_link_prediction \
    --qml_dim 12 \
    --optimize_feature_map_reps \
    --use_cv_evaluation \
    --cv_folds 5 \
    --random_state 42 \
    --results_dir results/production
```

---

## Tips

1. **Start Simple**: Begin with default settings, then add flags incrementally
2. **Use `--quantum_only`**: Faster iteration when focusing on quantum models
3. **Use `--use_cached_embeddings`**: Skip embedding training for faster runs
4. **Contrastive Learning**: Use `--use_contrastive_learning` to improve embedding separability
5. **Improved Features**: Use `--use_improved_features` for better feature engineering
6. **Full-Graph Embeddings**: Use `--full_graph_embeddings` for richer context (slower but better)
7. **Fast Mode**: Use `--fast_mode` for quick testing
8. **Cross-Validation**: Use `--use_cv_evaluation` for more robust evaluation

---

## Building Docker Containers from Personal Branches

When working on a personal branch, you can build Docker images directly from your branch code. This ensures you're testing the exact version of your branch.

### Step-by-Step Guide

**1. Navigate to Your Repository**
```bash
cd /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc
```

**2. Switch to Your Feature Branch**
```bash
git checkout your-feature-branch-name
```

If the branch isn't local yet:
```bash
# Create and switch to a new branch
git checkout -b your-feature-branch-name

# Or pull from remote
git fetch origin && git checkout your-feature-branch-name
```

**3. Build the Docker Image**

For CLI container:
```bash
docker build -f deployment/Dockerfile.cli -t hybrid-qml-kg-cli:your-feature-tag .
```

For API container:
```bash
docker build -f deployment/Dockerfile.api -t hybrid-qml-kg-api:your-feature-tag .
```

For Dashboard container:
```bash
docker build -f deployment/Dockerfile.dashboard -t hybrid-qml-kg-dashboard:your-feature-tag .
```

**4. Verify Your Image**
```bash
docker images
```

**5. Run the Container (Optional)**
```bash
# CLI container
docker run --rm -it \
  -v /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc:/app \
  -e PYTHONPATH=/app \
  hybrid-qml-kg-cli:your-feature-tag
```

### Using Commit Hash for Better Traceability

For even better traceability, include the short commit hash in your tag:

```bash
git checkout your-branch
docker build -f deployment/Dockerfile.cli -t hybrid-qml-kg-cli:feature-$(git rev-parse --short HEAD) .
```

This creates an image tagged with the specific commit, like `hybrid-qml-kg-cli:feature-a1b2c3d`.

### Example: Building from Personal Branch

```bash
# Ensure you're on your personal branch
git checkout your-personal-branch

# Build with branch name tag
docker build -f deployment/Dockerfile.cli -t hybrid-qml-kg-cli:personal-branch .

# Or build with commit hash
docker build -f deployment/Dockerfile.cli -t hybrid-qml-kg-cli:$(git rev-parse --short HEAD) .

# Verify
docker images | grep hybrid-qml-kg-cli

# Run
docker run --rm -it \
  -v /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc:/app \
  -e PYTHONPATH=/app \
  hybrid-qml-kg-cli:personal-branch
```

### Notes

- Always verify you're on the correct branch before building: `git branch` or `git status`
- Tag your images descriptively so you can track which branch/commit they came from
- The build context (`.`) uses the current directory, so ensure you're in the project root
- Mount volumes to persist data, models, and results between container runs

---

## Getting Help

To see all available flags:
```bash
python scripts/run_optimized_pipeline.py --help
```

