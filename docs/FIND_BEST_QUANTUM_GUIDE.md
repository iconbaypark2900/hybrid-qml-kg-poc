# Finding Best Quantum Configuration with Actual Kernels

## Overview

The `find_best_quantum_config.py` script tests quantum parameter configurations using **actual quantum kernels** (not the misleading RBF proxy). It uses the best classical parameters from exploration and finds the best quantum setup.

---

## Why This Script?

### Problem with Exploration Script
- Used RBF kernel as a proxy for quantum kernel
- Predicted 0.9510 PR-AUC for quantum
- **Actual quantum kernel**: 0.3729 PR-AUC (much worse!)
- **Gap**: 0.5781 (57.8% difference!)

### Solution
- Test quantum configs with **actual quantum kernels**
- Uses best classical parameters from exploration
- Finds quantum configs that actually work

---

## Usage

### Quick Test (Recommended)
Test only top 5 configurations from exploration (faster):

```bash
bash find_best_quantum.sh
```

Or manually:
```bash
python3 scripts/find_best_quantum_config.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --top_n 5 \
  --use_cached_embeddings
```

### Full Test (Slow!)
Test all quantum configurations:

```bash
python3 scripts/find_best_quantum_config.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --use_cached_embeddings
```

---

## Options

- `--top_n N`: Test only top N configs from exploration (default: None = all)
- `--use_cached_embeddings`: Use cached embeddings (faster, default: True)
- `--recommendations_file`: Path to recommendations JSON (default: latest)
- `--qml_dim`: Number of qubits (default: 12)
- `--relation`: Relation type (default: CtD)

---

## What It Does

1. **Loads best classical parameters** from exploration recommendations
2. **Loads/trains embeddings** (uses cached if available)
3. **Tests quantum configurations** with actual quantum kernels:
   - If `--top_n` specified: Tests top N from exploration CSV
   - Otherwise: Tests full parameter grid
4. **Evaluates each config**:
   - Test PR-AUC (primary metric)
   - Test ROC-AUC
   - Overfitting gap (train - test)
5. **Saves results**:
   - CSV with all results
   - JSON with best configuration

---

## Output

### CSV File
`results/quantum_actual_kernel_results_YYYYMMDD-HHMMSS.csv`

Columns:
- `test_pr_auc`: Test PR-AUC (primary metric)
- `test_roc_auc`: Test ROC-AUC
- `train_pr_auc`: Training PR-AUC
- `overfitting_gap`: Train - Test gap
- `param_*`: Configuration parameters

### JSON File
`results/best_quantum_config_actual_YYYYMMDD-HHMMSS.json`

Contains:
- Best configuration parameters
- Performance metrics
- Ready to use in pipeline

---

## Performance

### Time Estimates
- **Top 5 configs**: ~10-30 minutes (depends on kernel cache)
- **Full grid**: ~1-3 hours (many configs to test)

### Speed Tips
1. **Use `--top_n 5`**: Test only top candidates
2. **Use `--use_cached_embeddings`**: Skip embedding training
3. **Kernel caching**: Script uses cached kernels when available
4. **Start small**: Test top 3-5 first, expand if needed

---

## Example Output

```
================================================================================
TOP 5 QUANTUM CONFIGURATIONS (ACTUAL KERNELS)
================================================================================

1. Test PR-AUC: 0.4523, ROC-AUC: 0.5234
   Overfitting Gap: 0.1234
   Config:
     encoding: hybrid
     reduction_method: pca
     feature_selection_method: mutual_info
     feature_select_k_mult: 2.0
     pre_pca_dim: 128
     feature_map: ZZ
     feature_map_reps: 2
     entanglement: full
```

---

## Using Best Configuration

After finding the best config, use it in the pipeline:

```bash
python3 scripts/run_optimized_pipeline.py \
  --relation CtD \
  --qml_encoding hybrid \
  --qml_reduction_method pca \
  --qml_feature_selection_method mutual_info \
  --qml_feature_select_k_mult 2.0 \
  --qml_pre_pca_dim 128 \
  # ... other best params from JSON
```

---

## Troubleshooting

### "No exploration CSV found"
- Run `explore_parameters.py` first to generate exploration results
- Or use full grid mode (omit `--top_n`)

### "Kernel computation is slow"
- This is expected - quantum kernels are computationally expensive
- Use `--top_n` to limit number of configs tested
- Kernel caching will speed up subsequent runs

### "1D features not suitable"
- LDA reduction gives 1D for binary classification
- Script automatically skips these configs
- Use PCA instead of LDA

---

## Next Steps

1. **Run the script** with `--top_n 5` to find best quantum config
2. **Compare results** with classical models (0.8859 PR-AUC)
3. **If quantum is competitive** (>0.70 PR-AUC), use in hybrid ensemble
4. **If quantum is poor** (<0.50 PR-AUC), focus on classical models

---

## Notes

- **Actual quantum kernels are slow** - be patient!
- **Results may differ** from exploration predictions (RBF proxy was misleading)
- **Best quantum config** may not beat classical models
- **Classical models are already excellent** (0.8859 PR-AUC)
