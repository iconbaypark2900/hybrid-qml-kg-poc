# Experiments Directory

This directory contains research utilities and experimental scripts for investigating and improving the hybrid QML-KG pipeline.

## Purpose

These scripts are designed to:
- Explore optimization strategies for VQC training
- Analyze embedding quality and feature engineering approaches
- Conduct hyperparameter searches and architecture studies
- Validate model performance with statistical rigor

## Important Notes

⚠️ **These scripts are NOT part of the main pipeline entrypoint.**

- They are **research utilities** for investigation and development
- They are **excluded from CI/build** - intended for manual execution
- They may have dependencies on experimental libraries
- Results from these scripts inform improvements to the main pipeline

## Usage

Run scripts individually to investigate specific aspects:

```bash
# Compare optimizers
python scripts/compare_optimizers.py --optimizers COBYLA SPSA

# Search over ansatz architectures
python scripts/ansatz_search.py --ansatzes RealAmplitudes EfficientSU2 TwoLocal

# Hyperparameter grid search
python scripts/hyperparameter_search.py --n_splits 5

# Regularization path analysis
python scripts/regularization_path.py --penalty l2

# Multi-seed validation
python scripts/multi_seed_experiment.py --seeds 42 123 456 789 1011
```

## Integration with Main Pipeline

Findings from these experiments should be:
1. **Documented** in results/ with clear methodology
2. **Incorporated** into the main pipeline (`scripts/run_pipeline.py`) when validated
3. **Validated** with statistical significance before adoption

## Files

- `cross_validation_framework.py` - Cross-validation utilities
- `feature_engineering_qml.py` - QML feature encoding experiments
- `vqc_optimization_analysis.py` - VQC optimization analysis tools
- `IMPLEMENTATION_SUMMARY.md` - Summary of implementation approaches
- `README.md` - This file

## Contributing

When adding new experimental scripts:
1. Document the purpose and methodology in script docstring
2. Save results to `results/` directory with clear naming
3. Update this README if adding new categories of experiments
4. Ensure scripts can run independently (no hidden dependencies)
