# Hybrid Quantum-Classical Knowledge Graph Link Prediction System

## Overview

This project implements a hybrid quantum-classical system for link prediction in knowledge graphs, specifically focusing on biomedical applications using the Hetionet dataset. The system combines classical machine learning approaches with quantum machine learning to predict drug-disease treatment relationships.

## Architecture

The system consists of several interconnected layers:

1. **Knowledge Graph Layer (kg_layer)**: Handles loading, processing, and embedding of knowledge graph data
2. **Classical Baseline Layer (classical_baseline)**: Implements classical ML models (Logistic Regression, SVM, Random Forest)
3. **Quantum Layer (quantum_layer)**: Implements quantum ML models (QSVC, VQC) and quantum-classical ensembles
4. **Middleware Layer (middleware)**: Provides API endpoints for predictions
5. **Benchmarking Layer (benchmarking)**: Contains evaluation tools and dashboard
6. **Utils**: Common utilities for reproducibility, evaluation, and calibration

## Key Improvements Made

### 1. Enhanced Quantum Kernel Computation

#### Improved Caching Mechanism
- Added more robust caching for quantum kernel matrices with better cache key generation
- Included quantum configuration path in cache key to differentiate between different quantum setups
- Added proper error handling for cache operations

#### Performance Optimizations
- Implemented Nyström approximation for large datasets to reduce kernel computation time
- Added entrywise zero-noise extrapolation (ZNE) for scalable error mitigation
- Enhanced kernel observables computation for better debugging and analysis

### 2. Robust Error Handling and Logging

#### Quantum Executor Enhancements
- Added comprehensive fallback mechanisms for different sampler implementations
- Improved error messages with detailed traceback information
- Added multiple fallback options for different Qiskit versions and installations

#### Better Logging Throughout
- Enhanced logging with more informative messages
- Added debug-level logging for troubleshooting
- Improved error propagation and handling

### 3. Optimized Embedding Pipeline

#### Efficient Feature Preparation
- Added batch processing for better performance with large datasets
- Implemented proper NaN and empty string handling
- Added validation for missing entities with appropriate warnings
- Pre-allocated arrays for better memory efficiency

#### Deterministic Embeddings
- Maintained fallback to deterministic random embeddings when PyKEEN is unavailable
- Ensured reproducibility across runs with proper seeding

### 4. Quantum-Classical Ensemble Integration

#### New Ensemble Module
Created `quantum_classical_ensemble.py` with:
- Multiple ensemble methods (weighted average, voting, stacking)
- Proper calibration support
- Diversity evaluation metrics
- Factory function for easy instantiation

#### Flexible Combination Strategies
- Weighted averaging with configurable weights
- Soft voting for probability aggregation
- Stacking with meta-learner training
- Evaluation of ensemble diversity and improvement

## Usage Examples

### Basic Quantum-Classical Ensemble

```python
from quantum_layer.quantum_classical_ensemble import create_optimized_quantum_classical_ensemble

# Create an optimized ensemble
ensemble = create_optimized_quantum_classical_ensemble(
    ensemble_method="weighted_average",
    random_state=42
)

# Fit the ensemble (requires prepared features)
ensemble.fit(X_train, y_train, X_quantum=X_train_quantum, X_classical=X_train_classical)

# Make predictions
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)
```

### Using Different Ensemble Methods

```python
# Weighted average with custom weights
ensemble = QuantumClassicalEnsemble(
    quantum_model=quantum_model,
    classical_model=classical_model,
    ensemble_method="weighted_average",
    weights={"quantum": 0.6, "classical": 0.4}
)

# Stacking ensemble
ensemble = QuantumClassicalEnsemble(
    quantum_model=quantum_model,
    classical_model=classical_model,
    ensemble_method="stacking",
    use_stacking=True
)
```

### Evaluating Ensemble Diversity

```python
# Evaluate how diverse and effective the ensemble is
diversity_metrics = ensemble.evaluate_ensemble_diversity(
    X_test, y_test,
    X_quantum=X_test_quantum,
    X_classical=X_test_classical
)

print(f"Correlation between models: {diversity_metrics['correlation']:.3f}")
print(f"Disagreement rate: {diversity_metrics['disagreement_rate']:.3f}")
print(f"Ensemble improvement: {diversity_metrics['ensemble_improvement']:.3f}")
```

## Configuration Files

### Quantum Configuration (`config/quantum_config.yaml`)
Controls quantum execution parameters including:
- Execution mode (simulator, Heron, etc.)
- Noise models and error mitigation settings
- Sampler configurations
- Hardware-specific parameters

### Knowledge Graph Configuration (`config/kg_layer_config.yaml`)
Controls embedding and feature generation:
- Embedding dimensions
- QML dimension reduction
- Feature engineering options

## Running the System

### Training the Full Pipeline

```bash
python scripts/run_optimized_pipeline.py --relation CtD --results_dir results --fast_mode
```

### Running Quantum-Only

```bash
python scripts/run_optimized_pipeline.py --relation CtD --results_dir results --fast_mode --quantum_only
```

### Launching the Dashboard

```bash
streamlit run benchmarking/dashboard.py
```

### Starting the API

```bash
uvicorn middleware.api:app --reload
```

## Performance Considerations

### Quantum Kernel Optimization
- Use Nyström approximation for datasets > 500 samples
- Enable caching to avoid recomputing kernels
- Consider using smaller feature dimensions for quantum models

### Memory Management
- Large kernel matrices can consume significant memory
- Use appropriate batch sizes for feature preparation
- Monitor memory usage during training

### Execution Modes
- Simulator mode for development and testing
- Noisy simulation for realistic performance estimates
- Hardware execution for production (requires IBM Quantum access)

## Troubleshooting

### Common Issues

1. **Qiskit Import Errors**: Ensure proper Qiskit installation with required components
2. **Memory Issues**: Reduce dataset size or enable Nyström approximation
3. **Slow Execution**: Use fast_mode or reduce embedding dimensions
4. **Missing Entities**: Check that entity IDs match between training and test sets

### Debugging Tips

- Enable DEBUG logging for detailed information
- Check cache directories for saved computations
- Verify quantum configuration matches available hardware
- Ensure proper environment variables are set for IBM Quantum access

## Development Guidelines

### Adding New Features

1. Follow the existing modular architecture
2. Maintain backward compatibility where possible
3. Add comprehensive logging and error handling
4. Include appropriate tests and documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the existing code style
4. Add tests for new functionality
5. Submit a pull request with a clear description

## Future Enhancements

### Planned Improvements

1. Support for additional quantum algorithms
2. Enhanced ensemble methods with dynamic weight adjustment
3. Improved feature engineering techniques
4. Better integration with external knowledge bases
5. Advanced visualization tools for model interpretability

### Research Directions

1. Investigation of quantum advantage in specific domains
2. Scalability improvements for larger knowledge graphs
3. Transfer learning between different relation types
4. Integration with foundation models for enhanced representations