# Quantum Improvements Testing Suite

This directory contains a comprehensive testing framework for the quantum improvements implemented in the hybrid QML-KG system. The framework provides both terminal and dashboard interfaces for testing and visualizing the quantum improvements.

## Evaluation tiers (pipeline)

For pipeline evaluation, use the appropriate tier for your goal:

| Tier | Purpose | Script / command | Est. time |
|------|---------|------------------|-----------|
| **Smoke** | CI, quick sanity | `python scripts/e2e_smoke.py` or `python scripts/run_optimized_pipeline.py --relation CtD --cheap_mode` | < 5 min |
| **Quick iteration** | Development, ablation | `run_optimized_pipeline --fast_mode` (single split) | ~5–15 min |
| **Robust evaluation** | Reporting, comparisons | `run_optimized_pipeline` without fast_mode, `--use_cv_evaluation --cv_folds 5` | ~30–60 min |
| **Paper-ready** | Reproducible paper/PR | Robust + `--run_multimodel_fusion`, full-scale (no max_entities) | ~1–2 hrs |

See [TEST_COMMANDS.md](TEST_COMMANDS.md) for full commands and [CV_EVALUATION_GUIDE.md](../CV_EVALUATION_GUIDE.md) for K-fold CV.

## Pipeline evaluation

- **`scripts/e2e_smoke.py`**: Minimal end-to-end test (kg_loader + embedder + classical + optional QSVC). Does not invoke the full `run_optimized_pipeline`. Target: < 3 min.
- **`scripts/pipeline_smoke.py`**: Invokes `run_optimized_pipeline --cheap_mode` for a full pipeline smoke test. Validates fusion, ensemble path, JSON output. Target: < 5 min.
- **`scripts/run_optimized_pipeline.py`**: Main pipeline. Use `--fast_mode` for iteration; omit and add `--use_cv_evaluation` for robust reporting.

## Files

- `tests/test_quantum_improvements_terminal.py`: Terminal-based testing framework
- `tests/test_quantum_improvements_dashboard.py`: Streamlit dashboard for test visualization
- `run_tests.py`: Main script to run both terminal and dashboard tests
- `test_results_*.json`: Generated test results files

## Quantum Improvements Tested

1. **Quantum-Enhanced Embeddings**: Advanced embedding techniques optimized for quantum models
2. **Quantum Transfer Learning**: Framework for transferring knowledge from pre-trained quantum models
3. **Advanced Error Mitigation**: State-of-the-art techniques including ZNE, PEC, and CDR
4. **Quantum Circuit Optimization**: Advanced optimization techniques for quantum circuits
5. **Quantum Kernel Engineering**: Adaptive and trainable quantum kernels
6. **Quantum Variational Feature Selection**: Quantum variational algorithms for feature selection

## Usage

### Terminal Interface

Run all tests in terminal mode:
```bash
python tests/test_quantum_improvements_terminal.py
```

### Dashboard Interface

Run the Streamlit dashboard:
```bash
streamlit run tests/test_quantum_improvements_dashboard.py
```

Or use the main script to run both:
```bash
python run_tests.py --mode both
```

### Command Line Options

- `--mode`: Choose between "terminal", "dashboard", or "both" (default: both)
- `--port`: Port for dashboard (default: 8501)

## Test Results

Test results are saved in JSON format with timestamps. The dashboard automatically loads the most recent test results file.

## Features

- Comprehensive test coverage for all quantum improvements
- Real-time visualization of test results
- Performance metrics and quantum advantage indicators
- System health monitoring
- Interactive controls for running tests
- Detailed logs and debugging information

## Dependencies

- Python 3.8+
- Streamlit
- Plotly
- Pandas
- NumPy
- Qiskit (for full functionality)
- Scikit-learn