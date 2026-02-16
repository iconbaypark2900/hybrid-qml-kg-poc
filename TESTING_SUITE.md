# Quantum Improvements Testing Suite

This directory contains a comprehensive testing framework for the quantum improvements implemented in the hybrid QML-KG system. The framework provides both terminal and dashboard interfaces for testing and visualizing the quantum improvements.

## Files

- `test_quantum_improvements_terminal.py`: Terminal-based testing framework
- `test_quantum_improvements_dashboard.py`: Streamlit dashboard for test visualization
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
python test_quantum_improvements_terminal.py
```

### Dashboard Interface

Run the Streamlit dashboard:
```bash
streamlit run test_quantum_improvements_dashboard.py
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