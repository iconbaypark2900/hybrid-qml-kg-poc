#!/bin/bash
cd /home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc
source .venv/bin/activate
export PYTHONPATH=/home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc:$PYTHONPATH

echo "================================"
echo "Running Benchmark: All 3 Algorithms"
echo "================================"

echo ""
echo "1/3: Training QSVC (Quantum SVC)..."
python -m quantum_layer.qml_trainer --model_type QSVC --qml_dim 5 --results_dir results

echo ""
echo "2/3: Training Classical RBF-SVC..."
python scripts/rbf_svc_fixed.py

echo ""
echo "3/3: Training VQC (Variational Quantum Classifier)..."
python -m quantum_layer.qml_trainer --model_type VQC --qml_dim 5 --max_iter 50 --results_dir results

echo ""
echo "================================"
echo "✅ Benchmark Complete!"
echo "================================"
echo "Results saved in:"
echo "  - results/quantum_metrics_QSVC_*.json"
echo "  - results/quantum_metrics_VQC_*.json"
echo "  - results/rbf_svc_128d_fixed_*.json"