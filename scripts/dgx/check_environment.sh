#!/usr/bin/env bash
# check_environment.sh — Verify all prerequisites for hybrid-qml-kg-poc on DGX.
set -euo pipefail

PASS=0
FAIL=0

check() {
    local label=$1; shift
    if "$@" &>/dev/null; then
        echo "  [OK]  $label"
        ((PASS++))
    else
        echo "  [FAIL] $label"
        ((FAIL++))
    fi
}

echo "=== Environment Check: hybrid-qml-kg-poc ==="
echo ""

echo "--- Python ---"
check "python3 >= 3.9" python3 -c "import sys; assert sys.version_info >= (3, 9)"
check "pip available" pip --version

echo ""
echo "--- Core dependencies ---"
check "anndata" python3 -c "import anndata"
check "scanpy" python3 -c "import scanpy"
check "harmonypy" python3 -c "import harmonypy"
check "gseapy" python3 -c "import gseapy"
check "pydantic" python3 -c "import pydantic"
check "qiskit" python3 -c "import qiskit"
check "qiskit-machine-learning" python3 -c "import qiskit_machine_learning"
check "pykeen" python3 -c "import pykeen"
check "streamlit" python3 -c "import streamlit"

echo ""
echo "--- GPU / RAPIDS (optional) ---"
check "cupy" python3 -c "import cupy"
check "rapids-singlecell" python3 -c "import rapids_singlecell"
check "CUDA devices > 0" python3 -c "import cupy as cp; assert cp.cuda.runtime.getDeviceCount() > 0"

echo ""
echo "--- Data files ---"
check "hetionet nodes TSV" test -f data/hetionet-v1.0-nodes.tsv
check "hetionet edges SIF" test -f data/hetionet-v1.0-edges.sif
check "single_cell config" test -f config/single_cell_config.yaml
check "evidence_fusion config" test -f config/evidence_fusion_config.yaml
check "perturbation config" test -f config/perturbation_config.yaml

echo ""
echo "=== Summary: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "Run scripts/dgx/install_gpu_omics.sh to install missing dependencies."
    exit 1
fi
echo "Environment looks good!"
