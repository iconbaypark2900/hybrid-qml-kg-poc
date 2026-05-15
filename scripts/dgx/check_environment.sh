#!/usr/bin/env bash
# check_environment.sh — Verify all prerequisites for hybrid-qml-kg-poc on DGX.
# Exits non-zero if any required check fails. Optional/GPU checks never fail the run.
set -euo pipefail

PASS=0
FAIL=0
WARN=0

check() {
    local label=$1; shift
    if "$@" &>/dev/null; then
        echo "  [OK]   $label"
        PASS=$((PASS+1))
    else
        echo "  [FAIL] $label"
        FAIL=$((FAIL+1))
    fi
}

soft_check() {
    local label=$1; shift
    if "$@" &>/dev/null; then
        echo "  [OK]   $label"
        PASS=$((PASS+1))
    else
        echo "  [warn] $label (optional)"
        WARN=$((WARN+1))
    fi
}

echo "=== Environment Check: hybrid-qml-kg-poc ==="
echo ""

echo "--- Python ---"
check "python3 >= 3.9" python3 -c "import sys; assert sys.version_info >= (3, 9)"
check "pip available" python3 -m pip --version

echo ""
echo "--- Core scientific stack ---"
check "numpy" python3 -c "import numpy"
check "pandas" python3 -c "import pandas"
check "scikit-learn" python3 -c "import sklearn"
check "scipy" python3 -c "import scipy"
check "yaml" python3 -c "import yaml"

echo ""
echo "--- Quantum stack ---"
check "qiskit" python3 -c "import qiskit"
check "qiskit-machine-learning" python3 -c "import qiskit_machine_learning"
soft_check "qiskit-aer" python3 -c "import qiskit_aer"
soft_check "qiskit-ibm-runtime" python3 -c "import qiskit_ibm_runtime"
soft_check "pykeen (KG embeddings)" python3 -c "import pykeen"

echo ""
echo "--- Omics layers ---"
check "anndata" python3 -c "import anndata"
check "scanpy" python3 -c "import scanpy"
check "pydantic" python3 -c "import pydantic"
check "requests (validators)" python3 -c "import requests"
soft_check "harmonypy" python3 -c "import harmonypy"
soft_check "scrublet (doublets)" python3 -c "import scrublet"
soft_check "gseapy (enrichment)" python3 -c "import gseapy"
soft_check "leidenalg (clustering)" python3 -c "import leidenalg"

echo ""
echo "--- Dashboard ---"
check "streamlit" python3 -c "import streamlit"

echo ""
echo "--- GPU / RAPIDS (optional — DGX Spark only) ---"
soft_check "cupy" python3 -c "import cupy"
soft_check "rapids-singlecell" python3 -c "import rapids_singlecell"
soft_check "CUDA device count > 0" python3 -c "import cupy as cp; assert cp.cuda.runtime.getDeviceCount() > 0"

echo ""
echo "--- Data files ---"
check "hetionet nodes TSV" test -f data/hetionet-v1.0-nodes.tsv
check "hetionet edges SIF" test -f data/hetionet-v1.0-edges.sif
soft_check "trained embeddings cache" test -f data/entity_embeddings.npy

echo ""
echo "--- Config files ---"
check "single_cell config" test -f config/single_cell_config.yaml
check "evidence_fusion config" test -f config/evidence_fusion_config.yaml
check "perturbation config" test -f config/perturbation_config.yaml
check "entity_resolution config" test -f config/entity_resolution_config.yaml

echo ""
echo "--- New bio layers (import smoke test) ---"
check "entity_resolution.hetionet_resolver" python3 -c "from entity_resolution.hetionet_resolver import HetionetResolver"
check "single_cell_layer.loaders" python3 -c "from single_cell_layer.loaders import load_single_cell_config"
check "perturbation_layer.reversal_score" python3 -c "from perturbation_layer.reversal_score import compute_reversal_score"
check "evidence_layer.feature_fusion" python3 -c "from evidence_layer.feature_fusion import fuse_evidence"
check "validation_layer.clinical_trials" python3 -c "from validation_layer.clinical_trials_validator import query_clinical_trials"

echo ""
echo "=== Summary: $PASS passed, $FAIL failed, $WARN optional warnings ==="
if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "Required checks failed. Install missing packages with:"
    echo "  bash scripts/dgx/install_gpu_omics.sh           # CPU path"
    echo "  bash scripts/dgx/install_gpu_omics.sh --gpu     # add RAPIDS"
    exit 1
fi
echo ""
echo "Environment ready. Run scripts/dgx/run_smoke_test.sh next."
