#!/usr/bin/env bash
# install_gpu_omics.sh — Install omics + validation dependencies.
#
# Usage:
#   bash scripts/dgx/install_gpu_omics.sh                # CPU path (Scanpy, scrublet, etc.)
#   bash scripts/dgx/install_gpu_omics.sh --gpu          # also install RAPIDS-singlecell
#   CUDA_TAG=cu121 bash scripts/dgx/install_gpu_omics.sh --gpu
#
# RAPIDS CUDA tags (pick the one matching your driver):
#   cu118  → CUDA 11.8
#   cu121  → CUDA 12.1
#   cu122  → CUDA 12.2 (DGX Spark default at time of writing)
#   cu124  → CUDA 12.4
set -euo pipefail

GPU=0
DRY=0
for arg in "$@"; do
    case $arg in
        --gpu) GPU=1 ;;
        --dry-run) DRY=1 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
    esac
done

run() {
    echo "+ $*"
    if [ "$DRY" -eq 0 ]; then
        "$@"
    fi
}

echo "=== Installing omics dependencies (CPU path) ==="
run python3 -m pip install --upgrade pip
run python3 -m pip install -r requirements-omics.txt

if [ "$GPU" -eq 1 ]; then
    CUDA_TAG="${CUDA_TAG:-cu122}"
    echo ""
    echo "=== Installing RAPIDS-singlecell for $CUDA_TAG ==="

    # Detect CUDA runtime if nvidia-smi is present
    if command -v nvidia-smi &>/dev/null; then
        DRIVER_CUDA=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | head -1 || echo "unknown")
        echo "Detected driver: $DRIVER_CUDA"
    else
        echo "[warn] nvidia-smi not found — proceeding with CUDA_TAG=$CUDA_TAG"
    fi

    # rapids-singlecell depends on cuml/cudf; install from the NVIDIA index
    run python3 -m pip install rapids-singlecell --extra-index-url "https://pypi.nvidia.com"

    # Smoke check
    if [ "$DRY" -eq 0 ]; then
        if python3 -c "import rapids_singlecell" 2>/dev/null; then
            echo "[OK] rapids_singlecell imports cleanly"
        else
            echo "[FAIL] rapids_singlecell installed but does not import — check CUDA_TAG"
            exit 1
        fi
    fi
fi

echo ""
echo "Done. Verify with: bash scripts/dgx/check_environment.sh"
