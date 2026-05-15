#!/usr/bin/env bash
# install_gpu_omics.sh — Install omics + validation dependencies.
# Pass --gpu to also install RAPIDS-singlecell.
set -euo pipefail

GPU=0
for arg in "$@"; do
    [ "$arg" = "--gpu" ] && GPU=1
done

echo "=== Installing omics dependencies ==="
pip install -r requirements-omics.txt

if [ "$GPU" -eq 1 ]; then
    echo ""
    echo "=== Installing RAPIDS-singlecell (GPU) ==="
    # Adjust the CUDA version tag to match your driver (e.g. cu118, cu121, cu122)
    CUDA_TAG="${CUDA_TAG:-cu122}"
    pip install rapids-singlecell --extra-index-url "https://pypi.nvidia.com"
    echo "RAPIDS-singlecell installed."
fi

echo ""
echo "Done. Run scripts/dgx/check_environment.sh to verify."
