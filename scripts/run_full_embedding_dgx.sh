#!/usr/bin/env bash
# Full-graph KG embedding (RotatE) + classical evaluation — tuned for NVIDIA DGX Spark / CUDA hosts.
# PyKEEN uses GPU when torch.cuda.is_available() (see kg_layer/advanced_embeddings.py).
#
# Usage (from repo root):
#   ./scripts/run_full_embedding_dgx.sh
#   LOG_PATH=/path/to/run.log ./scripts/run_full_embedding_dgx.sh
#
# Optional environment overrides:
#   RELATION, EMBEDDING_METHOD, EMBEDDING_DIM, EMBEDDING_EPOCHS, NEGATIVE_SAMPLING,
#   QUANTUM_CONFIG, RESULTS_DIR, PYTHON (python binary)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

PY="${PYTHON:-}"
if [[ -z "$PY" ]] && [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PY="$PROJECT_ROOT/.venv/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="python3"
fi

RELATION="${RELATION:-CtD}"
EMBEDDING_METHOD="${EMBEDDING_METHOD:-RotatE}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
EMBEDDING_EPOCHS="${EMBEDDING_EPOCHS:-200}"
NEGATIVE_SAMPLING="${NEGATIVE_SAMPLING:-hard}"
QUANTUM_CONFIG="${QUANTUM_CONFIG:-config/quantum_config_dgx.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-$PROJECT_ROOT/$RESULTS_DIR/full_embedding_dgx_${STAMP}.log}"
mkdir -p "$(dirname "$LOG_PATH")"

echo "=== Full-graph embedding (DGX-oriented) ==="
echo "Repo:       $PROJECT_ROOT"
echo "Python:     $PY"
echo "Log:        $LOG_PATH"
echo "Relation:   $RELATION | method=$EMBEDDING_METHOD dim=$EMBEDDING_DIM epochs=$EMBEDDING_EPOCHS"
echo ""

if command -v nvidia-smi &>/dev/null; then
  echo "--- nvidia-smi (summary) ---"
  nvidia-smi -L 2>/dev/null || true
  echo ""
fi

echo "--- PyTorch / CUDA ---"
"$PY" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda version (build):", getattr(torch.version, "cuda", None))
    print("device 0:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CUDA not available — PyKEEN will train embeddings on CPU (slow).")
    print("Install CUDA-enabled PyTorch matching your driver (see docs/deployment/DGX_SPARK.md).")
PY
echo ""

echo "--- Starting pipeline (tee to log) ---"
set +e
"$PY" scripts/run_optimized_pipeline.py \
  --relation "$RELATION" \
  --full_graph_embeddings \
  --embedding_method "$EMBEDDING_METHOD" \
  --embedding_dim "$EMBEDDING_DIM" \
  --embedding_epochs "$EMBEDDING_EPOCHS" \
  --negative_sampling "$NEGATIVE_SAMPLING" \
  --classical_only \
  --results_dir "$RESULTS_DIR" \
  --quantum_config_path "$QUANTUM_CONFIG" \
  2>&1 | tee "$LOG_PATH"
exit "${PIPESTATUS[0]}"
