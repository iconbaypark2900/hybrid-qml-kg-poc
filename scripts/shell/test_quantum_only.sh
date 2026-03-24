#!/bin/bash

# Resolve project root (contains kg_layer/, scripts/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# Test only quantum configurations (skip classical models)
# Uses actual quantum kernels to find best quantum setup

python3 scripts/find_best_quantum_config.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --neg_ratio 2.0 \
  --qml_dim 12 \
  --top_n 5 \
  --use_cached_embeddings \
  --quantum_only \
  --random_state 42
