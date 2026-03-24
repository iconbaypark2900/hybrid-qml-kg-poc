#!/bin/bash

# Resolve project root (contains kg_layer/, scripts/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# Fix Quantum Performance Issues
# 
# Applies fixes from WHY_QUANTUM_UNDERPERFORMS.md:
# 1. Higher dimensions (24D, 32D) to preserve information
# 2. Less aggressive reduction (skip pre-PCA, select more features)
# 3. Simpler feature maps (reps=1, linear entanglement) to reduce overfitting
# 4. Hybrid features (quantum + classical)
# 5. Better regularization

python3 scripts/fix_quantum_performance.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --neg_ratio 2.0 \
  --use_cached_embeddings \
  --random_state 42
