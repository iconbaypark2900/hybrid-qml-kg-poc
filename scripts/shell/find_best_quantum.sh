#!/bin/bash

# Resolve project root (contains kg_layer/, scripts/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# Find best quantum configuration using actual quantum kernels
# Uses best classical parameters from exploration
# 
# This script tests quantum configs with ACTUAL quantum kernels (not RBF proxy)
# WARNING: This will be SLOW - each config takes time to compute quantum kernels
#
# Options:
#   --top_n 5    : Test only top 5 configs from exploration (faster)
#   --top_n 10   : Test top 10 configs (slower but more thorough)

# Test top 5 configurations from exploration (recommended for speed)
python3 scripts/find_best_quantum_config.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --neg_ratio 2.0 \
  --qml_dim 12 \
  --top_n 5 \
  --use_cached_embeddings \
  --recommendations_file results/parameter_recommendations_20260121-154510.json \
  --random_state 42

# Alternative: Test all configurations (very slow!)
# python3 scripts/find_best_quantum_config.py \
#   --relation CtD \
#   --pos_edge_sample 1500 \
#   --neg_ratio 2.0 \
#   --qml_dim 12 \
#   --use_cached_embeddings \
#   --recommendations_file results/parameter_recommendations_20260121-154510.json \
#   --random_state 42
