#!/bin/bash

# Resolve project root (contains kg_layer/, scripts/)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1

# Phase 2 Improvements: Expanded QSVC Grid + Hybrid Ensemble
# Builds on Phase 1: Task-Specific Fine-Tuning + Calibration + Phase 2 enhancements
# Expected improvements: +2-5% PR-AUC from expanded grid, +3-10% from hybrid ensemble

python3 scripts/run_optimized_pipeline.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --full_graph_embeddings \
  --embedding_method RotatE \
  --embedding_dim 128 \
  --embedding_epochs 200 \
  --use_evidence_weighting \
  --min_shared_genes 1 \
  --use_contrastive_learning \
  --contrastive_epochs 75 \
  --use_task_specific_finetuning \
  --task_specific_epochs 100 \
  --task_specific_lr 0.001 \
  --calibrate_probabilities \
  --calibration_method isotonic \
  --qml_dim 12 \
  --qml_encoding optimized_diff \
  --qml_reduction_method lda \
  --qml_feature_selection_method f_classif \
  --qml_feature_select_k_mult 4.0 \
  --qml_pre_pca_dim 0 \
  --qml_feature_map ZZ \
  --qml_feature_map_reps 2 \
  --qml_entanglement full \
  --negative_sampling hard \
  --neg_ratio 2.0 \
  --skip_svm_rbf \
  --skip_svm_linear \
  --skip_vqc \
  --random_state 42
