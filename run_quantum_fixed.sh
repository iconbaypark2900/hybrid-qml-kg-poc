#!/bin/bash
# Run Full Pipeline with Quantum Performance Fixes
# 
# Applies fixes from WHY_QUANTUM_UNDERPERFORMS.md:
# 1. Higher dimensions (24D) to preserve information
# 2. Less aggressive reduction (skip pre-PCA, select more features)
# 3. Simpler feature maps (reps=1, linear entanglement) to reduce overfitting
# 4. Better regularization (expanded C grid)

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
  --qml_dim 24 \
  --qml_encoding hybrid \
  --qml_reduction_method pca \
  --qml_feature_selection_method f_classif \
  --qml_feature_select_k_mult 6.0 \
  --qml_pre_pca_dim 0 \
  --qml_feature_map ZZ \
  --qml_feature_map_reps 1 \
  --qml_entanglement linear \
  --negative_sampling hard \
  --neg_ratio 2.0 \
  --skip_svm_rbf \
  --skip_svm_linear \
  --skip_vqc \
  --random_state 42
