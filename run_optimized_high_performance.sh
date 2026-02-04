#!/bin/bash
# Optimized command: Reverted to sampling + simplified quantum pipeline
# Based on successful run: QSVC achieved 0.735 PR-AUC with sampling
# Changes: Added sampling, removed pre-PCA, reduced feature selection

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
