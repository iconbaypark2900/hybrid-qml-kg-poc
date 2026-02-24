#!/bin/bash
# Optimized command targeting PR-AUC 0.80+
# Building on successful 0.735 run with aggressive improvements
# Changes: More qubits, deeper circuit, more features, more training

python3 scripts/run_optimized_pipeline.py \
  --relation CtD \
  --pos_edge_sample 1500 \
  --full_graph_embeddings \
  --embedding_method RotatE \
  --embedding_dim 128 \
  --embedding_epochs 250 \
  --use_evidence_weighting \
  --min_shared_genes 1 \
  --use_contrastive_learning \
  --contrastive_epochs 100 \
  --qml_dim 14 \
  --qml_encoding optimized_diff \
  --qml_reduction_method lda \
  --qml_feature_selection_method f_classif \
  --qml_feature_select_k_mult 5.5 \
  --qml_pre_pca_dim 0 \
  --qml_feature_map ZZ \
  --qml_feature_map_reps 3 \
  --qml_entanglement full \
  --negative_sampling hard \
  --neg_ratio 2.0 \
  --skip_svm_rbf \
  --skip_svm_linear \
  --skip_vqc \
  --random_state 42
