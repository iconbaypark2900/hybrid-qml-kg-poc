#!/bin/bash
# Run pipeline with best configurations from parameter exploration
#
# QUANTUM (Best Config):
#   Test PR-AUC: 0.9510, CV PR-AUC: 0.9771, Gap: 0.0450
#   Config: hybrid encoding, PCA reduction, mutual_info selection, pre_pca_dim 128
#
# CLASSICAL (Best Configs - GridSearchCV will find these automatically):
#   RandomForest: 
#     Grid search: n_estimators=[100,200,300], max_depth=[8,10,12,15], min_samples_split=[5,10,20]
#     Best from exploration: n_estimators=300, max_depth=8, min_samples_split=5
#     Fixed: min_samples_leaf=5, max_features='sqrt' (pipeline uses 5, exploration found 3)
#     Expected PR-AUC: ~0.6860
#   LogisticRegression:
#     Grid search: C=[0.01,0.1,0.3,1.0,3.0,10.0,30.0]
#     Best from exploration: C=0.01
#     Expected PR-AUC: ~0.8437
#
# Note: GridSearchCV will automatically find the best parameters from the grids above.
#       The best configs from exploration are already included in the search grids.

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
  --qml_encoding hybrid \
  --qml_reduction_method pca \
  --qml_feature_selection_method mutual_info \
  --qml_feature_select_k_mult 2.0 \
  --qml_pre_pca_dim 128 \
  --qml_feature_map ZZ \
  --qml_feature_map_reps 2 \
  --qml_entanglement full \
  --negative_sampling hard \
  --neg_ratio 2.0 \
  --skip_svm_rbf \
  --skip_svm_linear \
  --skip_vqc \
  --random_state 42
