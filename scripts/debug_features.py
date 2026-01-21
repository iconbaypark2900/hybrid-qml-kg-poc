#!/usr/bin/env python3
"""Debug feature building issues"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder
from kg_layer.enhanced_features import EnhancedFeatureBuilder

# Load data
df = load_hetionet_edges()
task_edges, entity_to_id, id_to_entity = extract_task_edges(df, relation_type="CtD")
train_df, test_df = prepare_link_prediction_dataset(task_edges)

# Load embeddings
embedder = HetionetEmbedder(embedding_dim=32, qml_dim=5)
embedder.load_saved_embeddings()

# Get embeddings dict
embeddings = {}
for ent_id, idx in embedder.entity_to_id.items():
    embeddings[ent_id] = embedder.entity_embeddings[idx]

print(f"Loaded {len(embeddings)} embeddings")
print(f"Sample embedding keys: {list(embeddings.keys())[:5]}")

# Check train_df columns and sample data
print(f"\nTrain DF columns: {train_df.columns.tolist()}")
print(f"Train DF shape: {train_df.shape}")
print(f"\nSample rows:")
print(train_df.head())

# Convert IDs to entity names
train_df_names = train_df.copy()
if 'source_id' in train_df.columns:
    train_df_names['source'] = train_df['source_id'].map(id_to_entity)
if 'target_id' in train_df.columns:
    train_df_names['target'] = train_df['target_id'].map(id_to_entity)

print(f"\nAfter mapping:")
print(train_df_names.head())
print(f"NaN in source: {train_df_names['source'].isna().sum()}")
print(f"NaN in target: {train_df_names['target'].isna().sum()}")

# Check if entity names match embeddings
sample_source = str(train_df_names['source'].iloc[0])
sample_target = str(train_df_names['target'].iloc[0])
print(f"\nSample source: {sample_source}")
print(f"Sample target: {sample_target}")
print(f"Source in embeddings: {sample_source in embeddings}")
print(f"Target in embeddings: {sample_target in embeddings}")

# Try building features
feature_builder = EnhancedFeatureBuilder(
    include_graph_features=True,
    include_domain_features=True,
    normalize=True
)

feature_builder.build_graph(task_edges)

print("\nBuilding features for first 5 rows...")
try:
    X_sample, feat_names = feature_builder.build_features(
        train_df_names.head(), embeddings, task_edges
    )
    print(f"Features shape: {X_sample.shape}")
    print(f"Features stats:")
    print(f"  Min: {X_sample.min()}")
    print(f"  Max: {X_sample.max()}")
    print(f"  Mean: {X_sample.mean()}")
    print(f"  NaN count: {np.isnan(X_sample).sum()}")
    print(f"  Inf count: {np.isinf(X_sample).sum()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
