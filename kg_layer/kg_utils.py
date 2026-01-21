# kg_layer/kg_utils.py

"""
Utility functions for knowledge graph processing.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_kg_consistency(df_edges: pd.DataFrame) -> bool:
    """
    Validate that the KG edge list is consistent.

    Args:
        df_edges (pd.DataFrame): The full Hetionet edge DataFrame, expected to have
                                 'source', 'metaedge', and 'target' columns.

    Returns:
        bool: True if KG is consistent, False otherwise
    """
    # Check for missing values
    if df_edges.isnull().any().any():
        logger.warning("KG contains null values")
        return False

    # Check for duplicate edges
    duplicates = df_edges.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"KG contains {duplicates} duplicate edges")

    return True

def get_entity_statistics(df_edges: pd.DataFrame) -> dict:
    """
    Get basic statistics about entities in the KG.

    Args:
        df_edges (pd.DataFrame): The full Hetionet edge DataFrame, expected to have
                                 'source', 'metaedge', and 'target' columns.

    Returns:
        stats: Dictionary with entity statistics
    """
    all_entities = pd.concat([df_edges["source"], df_edges["target"]])
    entity_types = all_entities.apply(lambda x: x.split("::")[0] if "::" in x else "Unknown")

    stats = {
        "total_entities": len(all_entities.unique()),
        "total_edges": len(df_edges),
        "entity_types": entity_types.value_counts().to_dict(),
        "relation_types": df_edges["metaedge"].value_counts().to_dict()
    }

    return stats

def sample_balanced_dataset(
    df_edges: pd.DataFrame,
    relation_type: str,
    num_positive: int,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample a balanced dataset for a specific relation type.

    Args:
        df_edges (pd.DataFrame): The full Hetionet edge DataFrame, expected to have
                                 'source', 'metaedge', and 'target' columns.
        relation_type (str): The metaedge abbreviation to filter for (e.g., 'CtD').
                             Defaults to "CtD".
        num_positive: The number of positive samples to generate.
        random_state: The random state for reproducibility.

    Returns:
        sampled_positive: DataFrame with sampled positive edges
    """
    positive_edges = df_edges[df_edges["metaedge"] == relation_type]

    if len(positive_edges) < num_positive:
        logger.warning(f"Requested {num_positive} positive samples but only {len(positive_edges)} available")
        sampled_positive = positive_edges
    else:
        sampled_positive = positive_edges.sample(n=num_positive, random_state=random_state)

    return sampled_positive

def create_entity_id_mapping(entities: List[str]) -> Tuple[dict, dict]:
    """
    Create bidirectional mapping between entities and integer IDs.

    Args:
        entities: List of unique entity identifiers

    Returns:
        entity_to_id: Mapping from entity to integer ID
        id_to_entity: Reverse mapping from integer ID to entity
    """
    entity_to_id = {entity: idx for idx, entity in enumerate(sorted(entities))}
    id_to_entity = {idx: entity for entity, idx in entity_to_id.items()}
    return entity_to_id, id_to_entity