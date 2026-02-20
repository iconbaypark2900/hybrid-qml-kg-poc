# benchmarking/negative_controls.py

"""
Generate negative control sets for hypothesis validation.
Random compounds/diseases and mock-mechanism candidates.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from kg_layer.hypothesis_graph import MechanismSubgraph

logger = logging.getLogger(__name__)


def get_random_controls(
    all_entities: List[str],
    mechanism_subgraph: MechanismSubgraph,
    entity_type: str = "Compound",
    n_controls: int = 50,
    random_state: Optional[int] = None,
) -> List[str]:
    """
    Sample random entities not in the mechanism subgraph.

    Args:
        all_entities: Full list of entity IDs (e.g., from task)
        mechanism_subgraph: Hypothesis mechanism subgraph
        entity_type: Filter by prefix (e.g., "Compound::", "Disease::")
        n_controls: Number of control entities to return
        random_state: Random seed

    Returns:
        List of entity IDs for negative controls
    """
    mechanism_nodes = mechanism_subgraph.nodes
    filtered = [e for e in all_entities if isinstance(e, str) and e.startswith(entity_type) and e not in mechanism_nodes]
    if not filtered:
        logger.warning("No entities outside mechanism subgraph for random controls")
        return []
    rng = np.random.default_rng(random_state)
    n = min(n_controls, len(filtered))
    return list(rng.choice(filtered, size=n, replace=False))


def get_mock_mechanism_controls(
    df_edges: pd.DataFrame,
    mechanism_subgraph: MechanismSubgraph,
    entity_type: str = "Compound",
    n_controls: int = 50,
    random_state: Optional[int] = None,
) -> List[str]:
    """
    Entities with similar degree to mechanism nodes but no biological link.

    Uses degree distribution; samples entities with comparable degree
    that are not in the mechanism subgraph.

    Args:
        df_edges: Full edges DataFrame (source, metaedge, target)
        mechanism_subgraph: Hypothesis mechanism subgraph
        entity_type: Filter by prefix
        n_controls: Number of controls
        random_state: Random seed

    Returns:
        List of entity IDs for mock-mechanism controls
    """
    mechanism_compounds = mechanism_subgraph.get_compound_ids()
    if not mechanism_compounds:
        mechanism_compounds = {n for n in mechanism_subgraph.nodes if isinstance(n, str) and n.startswith(entity_type)}

    # Degree per entity (out-degree from source)
    src_col = "source" if "source" in df_edges.columns else "source_id"
    degree = df_edges.groupby(src_col).size()
    mechanism_degrees = [degree.get(c, 0) for c in mechanism_compounds]
    mean_deg = np.mean(mechanism_degrees) if mechanism_degrees else 0
    std_deg = np.std(mechanism_degrees) if len(mechanism_degrees) > 1 else 1.0

    # Candidates: same entity type, not in mechanism, degree within 2 std of mean
    all_sources = set(df_edges[src_col].dropna().astype(str))
    candidates = [
        e for e in all_sources
        if e.startswith(entity_type)
        and e not in mechanism_subgraph.nodes
        and abs(degree.get(e, 0) - mean_deg) <= max(2 * std_deg, 1)
    ]
    if not candidates:
        candidates = [e for e in all_sources if e.startswith(entity_type) and e not in mechanism_subgraph.nodes]

    rng = np.random.default_rng(random_state)
    n = min(n_controls, len(candidates))
    return list(rng.choice(candidates, size=n, replace=False))


def get_non_lysosomal_genes(
    df_edges: pd.DataFrame,
    lysosomal_genes: Set[str],
    n_controls: int = 50,
    random_state: Optional[int] = None,
) -> List[str]:
    """
    Genes that are not in lysosomal pathways. Expected: directional score near random.

    Args:
        df_edges: Full edges with source, metaedge, target
        lysosomal_genes: Set of Gene::... IDs in lysosomal pathways
        n_controls: Number of control genes to return
        random_state: Random seed

    Returns:
        List of gene entity IDs not in lysosomal set
    """
    genes = set()
    for col in ["source", "target"]:
        if col in df_edges.columns:
            for v in df_edges[col].dropna().astype(str):
                if isinstance(v, str) and v.startswith("Gene::"):
                    genes.add(v)
    candidates = [g for g in genes if g not in lysosomal_genes]
    if not candidates:
        logger.warning("No non-lysosomal genes found for controls")
        return []
    rng = np.random.default_rng(random_state)
    n = min(n_controls, len(candidates))
    return list(rng.choice(candidates, size=n, replace=False))


def get_lipid_irrelevant_genes(
    df_edges: pd.DataFrame,
    lipid_related_genes: Optional[Set[str]] = None,
    n_controls: int = 50,
    random_state: Optional[int] = None,
) -> List[str]:
    """
    Placeholder: genes not in lipid-related pathways. Define lipid_related_genes when
    lipid ontology or pathway data is available. Expected: directional score near random.

    When lipid_related_genes is None, returns random genes (placeholder behavior).
    """
    genes = set()
    for col in ["source", "target"]:
        if col in df_edges.columns:
            for v in df_edges[col].dropna().astype(str):
                if isinstance(v, str) and v.startswith("Gene::"):
                    genes.add(v)
    if lipid_related_genes is not None:
        candidates = [g for g in genes if g not in lipid_related_genes]
    else:
        candidates = list(genes)
    if not candidates:
        return []
    rng = np.random.default_rng(random_state)
    n = min(n_controls, len(candidates))
    return list(rng.choice(candidates, size=n, replace=False))


def generate_control_pairs(
    controls: List[str],
    disease_ids: List[str],
    max_pairs: Optional[int] = None,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate (compound, disease) pairs for control evaluation.

    Args:
        controls: Compound entity IDs (negative controls)
        disease_ids: Disease entity IDs to pair with
        max_pairs: Cap on number of pairs (default: all combinations, capped at 500)
        random_state: Random seed

    Returns:
        DataFrame with columns source, target, label=0
    """
    rng = np.random.default_rng(random_state)
    pairs = [(c, d) for c in controls for d in disease_ids]
    if max_pairs and len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    return pd.DataFrame(pairs, columns=["source", "target"]).assign(label=0)
