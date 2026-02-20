# kg_layer/hypothesis_graph.py

"""
Mechanism subgraphs and hypothesis-specific graph structures for CLN5-BMP-lysosomal
hypothesis testing.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class MechanismSubgraph:
    """
    Captures a hypothesis-specific mechanism subgraph with seed nodes,
    relation types, and optional pathway filters.
    """

    hypothesis_id: str
    seed_entities: List[str]
    metaedges: List[str]
    pathway_filter: Optional[List[str]] = None
    lipid_nodes: List[str] = field(default_factory=list)
    nodes: Set[str] = field(default_factory=set)
    edges: pd.DataFrame = field(default_factory=pd.DataFrame)

    def get_compound_ids(self) -> Set[str]:
        """Return compound entity IDs in the subgraph."""
        return {n for n in self.nodes if isinstance(n, str) and n.startswith("Compound::")}

    def get_disease_ids(self) -> Set[str]:
        """Return disease entity IDs in the subgraph."""
        return {n for n in self.nodes if isinstance(n, str) and n.startswith("Disease::")}

    def get_gene_ids(self) -> Set[str]:
        """Return gene entity IDs in the subgraph."""
        return {n for n in self.nodes if isinstance(n, str) and n.startswith("Gene::")}


def _bfs_expand_subgraph(
    df_edges: pd.DataFrame,
    seed_set: Set[str],
    max_depth: int,
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Expand subgraph via BFS from seed nodes, including nodes up to max_depth hops.
    Preserves edge types and direction.
    """
    if max_depth < 1:
        return pd.DataFrame(columns=["source", "metaedge", "target"]), set(seed_set)

    visited = set(seed_set)
    queue: deque = deque((s, 0) for s in seed_set)

    # Build adjacency
    neighbors: Dict[str, List[str]] = {}
    for _, row in df_edges.iterrows():
        s, t = row["source"], row["target"]
        neighbors.setdefault(s, []).append(t)
        neighbors.setdefault(t, []).append(s)

    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in neighbors.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    # Include edges where both endpoints are in visited set
    mask = df_edges["source"].isin(visited) & df_edges["target"].isin(visited)
    sub_edges = df_edges[mask].copy()
    return sub_edges, visited


def load_hypothesis_config(hypothesis_id: str, config_dir: str = "config/hypotheses") -> Dict:
    """
    Load hypothesis configuration from YAML file.

    Args:
        hypothesis_id: Hypothesis ID (e.g., H-001, H-002, H-003)
        config_dir: Directory containing hypothesis config files

    Returns:
        Configuration dictionary with keys: hypothesis_id, seed_entities,
        metaedges, pathway_filter, lipid_nodes, etc.
    """
    path = Path(config_dir) / f"{hypothesis_id}.yaml"
    if not path.exists():
        logger.warning(f"Hypothesis config not found at {path}, using minimal defaults")
        return {
            "hypothesis_id": hypothesis_id,
            "seed_entities": [],
            "metaedges": ["CdG", "CuG", "DdG", "DuG", "CbG", "GpPW"],
            "pathway_filter": None,
            "lipid_nodes": [],
            "max_depth": 2,
        }
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config if config else {}


def build_mechanism_subgraph(
    df_edges: pd.DataFrame,
    config: Union[Dict, str],
    hypothesis_id: Optional[str] = None,
    config_dir: str = "config/hypotheses",
) -> MechanismSubgraph:
    """
    Build a mechanism subgraph from Hetionet edges given a hypothesis config.

    Args:
        df_edges: Full Hetionet edge DataFrame with columns ['source', 'metaedge', 'target']
        config: Either a config dict or hypothesis_id string (e.g., 'H-001')
        hypothesis_id: Used when config is a dict, for subgraph ID
        config_dir: Directory for hypothesis configs when config is hypothesis_id

    Returns:
        MechanismSubgraph with nodes and edges
    """
    if isinstance(config, str):
        hypothesis_id = config
        config = load_hypothesis_config(config, config_dir)

    hid = hypothesis_id or config.get("hypothesis_id", "unknown")
    seed_entities: List[str] = config.get("seed_entities") or []
    metaedges: List[str] = config.get("metaedges") or ["CdG", "CuG", "DdG", "DuG", "CbG", "GpPW"]
    pathway_filter: Optional[List[str]] = config.get("pathway_filter")
    lipid_nodes: List[str] = config.get("lipid_nodes") or []
    max_depth: int = min(3, max(1, config.get("max_depth", 2)))

    df = df_edges[["source", "metaedge", "target"]].astype(str)
    metaedge_set = set(metaedges)
    df_allowed = df[df["metaedge"].isin(metaedge_set)]

    # Optional pathway filter first
    if pathway_filter:
        pathway_set = set(pathway_filter)
        df_allowed = df_allowed[
            df_allowed["source"].isin(pathway_set) | df_allowed["target"].isin(pathway_set)
        ]

    if seed_entities:
        seed_set = set(seed_entities)
        seed_set.update(lipid_nodes)
        sub_edges, nodes = _bfs_expand_subgraph(df_allowed, seed_set, max_depth)
    else:
        sub_edges = df_allowed.copy()
        nodes = set()
        nodes.update(sub_edges["source"].tolist())
        nodes.update(sub_edges["target"].tolist())

    nodes.update(seed_entities)
    nodes.update(lipid_nodes)

    return MechanismSubgraph(
        hypothesis_id=hid,
        seed_entities=seed_entities,
        metaedges=metaedges,
        pathway_filter=pathway_filter,
        lipid_nodes=lipid_nodes,
        nodes=nodes,
        edges=sub_edges,
    )


def extract_hypothesis_edges(
    df_edges: pd.DataFrame,
    hypothesis_id: str,
    config_dir: str = "config/hypotheses",
) -> Tuple[pd.DataFrame, MechanismSubgraph]:
    """
    Load hypothesis config and extract filtered edges for the hypothesis.

    Args:
        df_edges: Full Hetionet edge DataFrame
        hypothesis_id: Hypothesis ID (e.g., H-001)
        config_dir: Directory containing hypothesis config files

    Returns:
        (filtered_edges, mechanism_subgraph)
    """
    config = load_hypothesis_config(hypothesis_id, config_dir)
    subgraph = build_mechanism_subgraph(df_edges, config, hypothesis_id=hypothesis_id, config_dir=config_dir)
    return subgraph.edges, subgraph
