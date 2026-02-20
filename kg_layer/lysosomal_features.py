# kg_layer/lysosomal_features.py

"""
Lysosomal feature block for H-002 hypothesis (lysosomal dysfunction mediation).
Uses GpPW-based pathway membership as proxy when assay data is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Placeholder: lysosomal pathway IDs when available (e.g., Reactome, WikiPathways)
# Populate via config when pathway data is integrated
DEFAULT_LYSOSOMAL_PATHWAY_IDS: Set[str] = set()


def load_lysosomal_assays(data_dir: str = "data/lysosomal_assays") -> Optional[pd.DataFrame]:
    """
    Load lysosomal assay readouts from CSV if present.

    Expected schema: entity_id, readout, value
    Returns None if directory or files do not exist.
    """
    path = Path(data_dir)
    if not path.exists():
        return None
    files = list(path.glob("*.csv"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if {"entity_id", "readout", "value"}.issubset(df.columns):
                dfs.append(df)
        except Exception as e:
            logger.warning(f"Could not load {f}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def build_lysosomal_gene_set(
    df_edges: pd.DataFrame,
    pathway_filter: Optional[List[str]] = None,
) -> Set[str]:
    """
    Build set of genes in lysosomal-related pathways via GpPW.

    When pathway_filter is None, uses all GpPW genes (broad proxy).
    When pathway IDs are provided, filters to those pathways.

    Args:
        df_edges: Full edges with source, metaedge, target
        pathway_filter: Lysosomal pathway IDs (e.g., Pathway::...)

    Returns:
        Set of Gene::... entity IDs
    """
    df = df_edges[["source", "metaedge", "target"]].astype(str)
    gpw = df[df["metaedge"] == "GpPW"]
    genes: Set[str] = set()
    for _, row in gpw.iterrows():
        s, t = row["source"], row["target"]
        if pathway_filter:
            pathway_ids = set(pathway_filter)
            if s.startswith("Gene::") and t.startswith("Pathway::") and t in pathway_ids:
                genes.add(s)
            elif t.startswith("Gene::") and s.startswith("Pathway::") and s in pathway_ids:
                genes.add(t)
        else:
            if s.startswith("Gene::"):
                genes.add(s)
            if t.startswith("Gene::"):
                genes.add(t)
    return genes


def build_lysosomal_features(
    df_pairs: pd.DataFrame,
    df_edges: pd.DataFrame,
    comp2g: Dict[str, Set[str]],
    dis2g: Dict[str, Set[str]],
    lysosomal_genes: Optional[Set[str]] = None,
    pathway_filter: Optional[List[str]] = None,
    assay_df: Optional[pd.DataFrame] = None,
    source_col: str = "source",
    target_col: str = "target",
) -> np.ndarray:
    """
    Build lysosomal feature block for compound-disease pairs.

    Features (when assays absent):
      - lysosomal_shared: count of shared genes that are in lysosomal pathways
      - lysosomal_compound: count of compound's genes in lysosomal pathways
      - lysosomal_disease: count of disease's genes in lysosomal pathways
      - lysosomal_overlap_ratio: shared / (min(c,d) + 1)

    When assay_df is provided (future): adds assay-based readout aggregates.

    Args:
        df_pairs: Compound-disease pairs
        df_edges: Full edges for GpPW
        comp2g, dis2g: Compound->genes, Disease->genes maps
        lysosomal_genes: Precomputed lysosomal gene set (or built from df_edges)
        pathway_filter: For building lysosomal_genes
        assay_df: Optional assay readouts
        source_col, target_col: Column names

    Returns:
        Feature matrix of shape (len(df_pairs), n_features)
    """
    if lysosomal_genes is None:
        lysosomal_genes = build_lysosomal_gene_set(df_edges, pathway_filter)

    features = []
    for _, row in df_pairs.iterrows():
        c = str(row[source_col])
        d = str(row[target_col])
        c_genes = comp2g.get(c, set())
        d_genes = dis2g.get(d, set())
        c_lyso = c_genes & lysosomal_genes
        d_lyso = d_genes & lysosomal_genes
        shared_lyso = c_lyso & d_lyso
        lysosomal_shared = len(shared_lyso)
        lysosomal_compound = len(c_lyso)
        lysosomal_disease = len(d_lyso)
        denom = min(len(c_lyso), len(d_lyso)) + 1
        lysosomal_overlap_ratio = lysosomal_shared / denom
        features.append([
            lysosomal_shared,
            lysosomal_compound,
            lysosomal_disease,
            lysosomal_overlap_ratio,
        ])

    arr = np.array(features, dtype=np.float32)
    if assay_df is not None and len(assay_df) > 0:
        # Placeholder: could add assay-derived features (e.g., mean readout per entity)
        pass
    return arr


def get_lysosomal_feature_names() -> List[str]:
    """Return names for the lysosomal feature block."""
    return [
        "lysosomal_shared",
        "lysosomal_compound",
        "lysosomal_disease",
        "lysosomal_overlap_ratio",
    ]
