"""
Mechanism-of-Action (MoA) features for link prediction.

Extracts pharmacological plausibility signals from Hetionet multi-relational
structure. These features capture whether a compound-disease pair shares
mechanistic evidence (binding targets, pathway overlap, drug class membership,
chemical/disease similarity to known treatments).

Designed to recalibrate prediction scores by penalizing structurally plausible
but mechanistically implausible pairs (e.g., Abacavir → ocular cancer).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MOA_FEATURE_NAMES = [
    "moa_binding_targets",           # |CbG(compound)|
    "moa_disease_genes",             # |DaG(disease)|
    "moa_shared_targets",            # |CbG(compound) ∩ DaG(disease)|
    "moa_target_overlap",            # shared / union (Jaccard)
    "moa_shared_pathway_genes",      # shared targets in same pathway via GpPW
    "moa_pharmacologic_classes",     # |PCiC(compound)|
    "moa_compound_similarity",       # |CrC(compound)|
    "moa_similar_compounds_treat",   # |CrC(compound) ∩ CtD_compounds|
    "moa_disease_similarity",        # |DrD(disease)|
    "moa_similar_diseases_treated",  # |DrD(disease) ∩ CtD_diseases|
]


@dataclass
class MoAIndex:
    """Pre-built lookup tables for MoA feature computation."""
    comp2bind: Dict[str, Set[str]] = field(default_factory=dict)   # CbG
    dis2gene: Dict[str, Set[str]] = field(default_factory=dict)    # DaG
    gene2pathway: Dict[str, Set[str]] = field(default_factory=dict)  # GpPW
    comp2class: Dict[str, Set[str]] = field(default_factory=dict)  # PCiC
    comp2similar: Dict[str, Set[str]] = field(default_factory=dict)  # CrC
    dis2similar: Dict[str, Set[str]] = field(default_factory=dict)   # DrD
    treating_compounds: Set[str] = field(default_factory=set)        # compounds in CtD
    treated_diseases: Set[str] = field(default_factory=set)          # diseases in CtD


def build_moa_index(
    all_edges_df: pd.DataFrame,
    train_ctd_df: pd.DataFrame,
) -> MoAIndex:
    """
    Build MoA lookup tables from the full Hetionet edge set.

    Args:
        all_edges_df: Full Hetionet edges (source, metaedge, target).
        train_ctd_df: Training CtD edges only (for known treatment pairs).
                      Must have 'source' and 'target' columns with entity string IDs.

    Returns:
        MoAIndex with all lookup tables populated.
    """
    idx = MoAIndex()

    df = all_edges_df[["source", "metaedge", "target"]].astype(str)

    # CbG: Compound binds Gene
    cbg = df[df["metaedge"] == "CbG"]
    for s, _, t in cbg.itertuples(index=False):
        if s.startswith("Compound::") and t.startswith("Gene::"):
            idx.comp2bind.setdefault(s, set()).add(t)

    # DaG: Disease associates Gene
    dag = df[df["metaedge"] == "DaG"]
    for s, _, t in dag.itertuples(index=False):
        if s.startswith("Disease::") and t.startswith("Gene::"):
            idx.dis2gene.setdefault(s, set()).add(t)

    # GpPW: Gene participates Pathway
    gpw = df[df["metaedge"] == "GpPW"]
    for s, _, t in gpw.itertuples(index=False):
        if s.startswith("Gene::"):
            idx.gene2pathway.setdefault(s, set()).add(t)

    # PCiC: Pharmacologic Class includes Compound
    pcic = df[df["metaedge"] == "PCiC"]
    for s, _, t in pcic.itertuples(index=False):
        # PCiC: source is PharmClass, target is Compound
        if t.startswith("Compound::"):
            idx.comp2class.setdefault(t, set()).add(s)

    # CrC: Compound resembles Compound
    crc = df[df["metaedge"] == "CrC"]
    for s, _, t in crc.itertuples(index=False):
        if s.startswith("Compound::") and t.startswith("Compound::"):
            idx.comp2similar.setdefault(s, set()).add(t)
            idx.comp2similar.setdefault(t, set()).add(s)

    # DrD: Disease resembles Disease
    drd = df[df["metaedge"] == "DrD"]
    for s, _, t in drd.itertuples(index=False):
        if s.startswith("Disease::") and t.startswith("Disease::"):
            idx.dis2similar.setdefault(s, set()).add(t)
            idx.dis2similar.setdefault(t, set()).add(s)

    # Known treating compounds/diseases from TRAINING CtD edges only
    for _, row in train_ctd_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        if src.startswith("Compound::"):
            idx.treating_compounds.add(src)
        if tgt.startswith("Disease::"):
            idx.treated_diseases.add(tgt)
        # Handle reversed columns
        if tgt.startswith("Compound::"):
            idx.treating_compounds.add(tgt)
        if src.startswith("Disease::"):
            idx.treated_diseases.add(src)

    logger.info(
        f"MoA index built: "
        f"{len(idx.comp2bind)} compounds with binding targets, "
        f"{len(idx.dis2gene)} diseases with gene associations, "
        f"{len(idx.gene2pathway)} genes with pathways, "
        f"{len(idx.comp2class)} compounds with pharma classes, "
        f"{len(idx.comp2similar)} compounds with similarity edges, "
        f"{len(idx.dis2similar)} diseases with similarity edges, "
        f"{len(idx.treating_compounds)} treating compounds, "
        f"{len(idx.treated_diseases)} treated diseases"
    )

    return idx


def compute_moa_features(
    compound_id: str,
    disease_id: str,
    moa: MoAIndex,
) -> np.ndarray:
    """
    Compute MoA feature vector for a single compound-disease pair.

    Returns:
        np.ndarray of shape (10,) with float32 features.
    """
    # Identify which is compound and which is disease
    comp = compound_id if compound_id.startswith("Compound::") else disease_id
    dis = disease_id if disease_id.startswith("Disease::") else compound_id

    # If neither is a compound or disease, return zeros
    if not comp.startswith("Compound::") or not dis.startswith("Disease::"):
        return np.zeros(len(MOA_FEATURE_NAMES), dtype=np.float32)

    bind_genes = moa.comp2bind.get(comp, set())
    dis_genes = moa.dis2gene.get(dis, set())

    # 1. Binding targets count
    n_bind = len(bind_genes)

    # 2. Disease gene associations count
    n_dis_genes = len(dis_genes)

    # 3. Shared binding targets (compound binds gene AND disease associates gene)
    shared = bind_genes & dis_genes
    n_shared = len(shared)

    # 4. Target overlap (Jaccard)
    union = bind_genes | dis_genes
    overlap = n_shared / len(union) if union else 0.0

    # 5. Shared pathway genes: how many shared genes share at least one pathway
    n_pathway_shared = 0
    if shared:
        for g in shared:
            if g in moa.gene2pathway:
                n_pathway_shared += 1

    # 6. Pharmacologic classes
    n_classes = len(moa.comp2class.get(comp, set()))

    # 7. Compound similarity count
    similar_comps = moa.comp2similar.get(comp, set())
    n_similar = len(similar_comps)

    # 8. Similar compounds that treat any disease
    n_similar_treat = len(similar_comps & moa.treating_compounds)

    # 9. Disease similarity count
    similar_dis = moa.dis2similar.get(dis, set())
    n_dis_similar = len(similar_dis)

    # 10. Similar diseases that are treated by any compound
    n_similar_treated = len(similar_dis & moa.treated_diseases)

    return np.array([
        n_bind,
        n_dis_genes,
        n_shared,
        overlap,
        n_pathway_shared,
        n_classes,
        n_similar,
        n_similar_treat,
        n_dis_similar,
        n_similar_treated,
    ], dtype=np.float32)


def build_moa_features_batch(
    links_df: pd.DataFrame,
    moa: MoAIndex,
    source_col: str = "source",
    target_col: str = "target",
) -> np.ndarray:
    """
    Compute MoA features for all rows in a links DataFrame.

    Args:
        links_df: DataFrame with source/target entity string ID columns.
        moa: Pre-built MoA index.
        source_col: Column name for source entity.
        target_col: Column name for target entity.

    Returns:
        np.ndarray of shape (N, 10) with float32 features.
    """
    results = []
    for _, row in links_df.iterrows():
        feats = compute_moa_features(str(row[source_col]), str(row[target_col]), moa)
        results.append(feats)
    return np.array(results, dtype=np.float32)
