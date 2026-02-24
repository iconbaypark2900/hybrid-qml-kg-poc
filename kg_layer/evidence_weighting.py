from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvidenceConfig:
    compound_gene_edges: Tuple[str, ...] = ("CbG", "CdG", "CuG")
    disease_gene_edges: Tuple[str, ...] = ("DaG", "DdG", "DuG")


def build_compound_disease_gene_maps(df_edges: pd.DataFrame, cfg: EvidenceConfig) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build maps:
      compound_id -> set(genes)
      disease_id  -> set(genes)
    using Hetionet metaedges (CbG/CdG/CuG and DaG/DdG/DuG).
    """
    df = df_edges[["source", "metaedge", "target"]].astype(str)
    cg = df[df["metaedge"].isin(cfg.compound_gene_edges)]
    dg = df[df["metaedge"].isin(cfg.disease_gene_edges)]

    comp2g: Dict[str, Set[str]] = {}
    dis2g: Dict[str, Set[str]] = {}
    for s, _, t in cg.itertuples(index=False):
        if not isinstance(s, str) or not isinstance(t, str):
            continue
        # expect Compound::... -> Gene::...
        if s.startswith("Compound::") and t.startswith("Gene::"):
            comp2g.setdefault(s, set()).add(t)
    for s, _, t in dg.itertuples(index=False):
        if not isinstance(s, str) or not isinstance(t, str):
            continue
        # expect Disease::... -> Gene::...
        if s.startswith("Disease::") and t.startswith("Gene::"):
            dis2g.setdefault(s, set()).add(t)
    return comp2g, dis2g


def shared_gene_count(comp2g: Dict[str, Set[str]], dis2g: Dict[str, Set[str]], compound: str, disease: str) -> int:
    gs = comp2g.get(str(compound))
    gd = dis2g.get(str(disease))
    if not gs or not gd:
        return 0
    # iterate smaller set
    if len(gs) > len(gd):
        gs, gd = gd, gs
    return int(sum(1 for g in gs if g in gd))


def add_shared_gene_evidence(
    df_pairs: pd.DataFrame,
    *,
    comp2g: Dict[str, Set[str]],
    dis2g: Dict[str, Set[str]],
    source_col: str = "source",
    target_col: str = "target",
    out_col: str = "evidence_shared_genes",
) -> pd.DataFrame:
    out = df_pairs.copy()
    if source_col not in out.columns or target_col not in out.columns:
        return out
    out[out_col] = [
        shared_gene_count(comp2g, dis2g, c, d)
        for c, d in zip(out[source_col].astype(str).tolist(), out[target_col].astype(str).tolist())
    ]
    return out


def weights_from_shared_genes(shared: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    shared = np.asarray(shared).astype(float)
    # 1 + alpha * log1p(shared)
    return 1.0 + float(alpha) * np.log1p(shared)

