from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvidenceConfig:
    compound_gene_edges: Tuple[str, ...] = ("CbG", "CdG", "CuG")
    disease_gene_edges: Tuple[str, ...] = ("DaG", "DdG", "DuG")


@dataclass(frozen=True)
class EvidenceConfigDirectional:
    """Separate counts for up-regulation vs down-regulation for perturbation direction support."""
    compound_gene_up: Tuple[str, ...] = ("CuG",)
    compound_gene_down: Tuple[str, ...] = ("CdG",)
    disease_gene_up: Tuple[str, ...] = ("DuG",)
    disease_gene_down: Tuple[str, ...] = ("DdG",)


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


def build_directional_gene_maps(
    df_edges: pd.DataFrame,
    cfg: EvidenceConfigDirectional,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Build maps for up/down regulation:
      comp2g_up, comp2g_down, dis2g_up, dis2g_down
    using CuG/DuG (up) and CdG/DdG (down) metaedges.
    """
    df = df_edges[["source", "metaedge", "target"]].astype(str)
    comp2g_up: Dict[str, Set[str]] = {}
    comp2g_down: Dict[str, Set[str]] = {}
    dis2g_up: Dict[str, Set[str]] = {}
    dis2g_down: Dict[str, Set[str]] = {}

    cg_up = df[df["metaedge"].isin(cfg.compound_gene_up)]
    cg_down = df[df["metaedge"].isin(cfg.compound_gene_down)]
    dg_up = df[df["metaedge"].isin(cfg.disease_gene_up)]
    dg_down = df[df["metaedge"].isin(cfg.disease_gene_down)]

    for s, _, t in cg_up.itertuples(index=False):
        if isinstance(s, str) and isinstance(t, str) and s.startswith("Compound::") and t.startswith("Gene::"):
            comp2g_up.setdefault(s, set()).add(t)
    for s, _, t in cg_down.itertuples(index=False):
        if isinstance(s, str) and isinstance(t, str) and s.startswith("Compound::") and t.startswith("Gene::"):
            comp2g_down.setdefault(s, set()).add(t)
    for s, _, t in dg_up.itertuples(index=False):
        if isinstance(s, str) and isinstance(t, str) and s.startswith("Disease::") and t.startswith("Gene::"):
            dis2g_up.setdefault(s, set()).add(t)
    for s, _, t in dg_down.itertuples(index=False):
        if isinstance(s, str) and isinstance(t, str) and s.startswith("Disease::") and t.startswith("Gene::"):
            dis2g_down.setdefault(s, set()).add(t)

    return comp2g_up, comp2g_down, dis2g_up, dis2g_down


def shared_gene_count_by_direction(
    comp2g_up: Dict[str, Set[str]],
    comp2g_down: Dict[str, Set[str]],
    dis2g_up: Dict[str, Set[str]],
    dis2g_down: Dict[str, Set[str]],
    compound: str,
    disease: str,
) -> Tuple[int, int]:
    """
    Count shared genes by regulation direction (up vs down) for compound-disease pair.
    Returns (evidence_up, evidence_down) where:
      - evidence_up: genes shared via compound upregulation AND disease upregulation
      - evidence_down: genes shared via compound downregulation AND disease downregulation
    """
    cup = comp2g_up.get(str(compound), set())
    cdown = comp2g_down.get(str(compound), set())
    dup = dis2g_up.get(str(disease), set())
    ddown = dis2g_down.get(str(disease), set())

    up_count = sum(1 for g in cup if g in dup) + sum(1 for g in cdown if g in ddown)
    down_count = sum(1 for g in cup if g in ddown) + sum(1 for g in cdown if g in dup)

    return int(up_count), int(down_count)


def add_directional_evidence(
    df_pairs: pd.DataFrame,
    *,
    comp2g_up: Dict[str, Set[str]],
    comp2g_down: Dict[str, Set[str]],
    dis2g_up: Dict[str, Set[str]],
    dis2g_down: Dict[str, Set[str]],
    source_col: str = "source",
    target_col: str = "target",
    out_col_up: str = "evidence_up",
    out_col_down: str = "evidence_down",
    out_col_ratio: str = "evidence_balanced",
) -> pd.DataFrame:
    """
    Add directional evidence columns: evidence_up, evidence_down, evidence_balanced.
    evidence_balanced is up/(up+down+eps) or 0.5 if both zero (neutral).
    """
    out = df_pairs.copy()
    if source_col not in out.columns or target_col not in out.columns:
        return out

    ups = []
    downs = []
    ratios = []
    eps = 1e-8
    for c, d in zip(out[source_col].astype(str).tolist(), out[target_col].astype(str).tolist()):
        up, down = shared_gene_count_by_direction(
            comp2g_up, comp2g_down, dis2g_up, dis2g_down, c, d
        )
        ups.append(up)
        downs.append(down)
        total = up + down + eps
        ratios.append(up / total if total > eps else 0.5)

    out[out_col_up] = ups
    out[out_col_down] = downs
    out[out_col_ratio] = ratios
    return out

