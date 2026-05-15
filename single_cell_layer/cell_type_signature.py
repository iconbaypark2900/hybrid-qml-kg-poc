from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from single_cell_layer.disease_signature import build_disease_signature

logger = logging.getLogger(__name__)


def stratify_de_by_cell_type(
    de_df: pd.DataFrame,
    cell_type_col: str = "cell_type",
) -> Dict[str, pd.DataFrame]:
    """
    Split a differential-expression DataFrame into one DataFrame per cell type.

    If the column does not exist, returns {"all_cells": de_df} so downstream
    code can stay generic.
    """
    if cell_type_col not in de_df.columns:
        logger.info(f"No '{cell_type_col}' column found; treating as single cohort.")
        return {"all_cells": de_df}

    groups: Dict[str, pd.DataFrame] = {}
    for ct, sub in de_df.groupby(cell_type_col):
        groups[str(ct)] = sub.reset_index(drop=True)
    logger.info(f"Stratified DE into {len(groups)} cell types.")
    return groups


def build_per_cell_type_signatures(
    de_df: pd.DataFrame,
    disease_id: str,
    tissue: str = "",
    cell_type_col: str = "cell_type",
    lfc_threshold: float = 0.5,
    padj_threshold: float = 0.05,
    top_n: int = 250,
    min_cells_per_type: int = 30,
) -> Dict[str, Dict]:
    """
    Build a disease signature for every cell type with sufficient evidence.

    Cell types with fewer than `min_cells_per_type` rows in the DE table are
    skipped — too few cells leads to unreliable signatures.

    Returns:
        {cell_type: signature_dict}  where signature_dict matches the schema
        from build_disease_signature.
    """
    groups = stratify_de_by_cell_type(de_df, cell_type_col=cell_type_col)
    signatures: Dict[str, Dict] = {}
    skipped: List[str] = []

    for ct, sub in groups.items():
        if len(sub) < min_cells_per_type:
            skipped.append(f"{ct} (n={len(sub)})")
            continue
        sig = build_disease_signature(
            sub,
            disease_id=disease_id,
            tissue=tissue,
            cell_type=ct,
            lfc_threshold=lfc_threshold,
            padj_threshold=padj_threshold,
            top_n=top_n,
        )
        signatures[ct] = sig

    if skipped:
        logger.info(f"Skipped cell types below min_cells_per_type: {skipped}")
    logger.info(f"Built signatures for {len(signatures)} cell types.")
    return signatures


def summarize_signature_overlap(
    signatures: Dict[str, Dict],
) -> pd.DataFrame:
    """
    Pairwise Jaccard overlap of up-gene sets across cell types.

    Useful for deciding which cell types carry distinct disease biology versus
    redundant signal. Returns a square DataFrame of Jaccard coefficients.
    """
    cell_types = list(signatures.keys())
    n = len(cell_types)
    rows: List[List[float]] = []
    for i, ct_i in enumerate(cell_types):
        up_i = set(signatures[ct_i].get("up_genes", []))
        row: List[float] = []
        for j, ct_j in enumerate(cell_types):
            up_j = set(signatures[ct_j].get("up_genes", []))
            union = up_i | up_j
            jacc = len(up_i & up_j) / len(union) if union else 0.0
            row.append(round(jacc, 3))
        rows.append(row)

    return pd.DataFrame(rows, index=cell_types, columns=cell_types)


def consensus_signature(
    signatures: Dict[str, Dict],
    min_cell_types: int = 2,
) -> Tuple[List[str], List[str]]:
    """
    Build a consensus up/down gene list — genes that appear in at least
    `min_cell_types` of the per-cell-type signatures.

    This is the conservative, cross-cell-type signal that is most likely to
    represent shared disease biology.

    Returns (consensus_up, consensus_down).
    """
    up_counts: Dict[str, int] = {}
    dn_counts: Dict[str, int] = {}

    for sig in signatures.values():
        for g in sig.get("up_genes", []):
            up_counts[g] = up_counts.get(g, 0) + 1
        for g in sig.get("down_genes", []):
            dn_counts[g] = dn_counts.get(g, 0) + 1

    consensus_up = sorted([g for g, c in up_counts.items() if c >= min_cell_types])
    consensus_down = sorted([g for g, c in dn_counts.items() if c >= min_cell_types])

    logger.info(
        f"Consensus signature: {len(consensus_up)} up, {len(consensus_down)} down "
        f"(min_cell_types={min_cell_types})"
    )
    return consensus_up, consensus_down
