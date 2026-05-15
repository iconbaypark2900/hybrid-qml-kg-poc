from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_cmap_csv(
    path: str,
    compound_col: str = "drug_name",
    gene_col: str = "gene_symbol",
    score_col: str = "log2_fc",
    direction_col: Optional[str] = "direction",
    cell_line_col: Optional[str] = "cell_line",
) -> pd.DataFrame:
    """
    Load a Connectivity Map (CMap) style signature CSV.

    CMap signatures predate LINCS L1000 and use slightly different conventions:
      - log2 fold change instead of z-score
      - "direction" column ("up" / "down") sometimes provided alongside the score
      - Older cell line nomenclature

    Returns a tidy DataFrame with columns:
      compound, gene, score, [direction], [cell_line]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CMap CSV not found: {p}")

    logger.info(f"Loading CMap CSV: {p}")
    df = pd.read_csv(p, dtype=str)

    missing = [c for c in [compound_col, gene_col, score_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"CMap CSV missing required columns: {missing}. Found: {list(df.columns)}"
        )

    rename_map = {compound_col: "compound", gene_col: "gene", score_col: "score"}
    if direction_col and direction_col in df.columns:
        rename_map[direction_col] = "direction"
    if cell_line_col and cell_line_col in df.columns:
        rename_map[cell_line_col] = "cell_line"

    df = df.rename(columns=rename_map)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["compound", "gene", "score"])

    logger.info(
        f"Loaded {len(df)} CMap rows: "
        f"{df['compound'].nunique()} compounds × {df['gene'].nunique()} genes"
    )
    return df


def load_cmap_rank_matrix(
    path: str,
    gene_col: str = "gene_symbol",
) -> pd.DataFrame:
    """
    Load a CMap rank-ordered gene expression matrix.

    The legacy CMap build 02 format is a wide matrix: rows = genes, columns =
    instance IDs (compound × cell-line × dose × timepoint). Values are ranks
    of differential expression magnitude.

    Returns a wide DataFrame indexed by gene with one column per signature.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CMap rank matrix not found: {p}")

    logger.info(f"Loading CMap rank matrix: {p}")
    # CMap rank matrices are usually .gct or .txt with a header line
    if p.suffix.lower() == ".gct":
        df = pd.read_csv(p, sep="\t", skiprows=2)
    else:
        df = pd.read_csv(p, sep="\t")

    if gene_col not in df.columns:
        # Fallback: assume the first column is the gene identifier
        gene_col = df.columns[0]

    df = df.set_index(gene_col)
    logger.info(f"CMap rank matrix loaded: {df.shape[0]} genes × {df.shape[1]} signatures")
    return df


def cmap_to_up_down(
    df: pd.DataFrame,
    top_n: int = 150,
    compound_col: str = "compound",
    gene_col: str = "gene",
    score_col: str = "score",
) -> Dict[str, Dict[str, List[str]]]:
    """
    Convert a tidy CMap signature DataFrame into per-compound up/down gene lists.

    For each compound, takes the top-N genes by score (up-regulated) and the
    bottom-N (down-regulated). Suitable as input to
    perturbation_layer.reversal_score.compute_reversal_score.

    Returns:
        {compound: {"up_genes": [...], "down_genes": [...]}}
    """
    if compound_col not in df.columns or gene_col not in df.columns or score_col not in df.columns:
        raise ValueError(
            f"DataFrame missing required columns {compound_col}/{gene_col}/{score_col}; "
            f"found: {list(df.columns)}"
        )

    out: Dict[str, Dict[str, List[str]]] = {}
    for compound, sub in df.groupby(compound_col):
        sorted_sub = sub.sort_values(score_col, ascending=False)
        up = sorted_sub.head(top_n)[gene_col].astype(str).tolist()
        down = sorted_sub.tail(top_n)[gene_col].astype(str).tolist()
        out[str(compound)] = {"up_genes": up, "down_genes": down}

    logger.info(f"Built up/down gene lists for {len(out)} compounds (top_n={top_n}).")
    return out
