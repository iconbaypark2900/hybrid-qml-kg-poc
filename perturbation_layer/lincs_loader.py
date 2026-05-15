from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_lincs_csv(
    path: str,
    compound_col: str = "pert_iname",
    gene_col: str = "gene_symbol",
    score_col: str = "zscore",
    dose_col: Optional[str] = "pert_dose",
    time_col: Optional[str] = "pert_time",
    cell_line_col: Optional[str] = "cell_id",
) -> pd.DataFrame:
    """
    Load a LINCS L1000 signature CSV (flat format).

    Expected columns: compound ID, gene symbol, z-score (or similar),
    optionally dose, timepoint, cell line.

    Returns a tidy DataFrame with columns:
      compound, gene, score, [dose], [timepoint], [cell_line]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"LINCS CSV not found: {p}")

    logger.info(f"Loading LINCS CSV: {p}")
    df = pd.read_csv(p, dtype=str)

    rename_map = {compound_col: "compound", gene_col: "gene", score_col: "score"}
    if dose_col and dose_col in df.columns:
        rename_map[dose_col] = "dose"
    if time_col and time_col in df.columns:
        rename_map[time_col] = "timepoint"
    if cell_line_col and cell_line_col in df.columns:
        rename_map[cell_line_col] = "cell_line"

    missing = [c for c in [compound_col, gene_col, score_col] if c not in df.columns]
    if missing:
        raise ValueError(f"LINCS CSV missing required columns: {missing}. Found: {list(df.columns)}")

    df = df.rename(columns=rename_map)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["compound", "gene", "score"])

    logger.info(
        f"Loaded {len(df)} LINCS rows: "
        f"{df['compound'].nunique()} compounds × {df['gene'].nunique()} genes"
    )
    return df


def load_lincs_gctx(
    path: str,
    sig_info_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a LINCS L1000 .gctx file (requires cmapPy).

    Falls back with a helpful error if cmapPy is not installed.
    """
    try:
        from cmapPy.pandasGEXpress.parse import parse
    except ImportError:
        raise ImportError(
            "cmapPy is required to load .gctx files. "
            "Install with: pip install cmapPy"
        )

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GCTX file not found: {p}")

    logger.info(f"Loading LINCS GCTX: {p}")
    gct = parse(str(p))
    df_wide = gct.data_df  # genes × signatures

    # Melt to tidy format
    df = df_wide.reset_index().melt(id_vars=df_wide.index.name or "gene", var_name="sig_id", value_name="score")
    df.columns = ["gene", "sig_id", "score"]

    if sig_info_path:
        sig_info = pd.read_csv(sig_info_path, sep="\t", dtype=str)
        df = df.merge(sig_info, on="sig_id", how="left")
        if "pert_iname" in df.columns:
            df = df.rename(columns={"pert_iname": "compound"})

    logger.info(f"GCTX loaded: {df['gene'].nunique()} genes, {df['sig_id'].nunique()} signatures")
    return df
