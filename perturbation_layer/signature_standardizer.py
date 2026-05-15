from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def standardize_gene_symbols(
    df: pd.DataFrame,
    gene_col: str = "gene",
    gene_mapper=None,
) -> pd.DataFrame:
    """
    Normalise gene symbols in a perturbation DataFrame to HGNC upper-case.

    If gene_mapper (GeneMapper) is provided, maps symbols to Hetionet Gene
    node IDs and drops genes that cannot be resolved.
    """
    df = df.copy()
    df[gene_col] = df[gene_col].str.strip().str.upper()

    if gene_mapper is not None:
        before = len(df)
        resolvable = set(gene_mapper.filter_resolved(df[gene_col].unique().tolist()))
        df = df[df[gene_col].isin(resolvable)]
        dropped = before - len(df)
        if dropped:
            logger.info(f"Gene normalisation: {dropped} rows dropped (unresolvable symbols)")

    df = df.drop_duplicates(subset=[gene_col])
    return df


def standardize_compound_ids(
    df: pd.DataFrame,
    compound_col: str = "compound",
    compound_mapper=None,
) -> pd.DataFrame:
    """
    Normalise compound identifiers and optionally filter to Hetionet-mapped compounds.
    """
    df = df.copy()
    df[compound_col] = df[compound_col].str.strip()

    if compound_mapper is not None:
        mapping = compound_mapper.map_many(df[compound_col].unique().tolist())
        df["compound_hetionet_id"] = df[compound_col].map(mapping)
        n_unmapped = df["compound_hetionet_id"].isna().sum()
        if n_unmapped:
            logger.info(
                f"Compound normalisation: {n_unmapped} rows have no Hetionet ID "
                "(kept; hetionet_id = NaN)"
            )

    return df
