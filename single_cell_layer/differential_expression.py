from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def run_de(
    adata,
    condition_key: str = "condition",
    disease_val: str = "disease",
    control_val: str = "control",
    cell_type_key: Optional[str] = "cell_type",
    n_genes: int = 500,
) -> pd.DataFrame:
    """
    Wilcoxon rank-sum differential expression: disease vs. control.

    If cell_type_key is set, DE is run per cell type and results are
    concatenated with a 'cell_type' column.

    Returns a tidy DataFrame with columns:
      gene, logfoldchanges, pvals_adj, scores, cell_type (if applicable)
    """
    try:
        import scanpy as sc
    except ImportError:
        raise ImportError("scanpy required. Run: pip install -r requirements-omics.txt")

    if condition_key not in adata.obs.columns:
        raise ValueError(
            f"Condition column '{condition_key}' not in adata.obs. "
            "Add disease/control labels before running DE."
        )

    results: List[pd.DataFrame] = []

    def _run_for_subset(sub_adata, label: str) -> Optional[pd.DataFrame]:
        if sub_adata.n_obs < 5:
            logger.warning(f"Too few cells for DE in '{label}'; skipping.")
            return None
        sc.tl.rank_genes_groups(
            sub_adata,
            groupby=condition_key,
            groups=[disease_val],
            reference=control_val,
            n_genes=n_genes,
            method="wilcoxon",
        )
        df = sc.get.rank_genes_groups_df(sub_adata, group=disease_val)
        df["cell_type"] = label
        return df

    if cell_type_key and cell_type_key in adata.obs.columns:
        for ct in adata.obs[cell_type_key].unique():
            subset = adata[adata.obs[cell_type_key] == ct].copy()
            df = _run_for_subset(subset, ct)
            if df is not None:
                results.append(df)
    else:
        df = _run_for_subset(adata, "all_cells")
        if df is not None:
            results.append(df)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    combined = combined.sort_values("scores", ascending=False)
    logger.info(
        f"DE complete: {len(combined)} gene-cell_type rows across "
        f"{combined.get('cell_type', pd.Series()).nunique()} cell types"
    )
    return combined
