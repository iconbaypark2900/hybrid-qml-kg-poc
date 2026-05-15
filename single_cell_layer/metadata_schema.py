from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Required obs (cell-level) columns for downstream pipeline compatibility.
# Keys match config single_cell_config.yaml keys.
REQUIRED_OBS_KEYS = ["condition"]         # disease | control label
RECOMMENDED_OBS_KEYS = ["cell_type", "batch_key", "tissue"]

# Valid values for condition column
CONDITION_DISEASE = "disease"
CONDITION_CONTROL = "control"
VALID_CONDITIONS = {CONDITION_DISEASE, CONDITION_CONTROL}


def validate_anndata(adata, config: dict) -> List[str]:
    """
    Validate that an AnnData object satisfies the metadata contract.

    Returns a list of warning strings (empty = passes).
    Does not raise; callers decide whether to abort on warnings.
    """
    warnings: List[str] = []

    condition_key = config.get("input", {}).get("condition_key", "condition")
    batch_key = config.get("input", {}).get("batch_key", "sample_id")
    cell_type_key = config.get("input", {}).get("cell_type_key", "cell_type")

    # Check condition column exists
    if condition_key not in adata.obs.columns:
        warnings.append(
            f"obs column '{condition_key}' not found. "
            f"Add a '{condition_key}' column with values 'disease' / 'control'."
        )
    else:
        unique_vals = set(adata.obs[condition_key].dropna().unique())
        unexpected = unique_vals - VALID_CONDITIONS
        if unexpected:
            warnings.append(
                f"Unexpected values in '{condition_key}': {unexpected}. "
                f"Expected a subset of {VALID_CONDITIONS}."
            )

    # Check batch column (recommended, not required)
    if batch_key not in adata.obs.columns:
        warnings.append(
            f"obs column '{batch_key}' not found. "
            "Batch correction will be skipped."
        )

    # Check cell type column (recommended)
    if cell_type_key not in adata.obs.columns:
        warnings.append(
            f"obs column '{cell_type_key}' not found. "
            "Cell-type-specific signatures will be unavailable."
        )

    if not warnings:
        logger.info("AnnData metadata schema validation passed.")
    else:
        for w in warnings:
            logger.warning(f"Schema warning: {w}")

    return warnings
