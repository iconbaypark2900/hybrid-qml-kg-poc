from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path("scripts/harmonize_rnaseq_cohorts.py")
SPEC = importlib.util.spec_from_file_location("harmonize_rnaseq_cohorts", SCRIPT_PATH)
harmonize = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["harmonize_rnaseq_cohorts"] = harmonize
SPEC.loader.exec_module(harmonize)


def test_harmonization_uses_identical_ordered_gene_universe_and_target_sum(tmp_path: Path) -> None:
    development_path = tmp_path / "development.csv"
    validation_path = tmp_path / "validation.csv"
    pd.DataFrame(
        {
            "gene": ["ENSG1.1", "ENSG2.2", "ENSG3.1"],
            "dev_a": [10, 20, 70],
            "dev_b": [20, 30, 50],
        }
    ).to_csv(development_path, index=False)
    pd.DataFrame(
        {
            "gene": ["ENSG3", "ENSG2", "ENSG4"],
            "val_a": [60.5, 40.5, 100.0],
        }
    ).to_csv(validation_path, index=False)

    development = harmonize.load_gene_count_matrix(development_path)
    validation = harmonize.load_gene_count_matrix(validation_path)
    dev_norm, val_norm, diagnostics = harmonize.normalize_shared_gene_universe(
        development,
        validation,
        target_sum=10000.0,
    )

    assert list(dev_norm.columns) == ["ENSG2", "ENSG3"]
    assert list(val_norm.columns) == ["ENSG2", "ENSG3"]
    assert diagnostics["n_shared_genes"] == 2
    assert diagnostics["development_genes_missing_in_validation"] == ["ENSG1"]
    assert diagnostics["validation_genes_not_in_development"] == ["ENSG4"]
    assert np.isclose(np.expm1(dev_norm.loc["dev_a"]).sum(), 10000.0)
    assert np.isclose(np.expm1(val_norm.loc["val_a"]).sum(), 10000.0)
