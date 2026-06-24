#!/usr/bin/env python3
"""Normalize two gene-count cohorts over one shared measured-gene universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gene_count_matrix(path: str | Path, *, gene_col: str = "gene") -> pd.DataFrame:
    frame = pd.read_csv(path)
    if gene_col not in frame.columns:
        raise ValueError(f"Count matrix {path} must contain gene column '{gene_col}'.")
    genes = frame[gene_col].astype(str).str.split(".", n=1).str[0]
    if genes.duplicated().any():
        duplicated = sorted(genes[genes.duplicated()].unique())
        raise ValueError(f"Count matrix {path} has duplicate genes after version stripping: {duplicated[:5]}")
    values = frame.drop(columns=[gene_col]).apply(pd.to_numeric, errors="raise")
    if values.empty:
        raise ValueError(f"Count matrix {path} has no sample columns.")
    if values.isna().any().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"Count matrix {path} contains missing or non-finite values.")
    if (values < 0).any().any():
        raise ValueError(f"Count matrix {path} contains negative values.")
    values.index = genes
    values.index.name = "gene"
    return values.astype(float)


def normalize_shared_gene_universe(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    target_sum: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    shared_genes = [gene for gene in development.index if gene in validation.index]
    if not shared_genes:
        raise ValueError("Development and validation count matrices have no shared genes.")
    development_shared = development.loc[shared_genes]
    validation_shared = validation.loc[shared_genes]

    def normalize(matrix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        sample_by_gene = matrix.T.to_numpy(dtype=float)
        totals = sample_by_gene.sum(axis=1)
        if np.any(totals <= 0):
            bad = matrix.columns[np.flatnonzero(totals <= 0)].tolist()
            raise ValueError(f"Samples have zero total count over the shared gene universe: {bad[:5]}")
        normalized = np.log1p(sample_by_gene * (float(target_sum) / totals)[:, None])
        return pd.DataFrame(normalized, index=matrix.columns, columns=shared_genes), totals

    development_normalized, development_totals = normalize(development_shared)
    validation_normalized, validation_totals = normalize(validation_shared)
    diagnostics = {
        "n_development_genes": int(len(development)),
        "n_validation_genes": int(len(validation)),
        "n_shared_genes": int(len(shared_genes)),
        "development_genes_missing_in_validation": sorted(set(development.index) - set(validation.index)),
        "validation_genes_not_in_development": sorted(set(validation.index) - set(development.index)),
        "development_library_total_shared_genes": {
            "min": float(development_totals.min()),
            "median": float(np.median(development_totals)),
            "max": float(development_totals.max()),
        },
        "validation_library_total_shared_genes": {
            "min": float(validation_totals.min()),
            "median": float(np.median(validation_totals)),
            "max": float(validation_totals.max()),
        },
    }
    return development_normalized, validation_normalized, diagnostics


def write_harmonized_outputs(
    development_normalized: pd.DataFrame,
    validation_normalized: pd.DataFrame,
    diagnostics: Dict[str, Any],
    *,
    development_counts_path: str | Path,
    validation_counts_path: str | Path,
    development_cohort: str,
    validation_cohort: str,
    target_sum: float,
    out_dir: str | Path,
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    development_path = out / "development_normalized_counts.csv"
    validation_path = out / "validation_normalized_counts.csv"
    genes_path = out / "shared_gene_universe.csv"
    development_missing_path = out / "development_genes_missing_in_validation.csv"
    validation_only_path = out / "validation_genes_not_in_development.csv"
    manifest_path = out / "harmonization_manifest.json"

    development_normalized.index.name = "sample_id"
    validation_normalized.index.name = "sample_id"
    development_normalized.to_csv(development_path)
    validation_normalized.to_csv(validation_path)
    pd.DataFrame({"gene": development_normalized.columns}).to_csv(genes_path, index=False)
    development_missing = list(diagnostics.get("development_genes_missing_in_validation", []))
    validation_only = list(diagnostics.get("validation_genes_not_in_development", []))
    pd.DataFrame({"gene": development_missing}).to_csv(development_missing_path, index=False)
    pd.DataFrame({"gene": validation_only}).to_csv(validation_only_path, index=False)
    manifest_diagnostics = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"development_genes_missing_in_validation", "validation_genes_not_in_development"}
    }
    manifest_diagnostics.update(
        {
            "n_development_genes_missing_in_validation": int(len(development_missing)),
            "n_validation_genes_not_in_development": int(len(validation_only)),
        }
    )

    manifest = {
        "schema_version": "1.0",
        "method": "shared_gene_universe_library_size_log1p",
        "formula": "log1p(count * target_sum / sample_total_over_shared_genes)",
        "target_sum": float(target_sum),
        "labels_used": False,
        "development_cohort": development_cohort,
        "validation_cohort": validation_cohort,
        "inputs": {
            "development_counts": {
                "path": str(development_counts_path),
                "sha256": _sha256(development_counts_path),
            },
            "validation_counts": {
                "path": str(validation_counts_path),
                "sha256": _sha256(validation_counts_path),
            },
        },
        "diagnostics": manifest_diagnostics,
        "outputs": {
            "development_normalized_counts": str(development_path),
            "validation_normalized_counts": str(validation_path),
            "shared_gene_universe": str(genes_path),
            "development_genes_missing_in_validation": str(development_missing_path),
            "validation_genes_not_in_development": str(validation_only_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest["outputs"], "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-counts", required=True)
    parser.add_argument("--validation-counts", required=True)
    parser.add_argument("--development-cohort", required=True)
    parser.add_argument("--validation-cohort", required=True)
    parser.add_argument("--gene-col", default="gene")
    parser.add_argument("--target-sum", type=float, default=10000.0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    development = load_gene_count_matrix(args.development_counts, gene_col=args.gene_col)
    validation = load_gene_count_matrix(args.validation_counts, gene_col=args.gene_col)
    development_normalized, validation_normalized, diagnostics = normalize_shared_gene_universe(
        development,
        validation,
        target_sum=args.target_sum,
    )
    outputs = write_harmonized_outputs(
        development_normalized,
        validation_normalized,
        diagnostics,
        development_counts_path=args.development_counts,
        validation_counts_path=args.validation_counts,
        development_cohort=args.development_cohort,
        validation_cohort=args.validation_cohort,
        target_sum=args.target_sum,
        out_dir=args.out_dir,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
