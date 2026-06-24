from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class CountMatrixValidationError(ValueError):
    """Raised when a counts matrix and metadata cannot be joined safely."""


@dataclass
class SimpleAnnData:
    """Small AnnData-compatible fallback for counts-first tests and CLI smoke runs."""

    X: np.ndarray
    obs: pd.DataFrame
    var: pd.DataFrame

    @property
    def n_obs(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_vars(self) -> int:
        return int(self.X.shape[1])

    @property
    def obs_names(self) -> pd.Index:
        return self.obs.index

    @property
    def var_names(self) -> pd.Index:
        return self.var.index

    def copy(self) -> "SimpleAnnData":
        return SimpleAnnData(self.X.copy(), self.obs.copy(), self.var.copy())


def load_count_matrix(
    counts_path: str | Path,
    metadata_path: str | Path,
    *,
    sample_id_col: str = "sample_id",
    gene_col: str = "gene",
    condition_col: str = "condition",
    case_label: str = "disease",
    control_label: str = "control",
    delimiter: Optional[str] = None,
):
    """Load a gene x sample count matrix and metadata as an AnnData-like object."""

    counts_file = Path(counts_path)
    metadata_file = Path(metadata_path)
    counts = pd.read_csv(counts_file, sep=_infer_sep(counts_file, delimiter))
    metadata = pd.read_csv(metadata_file, sep=_infer_sep(metadata_file, delimiter))

    if gene_col not in counts.columns:
        raise CountMatrixValidationError(f"Count matrix must include gene column '{gene_col}'.")
    if sample_id_col not in metadata.columns:
        raise CountMatrixValidationError(f"Metadata must include sample ID column '{sample_id_col}'.")
    if condition_col not in metadata.columns:
        raise CountMatrixValidationError(f"Metadata must include condition column '{condition_col}'.")

    genes = counts[gene_col].astype(str)
    if genes.duplicated().any():
        duplicated = sorted(genes[genes.duplicated()].unique())
        raise CountMatrixValidationError(f"Duplicate gene IDs in count matrix: {duplicated[:5]}")

    sample_cols = [col for col in counts.columns if col != gene_col]
    metadata_ids = metadata[sample_id_col].astype(str).tolist()
    missing_metadata = sorted(set(sample_cols) - set(metadata_ids))
    extra_metadata = sorted(set(metadata_ids) - set(sample_cols))
    if missing_metadata or extra_metadata:
        raise CountMatrixValidationError(
            "Count matrix sample columns must exactly match metadata sample IDs. "
            f"Missing metadata for: {missing_metadata[:5]}; metadata without counts: {extra_metadata[:5]}"
        )

    labels = set(metadata[condition_col].astype(str))
    missing_labels = {case_label, control_label} - labels
    if missing_labels:
        raise CountMatrixValidationError(
            f"Condition column '{condition_col}' must contain labels "
            f"'{case_label}' and '{control_label}'. Missing: {sorted(missing_labels)}"
        )

    matrix = counts.set_index(gene_col).loc[:, metadata_ids].apply(pd.to_numeric, errors="raise")
    if (matrix < 0).any().any():
        raise CountMatrixValidationError("Count matrix cannot contain negative counts.")

    obs = metadata.copy()
    obs[sample_id_col] = obs[sample_id_col].astype(str)
    obs = obs.set_index(sample_id_col, drop=False).loc[metadata_ids]
    var = pd.DataFrame(index=matrix.index.astype(str))
    return _make_anndata(matrix.T.to_numpy(dtype=float), obs, var)


def normalize_total_log1p(adata, *, target_sum: float = 10000.0):
    """Counts-per-target-sum normalization plus log1p transform."""

    X = np.asarray(adata.X, dtype=float)
    totals = X.sum(axis=1)
    scale = np.divide(target_sum, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    adata.X = np.log1p(X * scale[:, None])
    adata.obs["total_counts"] = totals
    adata.obs["n_genes_by_counts"] = (X > 0).sum(axis=1)
    return adata


def compute_qc_summary(adata, *, condition_col: str = "condition") -> dict[str, Any]:
    obs = adata.obs
    condition_counts = {
        str(k): int(v)
        for k, v in obs[condition_col].astype(str).value_counts().sort_index().items()
    } if condition_col in obs.columns else {}
    total_counts = obs.get("total_counts")
    n_genes = obs.get("n_genes_by_counts")
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "condition_counts": condition_counts,
        "total_counts_mean": _round_or_none(float(total_counts.mean())) if total_counts is not None else None,
        "n_genes_by_counts_mean": _round_or_none(float(n_genes.mean())) if n_genes is not None else None,
    }


def write_qc_outputs(summary: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "qc_summary_table.csv"
    md_path = out / "qc_summary_table.md"
    json_path = out / "qc_summary.json"

    rows = [
        {"metric": "n_cells", "value": summary["n_cells"]},
        {"metric": "n_genes", "value": summary["n_genes"]},
        {"metric": "total_counts_mean", "value": summary.get("total_counts_mean")},
        {"metric": "n_genes_by_counts_mean", "value": summary.get("n_genes_by_counts_mean")},
    ]
    for condition, count in summary.get("condition_counts", {}).items():
        rows.append({"metric": f"condition_count:{condition}", "value": count})

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_lines = ["# RNA-seq Counts QC Summary", "", "| Metric | Value |", "|---|---|"]
    md_lines.extend(f"| {row['metric']} | {row['value']} |" for row in rows)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path, "json": json_path}


def run_simple_de(
    adata,
    *,
    condition_col: str = "condition",
    case_label: str = "disease",
    control_label: str = "control",
    cell_type_col: Optional[str] = None,
    cell_type_value: Optional[str] = None,
) -> pd.DataFrame:
    """Deterministic effect-size table for fixtures and smoke tests only."""

    obs = adata.obs
    X = np.asarray(adata.X, dtype=float)
    if cell_type_col and cell_type_value is not None:
        mask = obs[cell_type_col].astype(str).to_numpy() == str(cell_type_value)
        X = X[mask]
        obs = obs.loc[mask].copy()

    case_mask = obs[condition_col].astype(str).to_numpy() == case_label
    control_mask = obs[condition_col].astype(str).to_numpy() == control_label
    if case_mask.sum() == 0 or control_mask.sum() == 0:
        raise CountMatrixValidationError(
            f"Need at least one '{case_label}' and one '{control_label}' observation for DE."
        )

    case_mean = X[case_mask].mean(axis=0)
    control_mean = X[control_mask].mean(axis=0)
    logfc = np.log2((case_mean + 1.0) / (control_mean + 1.0))
    scores = np.abs(logfc)
    pvals_adj = 1.0 / (1.0 + scores)
    genes = list(map(str, adata.var.index))
    df = pd.DataFrame(
        {
            "names": genes,
            "scores": scores,
            "logfoldchanges": logfc,
            "pvals_adj": pvals_adj,
        }
    ).sort_values("scores", ascending=False).reset_index(drop=True)
    if cell_type_value is not None:
        df["cell_type"] = str(cell_type_value)
    return df


def run_pydeseq2_de(
    adata,
    *,
    condition_col: str = "condition",
    case_label: str = "disease",
    control_label: str = "control",
    min_total_count: int = 10,
    n_cpus: int = 1,
) -> pd.DataFrame:
    """Run negative-binomial bulk RNA-seq DE using PyDESeq2."""

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as exc:
        raise ImportError(
            "PyDESeq2 is required for --de-method pydeseq2. "
            "Install requirements-omics.txt."
        ) from exc

    if condition_col not in adata.obs.columns:
        raise CountMatrixValidationError(f"Metadata must include condition column '{condition_col}'.")
    counts = pd.DataFrame(
        np.asarray(adata.X),
        index=adata.obs.index.astype(str),
        columns=adata.var.index.astype(str),
    )
    if (counts < 0).any().any():
        raise CountMatrixValidationError("PyDESeq2 counts cannot contain negative values.")
    rounded = np.rint(counts.to_numpy(dtype=float))
    if not np.allclose(counts.to_numpy(dtype=float), rounded):
        raise CountMatrixValidationError("PyDESeq2 requires raw integer counts before normalization.")
    counts = pd.DataFrame(rounded.astype("int64"), index=counts.index, columns=counts.columns)
    counts = counts.loc[:, counts.sum(axis=0) >= int(min_total_count)]
    if counts.empty:
        raise CountMatrixValidationError(
            f"No genes passed the PyDESeq2 minimum total count threshold ({min_total_count})."
        )

    metadata = adata.obs.loc[counts.index, [condition_col]].copy()
    metadata[condition_col] = metadata[condition_col].astype(str)
    labels = set(metadata[condition_col])
    missing = {case_label, control_label} - labels
    if missing:
        raise CountMatrixValidationError(
            f"Condition column '{condition_col}' is missing labels required for PyDESeq2: {sorted(missing)}"
        )

    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design=f"~{condition_col}",
        refit_cooks=True,
        n_cpus=max(1, int(n_cpus)),
        quiet=True,
        low_memory=True,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=[condition_col, case_label, control_label],
        cooks_filter=True,
        independent_filter=True,
        n_cpus=max(1, int(n_cpus)),
        quiet=True,
    )
    stats.summary()
    results = stats.results_df.copy()
    results.index = results.index.astype(str)
    results = results.reset_index(names="names")
    results["logfoldchanges"] = pd.to_numeric(results["log2FoldChange"], errors="coerce")
    results["scores"] = pd.to_numeric(results["stat"], errors="coerce").abs()
    results["pvals"] = pd.to_numeric(results["pvalue"], errors="coerce")
    results["pvals_adj"] = pd.to_numeric(results["padj"], errors="coerce")
    results["base_mean"] = pd.to_numeric(results["baseMean"], errors="coerce")
    results["de_method"] = "pydeseq2_wald"
    return (
        results[
            ["names", "scores", "logfoldchanges", "pvals", "pvals_adj", "base_mean", "de_method"]
        ]
        .sort_values(["pvals_adj", "scores"], ascending=[True, False], na_position="last")
        .reset_index(drop=True)
    )


def write_normalized_matrix(adata, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "normalized_counts.csv"
    matrix = pd.DataFrame(np.asarray(adata.X), index=adata.obs.index, columns=adata.var.index)
    matrix.index.name = "sample_id"
    matrix.to_csv(path)
    return path


def write_pipeline_manifest(
    *,
    out_dir: str | Path,
    input_path: str,
    input_format: str,
    metadata_path: Optional[str],
    qc_summary: dict[str, Any],
    outputs: dict[str, str],
    config_path: str,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_path": input_path,
        "input_format": input_format,
        "metadata_path": metadata_path,
        "config_path": config_path,
        "qc_summary": qc_summary,
        "outputs": outputs,
    }
    path = out / "rnaseq_counts_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def attach_metadata(
    adata,
    metadata_path: str | Path,
    *,
    sample_id_col: str = "sample_id",
) -> Any:
    """Attach metadata rows to an AnnData-like object by observation/sample ID."""

    metadata_file = Path(metadata_path)
    metadata = pd.read_csv(metadata_file, sep=_infer_sep(metadata_file, None))
    if sample_id_col not in metadata.columns:
        raise CountMatrixValidationError(f"Metadata must include sample ID column '{sample_id_col}'.")
    metadata[sample_id_col] = metadata[sample_id_col].astype(str)
    metadata = metadata.set_index(sample_id_col, drop=False)

    obs_ids = list(map(str, adata.obs.index))
    missing = sorted(set(obs_ids) - set(metadata.index))
    extra = sorted(set(metadata.index) - set(obs_ids))
    if missing or extra:
        raise CountMatrixValidationError(
            "Metadata sample IDs must exactly match observations. "
            f"Missing metadata for: {missing[:5]}; metadata without observations: {extra[:5]}"
        )

    joined = adata.obs.copy()
    for col in metadata.columns:
        joined[col] = metadata.loc[obs_ids, col].to_numpy()
    adata.obs = joined
    return adata


def _make_anndata(X: np.ndarray, obs: pd.DataFrame, var: pd.DataFrame):
    try:
        import anndata as ad
    except ImportError:
        return SimpleAnnData(X=X, obs=obs.copy(), var=var.copy())
    return ad.AnnData(X=X, obs=obs.copy(), var=var.copy())


def _infer_sep(path: Path, delimiter: Optional[str]) -> str:
    if delimiter is not None:
        return delimiter
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _round_or_none(value: float | None, ndigits: int = 3) -> float | None:
    return round(value, ndigits) if value is not None else None
