#!/usr/bin/env python3
"""Download and convert the open GSE225846 breast tumor/normal RNA-seq cohort."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import pandas as pd


ACCESSION = "GSE225846"
BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE225nnn/GSE225846"
COUNTS_NAME = "GSE225846_RawCountFile_rsemgenes.txt.gz"
SOFT_NAME = "GSE225846_family.soft.gz"
COUNTS_URL = f"{BASE_URL}/suppl/{COUNTS_NAME}"
SOFT_URL = f"{BASE_URL}/soft/{SOFT_NAME}"
GENE_PATTERN = re.compile(r"^(ENSG\d+)(?:\.\d+)?(?:_(.*))?$")


def download_file(url: str, path: Path, *, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return path
    temporary = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "hybrid-qml-kg-poc/geo-cohort"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temporary.replace(path)
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_soft_samples(path: str | Path) -> pd.DataFrame:
    records: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith("^SAMPLE = "):
                if current:
                    records.append(current)
                current = {"sample_id": line.split(" = ", 1)[1].strip()}
            elif current is None:
                continue
            elif line.startswith("!Sample_title = "):
                current["sample_title"] = line.split(" = ", 1)[1].strip()
            elif line.startswith("!Sample_characteristics_ch1 = "):
                value = line.split(" = ", 1)[1].strip()
                if ": " in value:
                    key, item = value.split(": ", 1)
                    current[key.strip().lower()] = item.strip()
            elif line.startswith("!Sample_relation = BioSample: "):
                current["biosample_url"] = line.split("BioSample: ", 1)[1].strip()
            elif line.startswith("!Sample_relation = SRA: "):
                current["sra_url"] = line.split("SRA: ", 1)[1].strip()
        if current:
            records.append(current)

    metadata = pd.DataFrame(records)
    required = {"sample_id", "sample_title", "accession", "type"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"GEO SOFT metadata is missing required fields: {sorted(missing)}")
    metadata["count_column"] = metadata["sample_title"].str.replace(r"^S_", "", regex=True)
    metadata["condition"] = metadata["type"].map({"tumor": "disease", "normal": "control"})
    if metadata["condition"].isna().any():
        unknown = sorted(metadata.loc[metadata["condition"].isna(), "type"].astype(str).unique())
        raise ValueError(f"Unsupported GSE225846 sample types: {unknown}")
    metadata = metadata.rename(
        columns={
            "accession": "patient_id",
            "age (at surgery)": "age_at_surgery",
            "er status": "er_status",
            "pr": "pr_status",
            "her2": "her2_status",
            "stage": "stage",
        }
    )
    metadata["cohort"] = ACCESSION
    metadata["batch"] = ACCESSION
    return metadata


def parse_counts(
    path: str | Path,
    metadata: pd.DataFrame,
    *,
    min_total_count: int,
    gene_allowlist: Optional[Set[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, float | int | bool]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = pd.read_csv(handle, sep="\t", low_memory=False)
    if "gene_id" not in raw.columns:
        raise ValueError("GSE225846 count file is missing the gene_id column.")

    count_to_sample = dict(zip(metadata["count_column"].astype(str), metadata["sample_id"].astype(str)))
    missing_counts = sorted(set(count_to_sample) - set(raw.columns))
    extra_counts = sorted(set(raw.columns) - {"gene_id"} - set(count_to_sample))
    if missing_counts or extra_counts:
        raise ValueError(
            "GEO count columns do not exactly match SOFT sample titles. "
            f"Missing counts: {missing_counts[:5]}; unannotated counts: {extra_counts[:5]}"
        )

    parsed = raw["gene_id"].astype(str).str.extract(GENE_PATTERN)
    parsed.columns = ["gene_id", "gene_symbol"]
    keep = parsed["gene_id"].notna()
    values = raw.loc[keep, list(count_to_sample)].apply(pd.to_numeric, errors="raise")
    if (values < 0).any().any():
        raise ValueError("GSE225846 count matrix contains negative values.")
    numeric = values.to_numpy(dtype=float)
    fractional = abs(numeric - numeric.round())
    fractional_entries = int((fractional > 1e-8).sum())
    value_diagnostics: Dict[str, float | int | bool] = {
        "n_values": int(numeric.size),
        "n_fractional_values": fractional_entries,
        "fractional_value_fraction": float(fractional_entries / numeric.size),
        "max_fractional_part": float(fractional.max()),
        "integer_valued": fractional_entries == 0,
    }

    counts = values.reset_index(drop=True).copy()
    counts.insert(0, "gene_id", parsed.loc[keep, "gene_id"].to_numpy())
    counts = counts.groupby("gene_id", as_index=False).sum()
    if gene_allowlist is not None:
        counts = counts.loc[counts["gene_id"].isin(gene_allowlist)].copy()
        missing_allowlist = sorted(gene_allowlist - set(counts["gene_id"]))
        if missing_allowlist:
            raise ValueError(
                "GSE225846 is missing genes required by the normalization allowlist: "
                f"{missing_allowlist[:10]}"
            )
    else:
        counts = counts.loc[counts.drop(columns="gene_id").sum(axis=1) >= int(min_total_count)].copy()
    counts = counts.rename(columns=count_to_sample).rename(columns={"gene_id": "gene"})

    gene_map = parsed.loc[keep].drop_duplicates("gene_id").rename(columns={"gene_id": "gene"})
    gene_map = gene_map.set_index("gene").loc[counts["gene"]].reset_index()
    return counts, gene_map, value_diagnostics


def load_gene_allowlist(path: str | Path) -> Set[str]:
    allowlist_path = Path(path)
    separator = "\t" if allowlist_path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(allowlist_path, sep=separator)
    column = next((name for name in ["gene", "gene_id"] if name in frame.columns), None)
    if column is None:
        raise ValueError("Gene allowlist must contain a 'gene' or 'gene_id' column.")
    genes = set(frame[column].astype(str).str.split(".", n=1).str[0])
    genes.discard("")
    if not genes:
        raise ValueError("Gene allowlist is empty.")
    return genes


def write_cohort(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    gene_map: pd.DataFrame,
    out_dir: Path,
    *,
    counts_raw: Path,
    soft_raw: Path,
    min_total_count: int,
    value_diagnostics: Dict[str, float | int | bool],
    gene_allowlist_path: Optional[Path],
    gene_allowlist_size: Optional[int],
) -> Dict[str, str]:
    converted = out_dir / "converted"
    converted.mkdir(parents=True, exist_ok=True)
    counts_path = converted / "gse225846_counts.csv"
    metadata_path = converted / "gse225846_metadata.csv"
    gene_map_path = converted / "gse225846_gene_map.csv"
    manifest_path = converted / "cohort_manifest.json"

    metadata_columns = [
        "sample_id", "condition", "patient_id", "sample_title", "count_column", "type",
        "age_at_surgery", "race", "er_status", "pr_status", "her2_status", "stage",
        "biosample_url", "sra_url", "cohort", "batch",
    ]
    for column in metadata_columns:
        if column not in metadata.columns:
            metadata[column] = ""
    metadata = metadata[metadata_columns].copy()
    counts.to_csv(counts_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    gene_map.to_csv(gene_map_path, index=False)

    patient_counts = metadata.groupby("patient_id")["condition"].nunique()
    manifest = {
        "source": "NCBI Gene Expression Omnibus",
        "accession": ACCESSION,
        "series_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ACCESSION}",
        "publication": {"pubmed_id": "37902422"},
        "organism": "Homo sapiens",
        "assay": "bulk RNA-seq",
        "reference_assembly": "hg38",
        "quantification": "STAR/RSEM expected gene counts as deposited by the study authors",
        "count_value_diagnostics": value_diagnostics,
        "pydeseq2_compatible_without_rounding": bool(value_diagnostics["integer_valued"]),
        "count_value_note": (
            "RSEM expected counts are preserved as deposited. Fractional values are suitable for library-size "
            "normalization and prediction but are not passed to the integer-count PyDESeq2 workflow."
        ),
        "min_total_count": int(min_total_count),
        "normalization_gene_universe": (
            {
                "mode": "development_gene_allowlist",
                "path": str(gene_allowlist_path),
                "sha256": sha256_file(gene_allowlist_path),
                "n_genes_requested": int(gene_allowlist_size or 0),
                "n_genes_matched": int(len(counts)),
                "minimum_total_count_filter_applied": False,
            }
            if gene_allowlist_path is not None
            else {
                "mode": "all_deposited_genes",
                "minimum_total_count_filter_applied": True,
            }
        ),
        "n_samples": int(len(metadata)),
        "n_patients": int(metadata["patient_id"].nunique()),
        "n_paired_patients": int((patient_counts > 1).sum()),
        "condition_counts": {str(k): int(v) for k, v in metadata["condition"].value_counts().sort_index().items()},
        "n_genes": int(len(counts)),
        "raw_files": {
            "counts": {"path": str(counts_raw), "url": COUNTS_URL, "sha256": sha256_file(counts_raw)},
            "soft": {"path": str(soft_raw), "url": SOFT_URL, "sha256": sha256_file(soft_raw)},
        },
        "outputs": {
            "counts": str(counts_path),
            "metadata": str(metadata_path),
            "gene_map": str(gene_map_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {**manifest["outputs"], "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="artifacts/external/geo_gse225846")
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument(
        "--gene-allowlist",
        default=None,
        help="Optional CSV/TSV gene or gene_id list used to harmonize the normalization universe.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    counts_raw = download_file(COUNTS_URL, raw_dir / COUNTS_NAME, overwrite=args.overwrite)
    soft_raw = download_file(SOFT_URL, raw_dir / SOFT_NAME, overwrite=args.overwrite)
    metadata = parse_soft_samples(soft_raw)
    gene_allowlist_path = Path(args.gene_allowlist) if args.gene_allowlist else None
    gene_allowlist = load_gene_allowlist(gene_allowlist_path) if gene_allowlist_path else None
    counts, gene_map, value_diagnostics = parse_counts(
        counts_raw,
        metadata,
        min_total_count=args.min_total_count,
        gene_allowlist=gene_allowlist,
    )
    outputs = write_cohort(
        counts,
        metadata,
        gene_map,
        out_dir,
        counts_raw=counts_raw,
        soft_raw=soft_raw,
        min_total_count=args.min_total_count,
        value_diagnostics=value_diagnostics,
        gene_allowlist_path=gene_allowlist_path,
        gene_allowlist_size=len(gene_allowlist) if gene_allowlist is not None else None,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
