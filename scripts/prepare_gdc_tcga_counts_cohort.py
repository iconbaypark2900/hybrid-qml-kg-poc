#!/usr/bin/env python3
"""Download and convert an open GDC TCGA STAR-count cohort."""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"


def query_gdc_files(project_id: str) -> List[Dict[str, Any]]:
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Transcriptome Profiling"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
            {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
        ],
    }
    body = {
        "filters": filters,
        "fields": "file_id,file_name,cases.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "5000",
    }
    request = urllib.request.Request(
        GDC_FILES_ENDPOINT,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "hybrid-qml-kg-poc/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    return list(payload["data"]["hits"])


def flatten_file_records(hits: Iterable[Dict[str, Any]], project_id: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for hit in hits:
        for case in hit.get("cases", []):
            case_id = str(case.get("submitter_id", "")).strip()
            sample_types = sorted(
                {
                    str(sample.get("sample_type", "")).strip()
                    for sample in case.get("samples", [])
                    if sample.get("sample_type")
                }
            )
            for sample_type in sample_types:
                records.append(
                    {
                        "project_id": project_id,
                        "case_submitter_id": case_id,
                        "sample_type": sample_type,
                        "file_id": str(hit["file_id"]),
                        "file_name": str(hit["file_name"]),
                    }
                )
    return records


def select_balanced_cohort(
    records: Iterable[Dict[str, str]],
    *,
    case_sample_type: str,
    control_sample_type: str,
    n_case: int,
    n_control: int,
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for sample_type, condition, requested in [
        (case_sample_type, "disease", n_case),
        (control_sample_type, "control", n_control),
    ]:
        candidates = sorted(
            (record for record in records if record["sample_type"] == sample_type),
            key=lambda record: (record["case_submitter_id"], record["file_id"]),
        )
        unique_cases: Dict[str, Dict[str, str]] = {}
        for record in candidates:
            unique_cases.setdefault(record["case_submitter_id"], record)
        if len(unique_cases) < requested:
            raise ValueError(
                f"Requested {requested} '{sample_type}' cases but only {len(unique_cases)} unique open cases are available."
            )
        for record in list(unique_cases.values())[:requested]:
            selected.append({**record, "condition": condition, "sample_id": record["case_submitter_id"]})
    return sorted(selected, key=lambda record: (record["condition"], record["sample_id"]))


def download_gdc_file(record: Dict[str, str], raw_dir: Path, *, overwrite: bool) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / record["file_name"]
    if path.exists() and path.stat().st_size > 0 and not overwrite:
        return path
    tmp_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        f"{GDC_DATA_ENDPOINT}/{record['file_id']}",
        headers={"User-Agent": "hybrid-qml-kg-poc/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, tmp_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp_path.replace(path)
    return path


def parse_star_counts(
    path: str | Path,
    *,
    count_column: str,
    gene_types: Iterable[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", comment="#", dtype={"gene_id": str, "gene_name": str, "gene_type": str})
    required = {"gene_id", "gene_name", "gene_type", count_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GDC STAR count file {path} is missing columns: {sorted(missing)}")
    frame = frame[frame["gene_id"].astype(str).str.startswith("ENSG")].copy()
    allowed_types = {str(value) for value in gene_types if str(value)}
    if allowed_types:
        frame = frame[frame["gene_type"].isin(allowed_types)].copy()
    frame["gene_id"] = frame["gene_id"].str.split(".", n=1).str[0]
    frame[count_column] = pd.to_numeric(frame[count_column], errors="raise").astype("int64")
    frame = (
        frame.groupby("gene_id", as_index=False)
        .agg(gene_symbol=("gene_name", "first"), gene_type=("gene_type", "first"), count=(count_column, "sum"))
    )
    return frame


def build_count_matrix(
    selected: List[Dict[str, str]],
    raw_dir: Path,
    *,
    count_column: str,
    gene_types: Iterable[str],
    min_total_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_series = []
    gene_maps = []
    for record in selected:
        parsed = parse_star_counts(
            raw_dir / record["file_name"],
            count_column=count_column,
            gene_types=gene_types,
        )
        count_series.append(parsed.set_index("gene_id")["count"].rename(record["sample_id"]))
        gene_maps.append(parsed[["gene_id", "gene_symbol", "gene_type"]])
    counts = pd.concat(count_series, axis=1, join="inner").fillna(0).astype("int64")
    counts = counts.loc[counts.sum(axis=1) >= int(min_total_count)].copy()
    counts.index.name = "gene"
    gene_map = (
        pd.concat(gene_maps, ignore_index=True)
        .drop_duplicates("gene_id")
        .set_index("gene_id")
        .loc[counts.index]
        .reset_index()
    )
    return counts.reset_index(), gene_map


def write_cohort(
    selected: List[Dict[str, str]],
    counts: pd.DataFrame,
    gene_map: pd.DataFrame,
    out_dir: Path,
    *,
    project_id: str,
    count_column: str,
    gene_types: List[str],
    min_total_count: int,
) -> Dict[str, str]:
    converted_dir = out_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    counts_path = converted_dir / f"{project_id.lower().replace('-', '_')}_counts.csv"
    metadata_path = converted_dir / f"{project_id.lower().replace('-', '_')}_metadata.csv"
    gene_map_path = converted_dir / f"{project_id.lower().replace('-', '_')}_gene_map.csv"
    manifest_path = converted_dir / "cohort_manifest.json"

    metadata = pd.DataFrame(selected)[
        ["sample_id", "condition", "project_id", "case_submitter_id", "sample_type", "file_id", "file_name"]
    ].copy()
    metadata["batch"] = project_id
    counts.to_csv(counts_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    gene_map.to_csv(gene_map_path, index=False)

    manifest = {
        "source": "NCI Genomic Data Commons",
        "api_files_endpoint": GDC_FILES_ENDPOINT,
        "api_data_endpoint": GDC_DATA_ENDPOINT,
        "project_id": project_id,
        "workflow_type": "STAR - Counts",
        "count_column": count_column,
        "gene_types": gene_types,
        "min_total_count": int(min_total_count),
        "n_samples": int(len(metadata)),
        "condition_counts": {str(k): int(v) for k, v in metadata["condition"].value_counts().sort_index().items()},
        "n_genes": int(len(counts)),
        "samples": selected,
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
    parser.add_argument("--project-id", default="TCGA-BRCA")
    parser.add_argument("--case-sample-type", default="Primary Tumor")
    parser.add_argument("--control-sample-type", default="Solid Tissue Normal")
    parser.add_argument("--n-case", type=int, default=30)
    parser.add_argument("--n-control", type=int, default=30)
    parser.add_argument("--count-column", default="unstranded")
    parser.add_argument("--gene-types", default="protein_coding")
    parser.add_argument("--min-total-count", type=int, default=10)
    parser.add_argument("--out-dir", default="artifacts/external/gdc_tcga_brca")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.n_case < 1 or args.n_control < 1:
        raise SystemExit("--n-case and --n-control must be positive.")
    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    gene_types = [value.strip() for value in args.gene_types.split(",") if value.strip()]

    records = flatten_file_records(query_gdc_files(args.project_id), args.project_id)
    selected = select_balanced_cohort(
        records,
        case_sample_type=args.case_sample_type,
        control_sample_type=args.control_sample_type,
        n_case=args.n_case,
        n_control=args.n_control,
    )
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {
            pool.submit(download_gdc_file, record, raw_dir, overwrite=args.overwrite): record
            for record in selected
        }
        for future in as_completed(futures):
            future.result()

    counts, gene_map = build_count_matrix(
        selected,
        raw_dir,
        count_column=args.count_column,
        gene_types=gene_types,
        min_total_count=args.min_total_count,
    )
    outputs = write_cohort(
        selected,
        counts,
        gene_map,
        out_dir,
        project_id=args.project_id,
        count_column=args.count_column,
        gene_types=gene_types,
        min_total_count=args.min_total_count,
    )
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
