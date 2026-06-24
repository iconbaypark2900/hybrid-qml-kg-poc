#!/usr/bin/env python3
"""Counts-first RNA-seq ingestion and signature export pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from single_cell_layer.count_matrix import (
    attach_metadata,
    compute_qc_summary,
    load_count_matrix,
    normalize_total_log1p,
    run_pydeseq2_de,
    run_simple_de,
    write_normalized_matrix,
    write_pipeline_manifest,
    write_qc_outputs,
)
from single_cell_layer.disease_signature import build_disease_signature
from single_cell_layer.signature_export import export_disease_signature


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--format", required=True, choices=["h5ad", "10x", "count-matrix"])
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--config", default="config/single_cell_config.yaml")
    parser.add_argument("--out-dir", default="artifacts/single_cell")
    parser.add_argument("--signatures-dir", default="artifacts/signatures")
    parser.add_argument("--condition-col", default="condition")
    parser.add_argument("--case-label", default="disease")
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--sample-id-col", default="sample_id")
    parser.add_argument("--gene-col", default="gene")
    parser.add_argument("--disease-id", default="disease")
    parser.add_argument("--tissue", default="")
    parser.add_argument("--top-n", type=int, default=250)
    parser.add_argument("--lfc-threshold", type=float, default=0.0)
    parser.add_argument("--padj-threshold", type=float, default=1.0)
    parser.add_argument("--de-method", choices=["simple", "pydeseq2"], default="simple")
    parser.add_argument("--de-min-total-count", type=int, default=10)
    parser.add_argument("--de-n-cpus", type=int, default=1)
    parser.add_argument(
        "--skip-de",
        action="store_true",
        help="Run validation, QC, and normalization only; do not use labels to derive signatures.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.format == "count-matrix" and not args.metadata:
        raise SystemExit("--metadata is required for --format count-matrix")

    if args.format == "count-matrix":
        adata = load_count_matrix(
            args.input,
            args.metadata,
            sample_id_col=args.sample_id_col,
            gene_col=args.gene_col,
            condition_col=args.condition_col,
            case_label=args.case_label,
            control_label=args.control_label,
        )
    elif args.format == "h5ad":
        from single_cell_layer.ingest_h5ad import load_h5ad

        adata = load_h5ad(args.input, config)
        if args.metadata:
            attach_metadata(adata, args.metadata, sample_id_col=args.sample_id_col)
    elif args.format == "10x":
        if not args.metadata:
            raise SystemExit("--metadata is required for --format 10x")
        from single_cell_layer.ingest_10x import load_10x

        adata = load_10x(args.input, config)
        attach_metadata(adata, args.metadata, sample_id_col=args.sample_id_col)
    else:
        raise SystemExit(f"Unsupported RNA-seq input format: {args.format}")

    labels = set(adata.obs[args.condition_col].astype(str)) if args.condition_col in adata.obs.columns else set()
    missing_labels = {args.case_label, args.control_label} - labels
    if missing_labels:
        raise SystemExit(
            f"Condition column '{args.condition_col}' must contain labels "
            f"'{args.case_label}' and '{args.control_label}'. Missing: {sorted(missing_labels)}"
        )

    raw_adata = adata.copy()
    normalization_cfg = config.get("normalization", {}) or config.get("single_cell", {}).get("normalization", {})
    target_sum = float(normalization_cfg.get("target_sum", 10000))
    normalize_total_log1p(adata, target_sum=target_sum)

    out_dir = Path(args.out_dir)
    signatures_dir = Path(args.signatures_dir)
    qc_dir = out_dir / "qc"
    qc_summary = compute_qc_summary(adata, condition_col=args.condition_col)
    qc_outputs = write_qc_outputs(qc_summary, qc_dir)
    normalized_path = write_normalized_matrix(adata, out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    de_path = None
    disease_sig_path = None
    cell_type_sig_path = None
    if not args.skip_de:
        if args.de_method == "pydeseq2":
            de_df = run_pydeseq2_de(
                raw_adata,
                condition_col=args.condition_col,
                case_label=args.case_label,
                control_label=args.control_label,
                min_total_count=args.de_min_total_count,
                n_cpus=args.de_n_cpus,
            )
        else:
            de_df = run_simple_de(
                adata,
                condition_col=args.condition_col,
                case_label=args.case_label,
                control_label=args.control_label,
            )
        de_path = out_dir / "differential_expression.csv"
        de_df.to_csv(de_path, index=False)

        disease_signature = build_disease_signature(
            de_df,
            disease_id=args.disease_id,
            tissue=args.tissue,
            cell_type="all_cells",
            lfc_threshold=args.lfc_threshold,
            padj_threshold=args.padj_threshold,
            top_n=args.top_n,
        )
        disease_sig_path = export_disease_signature(disease_signature, out_dir=str(signatures_dir))

        if args.cell_type_col and args.cell_type_col in adata.obs.columns:
            cell_type_signatures = {}
            for cell_type in sorted(map(str, adata.obs[args.cell_type_col].dropna().unique())):
                sub_de = run_simple_de(
                    adata,
                    condition_col=args.condition_col,
                    case_label=args.case_label,
                    control_label=args.control_label,
                    cell_type_col=args.cell_type_col,
                    cell_type_value=cell_type,
                )
                cell_type_signatures[cell_type] = build_disease_signature(
                    sub_de,
                    disease_id=args.disease_id,
                    tissue=args.tissue,
                    cell_type=cell_type,
                    lfc_threshold=args.lfc_threshold,
                    padj_threshold=args.padj_threshold,
                    top_n=args.top_n,
                )
            cell_type_sig_path = signatures_dir / "cell_type_signatures.json"
            signatures_dir.mkdir(parents=True, exist_ok=True)
            cell_type_sig_path.write_text(json.dumps(cell_type_signatures, indent=2) + "\n", encoding="utf-8")

    outputs = {
        "qc_summary_csv": str(qc_outputs["csv"]),
        "qc_summary_markdown": str(qc_outputs["markdown"]),
        "normalized_counts": str(normalized_path),
        "de_skipped": bool(args.skip_de),
    }
    if de_path is not None and disease_sig_path is not None:
        outputs["differential_expression"] = str(de_path)
        outputs["disease_signature"] = str(disease_sig_path)
        outputs["de_method"] = args.de_method
    if cell_type_sig_path is not None:
        outputs["cell_type_signatures"] = str(cell_type_sig_path)

    manifest_path = write_pipeline_manifest(
        out_dir=out_dir,
        input_path=args.input,
        input_format=args.format,
        metadata_path=args.metadata,
        qc_summary=qc_summary,
        outputs=outputs,
        config_path=args.config,
    )
    outputs["manifest"] = str(manifest_path)

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
