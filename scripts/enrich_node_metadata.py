#!/usr/bin/env python3
"""
Enrich `data/hetionet_nodes_metadata.csv` with human-readable names.

Why:
- The project stores nodes as stable IDs like `Compound::DB00688` and `Disease::DOID:0060048`.
- Non-technical audiences need names (e.g., "Dexamethasone", "COVID-19").

How:
- Prefer a local Hetionet nodes TSV if available.
- Otherwise, optionally download Hetionet nodes TSV from known mirrors (best-effort).
- Merge `name` onto the existing metadata file, filling blanks only (never overwrites non-empty names).

This script is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "hetionet_nodes_metadata.csv"


def _read_nodes_tsv(path: Path) -> pd.DataFrame:
    # Try headered TSV first (expected).
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.fillna("")
    return df


def _locate_local_nodes_file(data_dir: Path) -> Optional[Path]:
    candidates = [
        data_dir / "hetionet-v1.0-nodes.tsv",
        data_dir / "hetionet-v1.0-nodes.tsv.gz",
        data_dir / "hetionet-v1.0-nodes.csv",
        data_dir / "hetionet-v1.0-nodes.csv.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _try_download_nodes_table(out_path: Path) -> Tuple[bool, str]:
    """
    Best-effort download of Hetionet nodes table. Network access may be disabled in some environments.
    Returns (ok, message).
    """
    # Mirrors patterned after kg_loader download logic; node table location can vary across branches.
    candidates = [
        ("https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv.gz", True),
        ("https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv.gz", True),
        ("https://raw.githubusercontent.com/hetio/hetionet/master/hetnet/tsv/hetionet-v1.0-nodes.tsv.gz", True),
        ("https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-nodes.tsv.gz", True),
        ("https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv", False),
        ("https://github.com/hetio/hetionet/raw/main/hetnet/tsv/hetionet-v1.0-nodes.tsv", False),
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url, gz in candidates:
        try:
            df = pd.read_csv(
                url,
                sep="\t",
                dtype=str,
                compression=("gzip" if gz else None),
            ).fillna("")
            if len(df) == 0:
                raise ValueError("Downloaded nodes table is empty.")
            # Save as plain TSV for repeatable local usage.
            df.to_csv(out_path, sep="\t", index=False)
            return True, f"Downloaded nodes table from {url} → {out_path}"
        except Exception as e:
            last_err = e
            continue
    return False, f"Failed to download nodes table (last error: {last_err})"


def _infer_id_and_name_columns(df_nodes: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to infer the node ID and name columns from a Hetionet nodes table.
    Common formats include columns like:
      - id / identifier / node_id
      - name / label
    """
    cols = {c.lower(): c for c in df_nodes.columns}
    id_candidates = ["node_id", "id", "identifier"]
    name_candidates = ["name", "label"]

    id_col = None
    name_col = None
    for k in id_candidates:
        if k in cols:
            id_col = cols[k]
            break
    for k in name_candidates:
        if k in cols:
            name_col = cols[k]
            break

    if not id_col or not name_col:
        raise ValueError(f"Could not infer id/name columns from nodes table columns: {list(df_nodes.columns)}")
    return id_col, name_col


def enrich_metadata(metadata_csv: Path, nodes_table: pd.DataFrame) -> Tuple[int, int]:
    """
    Returns (filled_names, total_rows).
    """
    md = pd.read_csv(metadata_csv, dtype=str).fillna("")
    if "node_id" not in md.columns:
        raise ValueError(f"{metadata_csv} missing required column 'node_id'")
    for col in ["name", "namespace", "external_url"]:
        if col not in md.columns:
            md[col] = ""

    id_col, name_col = _infer_id_and_name_columns(nodes_table)
    nodes = nodes_table[[id_col, name_col]].copy()
    nodes = nodes.rename(columns={id_col: "node_id", name_col: "name_src"})
    nodes["node_id"] = nodes["node_id"].astype(str).str.strip()
    nodes["name_src"] = nodes["name_src"].astype(str).str.strip()
    nodes = nodes[nodes["node_id"] != ""]

    merged = md.merge(nodes, on="node_id", how="left")
    before_blank = (merged["name"].astype(str).str.strip() == "").sum()
    # Fill only blanks
    merged.loc[merged["name"].astype(str).str.strip() == "", "name"] = merged["name_src"].fillna("")
    after_blank = (merged["name"].astype(str).str.strip() == "").sum()
    filled = int(before_blank - after_blank)

    merged = merged.drop(columns=["name_src"])
    merged.to_csv(metadata_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return filled, int(len(merged))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_csv", type=str, default=str(DEFAULT_METADATA))
    ap.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--download_if_missing", action="store_true", help="Attempt to download Hetionet nodes TSV if not found locally.")
    args = ap.parse_args()

    metadata_csv = Path(args.metadata_csv)
    data_dir = Path(args.data_dir)
    if not metadata_csv.exists():
        raise SystemExit(f"Metadata file not found: {metadata_csv}")

    nodes_path = _locate_local_nodes_file(data_dir)
    if nodes_path is None and args.download_if_missing:
        ok, msg = _try_download_nodes_table(data_dir / "hetionet-v1.0-nodes.tsv")
        print(msg)
        if ok:
            nodes_path = data_dir / "hetionet-v1.0-nodes.tsv"

    if nodes_path is None:
        raise SystemExit(
            "No Hetionet nodes table found.\n"
            "Options:\n"
            f"  - Place `hetionet-v1.0-nodes.tsv` into {data_dir}\n"
            "  - Or rerun with `--download_if_missing` (requires network access)\n"
        )

    # Read nodes table
    if str(nodes_path).endswith(".csv") or str(nodes_path).endswith(".csv.gz"):
        df_nodes = pd.read_csv(nodes_path, dtype=str).fillna("")
    else:
        df_nodes = _read_nodes_tsv(nodes_path)

    filled, total = enrich_metadata(metadata_csv, df_nodes)
    print(f"Enriched metadata: filled {filled} names (total rows: {total}) → {metadata_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

