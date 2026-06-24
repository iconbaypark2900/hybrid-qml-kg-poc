#!/usr/bin/env python3
"""Build candidate-to-target provenance from local Hetionet artifacts.

This script maps ranked drug-repurposing candidates to shared compound/disease
Gene targets using only local KG files. It does not infer clinical efficacy; the
output is provenance for structure coverage and hypothesis review.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

COMPOUND_GENE_EDGES = {"CbG", "CdG", "CuG"}
DISEASE_GENE_EDGES = {"DaG", "DdG", "DuG"}
TARGET_SOURCE = "hetionet_shared_compound_disease_gene_edges"


def normalize_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def build_candidate_target_map(
    *,
    ranking_comparison: str | Path,
    nodes: str | Path,
    edges: str | Path,
    out: str | Path,
    disease_name: str = "breast cancer",
    score_column: str = "kg_omics_final_score",
    top_k: int | None = None,
) -> dict[str, Any]:
    ranking = pd.read_csv(ranking_comparison)
    required_cols = {"candidate_id", "compound", score_column}
    missing_cols = sorted(required_cols - set(ranking.columns))
    if missing_cols:
        raise ValueError(f"ranking_comparison missing required columns: {missing_cols}")

    ranking = ranking.sort_values(score_column, ascending=False)
    if top_k and top_k > 0:
        ranking = ranking.head(top_k)

    compound_names = {normalize_name(value) for value in ranking["compound"].dropna().unique()}
    compound_by_name, disease_id, disease_label = _read_node_maps(nodes, compound_names, disease_name)
    candidate_compound_ids = {value for value in compound_by_name.values() if value}

    compound_to_genes, disease_genes = _read_gene_edges(edges, candidate_compound_ids, disease_id)
    rows = []
    mapped_count = 0
    for _, row in ranking.iterrows():
        candidate_id = str(row["candidate_id"])
        compound_name = str(row["compound"])
        compound_id = compound_by_name.get(normalize_name(compound_name))
        target_ids: list[str] = []
        notes: list[str] = []
        if not disease_id:
            mapping_status = "disease_not_found"
            notes.append(f"Disease name {disease_name!r} was not found in local Hetionet nodes.")
        elif not compound_id:
            mapping_status = "compound_not_found"
            notes.append("Compound name was not found in local Hetionet nodes.")
        else:
            compound_genes = compound_to_genes.get(compound_id, set())
            target_ids = sorted(compound_genes & disease_genes)
            if target_ids:
                mapping_status = "mapped"
                mapped_count += 1
            else:
                mapping_status = "no_shared_targets"
                notes.append("Compound and disease were found, but no shared Gene targets matched the configured KG edge types.")
        rows.append(
            {
                "candidate_id": candidate_id,
                "compound_name": compound_name,
                "compound_kg_id": compound_id or "",
                "disease_name": disease_label or disease_name,
                "disease_kg_id": disease_id or "",
                "mapping_status": mapping_status,
                "target_ids": "|".join(target_ids),
                "target_count": len(target_ids),
                "compound_gene_count": len(compound_to_genes.get(compound_id or "", set())),
                "disease_gene_count": len(disease_genes),
                "target_source": TARGET_SOURCE,
                "notes": " | ".join(notes),
            }
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else _fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "schema_version": "1.0",
        "status": "ready" if mapped_count else "not_ready",
        "ranking_comparison": str(ranking_comparison),
        "nodes": str(nodes),
        "edges": str(edges),
        "out": str(out_path),
        "disease_name": disease_label or disease_name,
        "disease_kg_id": disease_id,
        "candidate_count": len(rows),
        "mapped_candidate_count": mapped_count,
        "target_source": TARGET_SOURCE,
        "compound_gene_edges": sorted(COMPOUND_GENE_EDGES),
        "disease_gene_edges": sorted(DISEASE_GENE_EDGES),
        "claim_policy": "Candidate targets are KG provenance for research hypotheses only; they are not clinical validation.",
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _fieldnames() -> list[str]:
    return [
        "candidate_id",
        "compound_name",
        "compound_kg_id",
        "disease_name",
        "disease_kg_id",
        "mapping_status",
        "target_ids",
        "target_count",
        "compound_gene_count",
        "disease_gene_count",
        "target_source",
        "notes",
    ]


def _read_node_maps(
    nodes: str | Path,
    compound_names: set[str],
    disease_name: str,
) -> tuple[dict[str, str], str | None, str | None]:
    compound_by_name: dict[str, str] = {}
    disease_id: str | None = None
    disease_label: str | None = None
    normalized_disease = normalize_name(disease_name)
    with Path(nodes).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            kind = row.get("kind")
            name = row.get("name", "")
            normalized = normalize_name(name)
            if kind == "Compound" and normalized in compound_names:
                compound_by_name[normalized] = row.get("id", "")
            elif kind == "Disease" and normalized == normalized_disease:
                disease_id = row.get("id") or None
                disease_label = name
    return compound_by_name, disease_id, disease_label


def _read_gene_edges(
    edges: str | Path,
    compound_ids: set[str],
    disease_id: str | None,
) -> tuple[dict[str, set[str]], set[str]]:
    compound_to_genes: dict[str, set[str]] = defaultdict(set)
    disease_genes: set[str] = set()
    with Path(edges).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            source = row.get("source", "")
            metaedge = row.get("metaedge", "")
            target = row.get("target", "")
            if target.startswith("Gene::") and source in compound_ids and metaedge in COMPOUND_GENE_EDGES:
                compound_to_genes[source].add(target)
            elif disease_id and target.startswith("Gene::") and source == disease_id and metaedge in DISEASE_GENE_EDGES:
                disease_genes.add(target)
    return dict(compound_to_genes), disease_genes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-comparison", default="artifacts/benchmarks/rnaseq_quantum_tcga_brca_60_harmonized/ranking_comparison.csv")
    parser.add_argument("--nodes", default="data/hetionet-v1.0-nodes.tsv")
    parser.add_argument("--edges", default="data/hetionet-v1.0-edges.sif")
    parser.add_argument("--out", default="artifacts/repurposing/brca_external_validation/candidate_target_map.csv")
    parser.add_argument("--disease-name", default="breast cancer")
    parser.add_argument("--score-column", default="kg_omics_final_score")
    parser.add_argument("--top-k", type=int, default=0, help="Limit ranked candidates before mapping; 0 maps all rows.")
    args = parser.parse_args()

    manifest = build_candidate_target_map(
        ranking_comparison=args.ranking_comparison,
        nodes=args.nodes,
        edges=args.edges,
        out=args.out,
        disease_name=args.disease_name,
        score_column=args.score_column,
        top_k=args.top_k or None,
    )
    print(json.dumps({"status": manifest["status"], "out": manifest["out"], "candidate_count": manifest["candidate_count"], "mapped_candidate_count": manifest["mapped_candidate_count"]}, indent=2))
    return 0 if manifest["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
