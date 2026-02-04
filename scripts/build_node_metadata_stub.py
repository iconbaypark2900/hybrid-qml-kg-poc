#!/usr/bin/env python3
"""
Generate a node metadata stub CSV for the dashboard.

This project ships Hetionet edges (data/hetionet-v1.0-edges.sif) but not full node-name metadata.
This script extracts unique node IDs from the edge list and writes a CSV users can optionally
fill in with human-readable names.

Output columns:
  - node_id
  - name (blank)
  - namespace (best-effort)
  - external_url (best-effort resolver link)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def node_type(node_id: str) -> str:
    return node_id.split("::", 1)[0] if "::" in node_id else "Unknown"


def infer_namespace_and_url(node_id: str) -> tuple[str, str]:
    """
    Best-effort resolver URLs. We avoid hard dependencies / API calls.
    """
    t = node_type(node_id)
    local_id = node_id.split("::", 1)[1] if "::" in node_id else node_id

    if t == "Compound" and local_id.startswith("DB"):
        # DrugBank often requires license for full access; link still helps.
        return "DrugBank", f"https://go.drugbank.com/drugs/{local_id}"
    if t == "Disease" and local_id.startswith("DOID:"):
        return "DOID", f"https://disease-ontology.org/?id={local_id}"
    if t == "Gene" and local_id.isdigit():
        return "Entrez", f"https://www.ncbi.nlm.nih.gov/gene/{local_id}"
    if t == "Anatomy" and local_id.startswith("UBERON:"):
        return "UBERON", f"https://www.ebi.ac.uk/ols/ontologies/uberon/terms?iri=http://purl.obolibrary.org/obo/{local_id.replace(':','_')}"
    if t in ("Biological Process", "Molecular Function", "Cellular Component") and local_id.startswith("GO:"):
        return "GO", f"https://www.ebi.ac.uk/QuickGO/term/{local_id}"
    if t == "Side Effect" and local_id.startswith("C"):
        # UMLS CUIs often need credentials for some portals; still provide a common lookup pattern.
        return "UMLS", f"https://uts.nlm.nih.gov/uts/umls/concept/{local_id}"
    if t == "Symptom" and local_id.startswith("D"):
        return "MeSH", f"https://meshb.nlm.nih.gov/record/ui?ui={local_id}"
    if t == "Pathway":
        return "Pathway", ""
    if t == "Pharmacologic Class" and local_id.startswith("N"):
        return "NDF-RT", ""
    return "", ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=str, default="data/hetionet-v1.0-edges.sif", help="Path to Hetionet edges file")
    ap.add_argument("--out", type=str, default="data/hetionet_nodes_metadata.csv", help="Output CSV path")
    ap.add_argument("--max_nodes", type=int, default=200000, help="Safety cap on number of unique nodes")
    args = ap.parse_args()

    edges_path = Path(args.edges)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()

    # Stream the TSV; first line may be a header.
    with edges_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        has_header = header[:3] == ["source", "metaedge", "target"]
        if not has_header:
            # rewind
            f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            s, _, t = parts[0], parts[1], parts[2]
            if s and s not in seen:
                seen.add(s)
            if t and t not in seen:
                seen.add(t)
            if len(seen) >= int(args.max_nodes):
                break

    rows = []
    for nid in sorted(seen):
        ns, url = infer_namespace_and_url(nid)
        rows.append(
            {
                "node_id": nid,
                "name": "",
                "namespace": ns,
                "external_url": url,
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["node_id", "name", "namespace", "external_url"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} nodes to {out_path}")


if __name__ == "__main__":
    main()

