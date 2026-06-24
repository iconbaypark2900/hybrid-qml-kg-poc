#!/usr/bin/env python3
"""Discover open GEO RNA-seq cohorts through the public NCBI E-utilities API."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = (
    "breast neoplasms[MeSH Terms] AND "
    "expression profiling by high throughput sequencing[Filter] AND "
    "gse[Entry Type] AND (normal[All Fields] OR adjacent[All Fields])"
)


def _get_json(endpoint: str, params: Dict[str, Any], *, attempts: int = 3) -> Dict[str, Any]:
    url = f"{EUTILS}/{endpoint}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "hybrid-qml-kg-poc/geo-discovery"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except Exception as exc:  # pragma: no cover - exercised only on network failure
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise RuntimeError(f"NCBI request failed after {attempts} attempts: {url}") from last_error


def _chunks(values: List[str], size: int) -> Iterable[List[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def discover_geo_cohorts(query: str, *, retmax: int, min_samples: int) -> Dict[str, Any]:
    search = _get_json(
        "esearch.fcgi",
        {"db": "gds", "retmode": "json", "retmax": int(retmax), "term": query},
    )["esearchresult"]
    ids = list(search.get("idlist", []))
    records: List[Dict[str, Any]] = []
    for batch in _chunks(ids, 100):
        payload = _get_json(
            "esummary.fcgi",
            {"db": "gds", "retmode": "json", "id": ",".join(batch)},
        )["result"]
        for uid in batch:
            item = payload.get(uid, {})
            samples = item.get("samples") or []
            n_samples = item.get("n_samples")
            if n_samples is None:
                n_samples = len(samples) if isinstance(samples, list) else 0
            n_samples = int(n_samples or 0)
            if n_samples < min_samples:
                continue
            records.append(
                {
                    "accession": item.get("accession"),
                    "title": item.get("title"),
                    "summary": item.get("summary"),
                    "n_samples": n_samples,
                    "platform": item.get("gpl"),
                    "organism": item.get("taxon"),
                    "publication_date": item.get("pdat"),
                    "supplementary_file": item.get("suppfile"),
                    "ftp_link": item.get("ftplink"),
                    "uid": uid,
                }
            )
    records.sort(key=lambda row: (-int(row["n_samples"]), str(row.get("accession") or "")))
    return {
        "source": "NCBI GEO via E-utilities",
        "query": query,
        "search_count": int(search.get("count") or 0),
        "returned_ids": len(ids),
        "min_samples": int(min_samples),
        "cohorts": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--retmax", type=int, default=200)
    parser.add_argument("--min-samples", type=int, default=40)
    parser.add_argument("--out", default=None, help="Optional JSON output path; results are always printed.")
    args = parser.parse_args()

    result = discover_geo_cohorts(args.query, retmax=args.retmax, min_samples=args.min_samples)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
