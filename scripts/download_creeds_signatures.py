#!/usr/bin/env python3
"""Download public CREEDS drug perturbation signatures for RNA-seq ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://maayanlab.cloud/CREEDS/download/single_drug_perturbations-v1.0.json"
FALLBACK_URL = "http://amp.pharm.mssm.edu/CREEDS/download/single_drug_perturbations-v1.0.json"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "external" / "creeds" / "single_drug_perturbations-v1.0.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out_path: Path, *, timeout: int = 300) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "hybrid-qml-kg-poc/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    out_path.write_bytes(payload)
    records = json.loads(payload.decode("utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected JSON list from CREEDS, got {type(records)}")
    manifest = {
        "url": url,
        "path": str(out_path.relative_to(REPO_ROOT)),
        "sha256": _sha256(out_path),
        "n_records": len(records),
        "human_records": sum(1 for r in records if str(r.get("organism", "")).lower() == "human"),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--fallback-url", default=FALLBACK_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    if out_path.exists() and out_path.stat().st_size > 0:
        records = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"CREEDS already present ({len(records)} records) → {out_path.relative_to(REPO_ROOT)}")
        return 0

    for url in (args.url, args.fallback_url):
        if not url:
            continue
        try:
            manifest = download(url, out_path)
            print(json.dumps(manifest, indent=2))
            return 0
        except Exception as exc:
            print(f"Download failed for {url}: {exc}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
