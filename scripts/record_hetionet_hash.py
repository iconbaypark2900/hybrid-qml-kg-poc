#!/usr/bin/env python3
"""Record SHA-256 content hashes of the Hetionet snapshot.

Per ``preregistration/osf_preregistration_v1.md`` §3.1 and §9.2, the
manuscript's reproducibility appendix must record a content hash of the
Hetionet snapshot used. This script computes and writes those hashes to
``docs/reproducibility/hetionet_snapshot.md``.

Idempotent — re-running produces the same output if the data files have
not changed. Run before OSF preregistration submission and again before
the pre-submission clean-room reproducibility check.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_FILES = (
    "data/hetionet-v1.0-edges.sif",
    "data/hetionet-v1.0-nodes.tsv",
)
OUTPUT_PATH = "docs/reproducibility/hetionet_snapshot.md"


def sha256_file(path: str, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    rows: list[dict] = []
    for rel in DATA_FILES:
        abspath = os.path.join(REPO_ROOT, rel)
        if not os.path.isfile(abspath):
            print(f"ERROR: missing file {abspath}", file=sys.stderr)
            return 1
        size = os.path.getsize(abspath)
        mtime = dt.datetime.fromtimestamp(
            os.path.getmtime(abspath), tz=dt.timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
        digest = sha256_file(abspath)
        rows.append(
            {"path": rel, "size_bytes": size, "mtime_utc": mtime, "sha256": digest}
        )

    today = dt.date.today().isoformat()
    out_abs = os.path.join(REPO_ROOT, OUTPUT_PATH)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)

    lines = [
        "# Hetionet v1.0 snapshot — content hashes",
        "",
        f"Recorded: {today}",
        "",
        "Per `preregistration/osf_preregistration_v1.md` §3.1 and §9.2, the",
        "Hetionet snapshot used by this project is identified by its SHA-256",
        "content hash. The files below are tracked outside Git (see",
        "`.gitignore`); these hashes are the canonical identifier for",
        "reproducibility.",
        "",
        "| File | Size (bytes) | Last modified (UTC) | SHA-256 |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['path']}` | {row['size_bytes']:,} | {row['mtime_utc']} | "
            f"`{row['sha256']}` |"
        )
    lines += [
        "",
        "## Regenerating",
        "",
        "```bash",
        "python scripts/record_hetionet_hash.py",
        "```",
        "",
        "Idempotent — re-running produces the same hashes if the data files",
        "have not changed. Run before OSF preregistration submission and",
        "again before the pre-submission clean-room reproducibility check.",
        "",
    ]

    content = "\n".join(lines)
    with open(out_abs, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Wrote {OUTPUT_PATH}")
    for row in rows:
        print(f"  {row['path']}: {row['sha256']}  ({row['size_bytes']:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
