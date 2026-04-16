#!/usr/bin/env python3
"""
Append-only provenance registry for pipeline benchmark runs.

Every run records embedding config, reduction config, model config, backend,
circuit metadata, and metrics to ``results/benchmark_registry.jsonl``.

Usage:
    python scripts/benchmark_registry.py --list
    python scripts/benchmark_registry.py --list --path results/benchmark_registry.jsonl
"""

import argparse
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_REGISTRY = _PROJECT_ROOT / "results" / "benchmark_registry.jsonl"


def register_run(
    *,
    run_id: str,
    relation: str,
    embedding: dict,
    reduction: dict,
    model: dict,
    backend: dict,
    metrics: dict,
    negative_sampling: dict | None = None,
    circuit: dict | None = None,
    split: dict | None = None,
    notes: str = "",
    registry_path: str | None = None,
) -> Path:
    """Append one provenance record to the registry file.

    Returns the ``Path`` to the registry file.
    """
    path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.version,
        "relation": relation,
        "embedding": embedding,
        "reduction": reduction,
        "model": model,
        "backend": backend,
        "metrics": metrics,
        "negative_sampling": negative_sampling or {"strategy": "random"},
        "circuit": circuit,
        "split": split or {},
        "notes": notes,
    }

    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")

    return path


def load_registry(registry_path: str | None = None) -> list[dict]:
    """Read all records, oldest first. Returns ``[]`` if file is absent."""
    path = Path(registry_path) if registry_path else _DEFAULT_REGISTRY
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarise_registry(registry_path: str | None = None) -> None:
    """Print a formatted table of all runs to stdout."""
    records = load_registry(registry_path)
    if not records:
        print("(no runs recorded)")
        return

    header = f"{'run_id':<20} {'model':<30} {'type':<12} {'backend':<25} {'neg_strategy':<16} {'PR-AUC':>8}"
    print(header)
    print("-" * len(header))
    for r in records:
        m = r.get("model", {})
        b = r.get("backend", {})
        ns = r.get("negative_sampling", {})
        pr = r.get("metrics", {}).get("pr_auc")
        print(
            f"{r.get('run_id', '?'):<20} "
            f"{m.get('name', '?'):<30} "
            f"{m.get('type', '?'):<12} "
            f"{b.get('name', '?'):<25} "
            f"{ns.get('strategy', '?'):<16} "
            f"{pr if pr is not None else 'N/A':>8}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark registry CLI")
    parser.add_argument("--list", action="store_true", help="List all recorded runs")
    parser.add_argument("--path", type=str, default=None, help="Custom registry JSONL path")
    args = parser.parse_args()

    if args.list:
        summarise_registry(args.path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
