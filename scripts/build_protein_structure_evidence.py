#!/usr/bin/env python3
"""Build AlphaFold-like local protein structure evidence from a registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structure_layer import build_protein_structure_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="artifacts/structures/alphafold/brca_anastrozole_targets/structure_registry.json")
    parser.add_argument("--out", default="artifacts/repurposing/brca_external_validation/protein_structure_evidence.json")
    args = parser.parse_args()

    evidence = build_protein_structure_evidence(args.registry)
    payload = {
        "schema_version": "1.0",
        "status": "ready" if any(row.get("parse_success") for row in evidence) else "not_ready",
        "registry": args.registry,
        "protein_count": len(evidence),
        "parsed_count": sum(1 for row in evidence if row.get("parse_success")),
        "proteins": evidence,
        "claim_policy": "Local structure evidence is for hypothesis review only; it does not prove therapeutic efficacy.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(out), "protein_count": payload["protein_count"], "parsed_count": payload["parsed_count"]}, indent=2))
    return 0 if payload["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
