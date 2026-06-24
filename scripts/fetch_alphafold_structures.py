#!/usr/bin/env python3
"""Fetch public AlphaFold DB structures for mapped KG targets.

The output is a local structure registry compatible with ``structure_layer``.
Network access is only used during this import step; downstream ranking and UI
inspection read local PDB artifacts from disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
LICENSE_NOTE = "AlphaFold DB public model cached locally; review AlphaFold DB terms before redistribution or model-training use."
SOURCE_TOOL = "alphafold_db_cached"

FetchJson = Callable[[str], Any]
FetchBytes = Callable[[str], bytes]


def fetch_alphafold_structures(
    *,
    target_map: str | Path,
    out_dir: str | Path,
    registry_out: str | Path | None = None,
    fetch_json: FetchJson | None = None,
    fetch_bytes: FetchBytes | None = None,
) -> dict[str, Any]:
    """Download PDB structures and write a local registry."""

    fetch_json = fetch_json or _fetch_json
    fetch_bytes = fetch_bytes or _fetch_bytes
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    registry_path = Path(registry_out) if registry_out else out_path / "structure_registry.json"

    targets = _read_target_map(target_map)
    artifacts: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = []
    for target in targets:
        uniprot_id = target["uniprot_id"]
        api_url = API_TEMPLATE.format(uniprot_id=uniprot_id)
        api_payload = fetch_json(api_url)
        if not isinstance(api_payload, list) or not api_payload:
            raise ValueError(f"AlphaFold DB API returned no predictions for {uniprot_id}")
        entry = api_payload[0]
        pdb_url = entry.get("pdbUrl")
        if not pdb_url:
            raise ValueError(f"AlphaFold DB API prediction missing pdbUrl for {uniprot_id}")
        model_entity_id = str(entry.get("modelEntityId") or f"AF-{uniprot_id}-F1")
        pdb_name = f"{target['target_id'].replace('::', '_')}_{model_entity_id}.pdb"
        pdb_path = out_path / pdb_name
        content = fetch_bytes(str(pdb_url))
        pdb_path.write_bytes(content)
        checksum = hashlib.sha256(content).hexdigest()
        artifacts.append(
            {
                "target_id": target["target_id"],
                "target_name": target.get("target_name") or target["target_id"],
                "sequence_hash": f"sha256:{entry.get('sequenceChecksum') or checksum}",
                "source_tool": SOURCE_TOOL,
                "source_version": str(entry.get("latestVersion") or entry.get("modelCreatedDate") or "unknown"),
                "artifact_path": pdb_path.name,
                "artifact_format": "pdb",
                "license_note": LICENSE_NOTE,
                "confidence": {
                    "score_type": "pLDDT",
                    "global_metric_value": entry.get("globalMetricValue"),
                    "fraction_plddt_very_high": entry.get("fractionPlddtVeryHigh"),
                    "fraction_plddt_confident": entry.get("fractionPlddtConfident"),
                    "fraction_plddt_low": entry.get("fractionPlddtLow"),
                    "fraction_plddt_very_low": entry.get("fractionPlddtVeryLow"),
                },
                "metadata": {
                    "uniprot_id": uniprot_id,
                    "gene_symbol": target.get("gene_symbol"),
                    "expected_residue_count": _sequence_length(entry),
                    "model_entity_id": model_entity_id,
                    "model_created_date": entry.get("modelCreatedDate"),
                    "organism": entry.get("organismScientificName"),
                    "api_url": api_url,
                    "pdb_url": pdb_url,
                    "cif_url": entry.get("cifUrl"),
                    "pae_doc_url": entry.get("paeDocUrl"),
                    "sha256": checksum,
                },
            }
        )
        downloads.append(
            {
                "target_id": target["target_id"],
                "target_name": target.get("target_name"),
                "uniprot_id": uniprot_id,
                "pdb_url": pdb_url,
                "artifact_path": str(pdb_path),
                "sha256": checksum,
            }
        )

    registry = {
        "schema_version": "1.0",
        "source": "alphafold_db_public_api_cached_local_pdb",
        "source_api": API_TEMPLATE,
        "claim_policy": "Local structure artifacts support research hypothesis inspection only; they do not prove therapeutic efficacy.",
        "artifacts": artifacts,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "ready" if artifacts else "not_ready",
        "target_count": len(targets),
        "artifact_count": len(artifacts),
        "registry": str(registry_path),
        "out_dir": str(out_path),
        "downloads": downloads,
        "claim_policy": registry["claim_policy"],
    }
    registry_path.with_suffix(registry_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _read_target_map(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"target_id", "uniprot_id"}
    missing = sorted(required - set(rows[0].keys() if rows else []))
    if missing:
        raise ValueError(f"target map missing required columns: {missing}")
    targets = []
    for row in rows:
        target_id = str(row.get("target_id") or "").strip()
        uniprot_id = str(row.get("uniprot_id") or "").strip()
        if not target_id or not uniprot_id:
            continue
        targets.append({key: str(value or "").strip() for key, value in row.items()})
    if not targets:
        raise ValueError("target map did not contain any target_id/uniprot_id rows")
    return targets


def _fetch_json(url: str) -> Any:
    with urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch_bytes(url: str) -> bytes:
    with urlopen(url, timeout=120) as response:
        return response.read()


def _sequence_length(entry: dict[str, Any]) -> int | None:
    sequence = entry.get("uniprotSequence") or entry.get("sequence")
    if isinstance(sequence, str) and sequence:
        return len(sequence)
    start = entry.get("sequenceStart")
    end = entry.get("sequenceEnd")
    if isinstance(start, int) and isinstance(end, int) and end >= start:
        return end - start + 1
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-map", default="artifacts/repurposing/brca_external_validation/anastrozole_target_uniprot_map.csv")
    parser.add_argument("--out-dir", default="artifacts/structures/alphafold/brca_anastrozole_targets")
    parser.add_argument("--registry-out", default="artifacts/structures/alphafold/brca_anastrozole_targets/structure_registry.json")
    args = parser.parse_args()

    manifest = fetch_alphafold_structures(
        target_map=args.target_map,
        out_dir=args.out_dir,
        registry_out=args.registry_out,
    )
    print(json.dumps({"status": manifest["status"], "artifact_count": manifest["artifact_count"], "registry": manifest["registry"]}, indent=2))
    return 0 if manifest["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
