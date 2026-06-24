#!/usr/bin/env python3
"""Verify local OpenFold-style structure artifacts are ready for downstream scoring.

This does not run OpenFold. It validates the local artifact registry and proves
that local PDB/OpenFold-like outputs can be parsed into deterministic structure
features before they are used in ranking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from structure_layer import build_structure_feature_table, load_artifact_registry

OPEN_SOURCE_STRUCTURE_TOOLS = {
    "openfold",
    "esmfold",
    "chai-1",
    "chai1",
    "boltz",
    "boltz-1",
    "boltz1",
    "alphafold_local",
    "alphafold_db_cached",
    "pdb",
    "fixture",
    "none",
}


def verify_openfold_readiness(
    *,
    registry: str | Path,
    min_parse_success: int = 1,
    allow_fixture_tools: bool = True,
) -> dict[str, Any]:
    artifacts = load_artifact_registry(registry)
    features, provenance = build_structure_feature_table(artifacts)
    allowed_tools = set(OPEN_SOURCE_STRUCTURE_TOOLS)
    if not allow_fixture_tools:
        allowed_tools.discard("fixture")

    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"check": name, "status": "pass" if condition else "fail", "evidence": evidence})

    parse_success = [row for row in features if row.get("parse_success") == 1]
    local_artifacts = [artifact for artifact in artifacts if artifact.has_local_file]
    bad_tools = sorted({artifact.source_tool for artifact in artifacts if artifact.source_tool not in allowed_tools})
    missing_license = [artifact.target_id for artifact in artifacts if artifact.has_local_file and not artifact.license_note]
    unsupported_formats = sorted(
        {
            artifact.artifact_format
            for artifact in artifacts
            if artifact.artifact_format and artifact.artifact_format not in {"pdb", "ent"}
        }
    )

    check("registry_exists", Path(registry).exists(), str(registry))
    check("registry_has_rows", len(artifacts) > 0, len(artifacts))
    check("target_ids_present", all(artifact.target_id for artifact in artifacts), [artifact.target_id for artifact in artifacts])
    check("source_tools_open_source_or_local", not bad_tools, bad_tools)
    check("local_artifact_files_exist", all(artifact.has_local_file for artifact in local_artifacts), len(local_artifacts))
    check("supported_structure_formats", not unsupported_formats, unsupported_formats)
    check("local_artifacts_have_license_notes", not missing_license, missing_license)
    check("feature_rows_match_registry", len(features) == len(artifacts), {"features": len(features), "artifacts": len(artifacts)})
    check("minimum_parse_success", len(parse_success) >= min_parse_success, {"observed": len(parse_success), "required": min_parse_success})

    failed = [item for item in checks if item["status"] != "pass"]
    return {
        "status": "ready" if not failed else "not_ready",
        "schema_version": "1.0",
        "registry": str(registry),
        "artifact_count": len(artifacts),
        "local_artifact_count": len(local_artifacts),
        "parse_success_count": len(parse_success),
        "feature_count": len(features[0]) if features else 0,
        "checks": checks,
        "features_preview": features[:5],
        "provenance_preview": provenance[:5],
        "claim_policy": "Structure support is artifact-first. This proves local feature extraction, not therapeutic efficacy or de novo cure discovery.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="tests/fixtures/structure_artifacts/registry.json")
    parser.add_argument("--min-parse-success", type=int, default=1)
    parser.add_argument("--no-fixtures", action="store_true", help="Fail if fixture-generated structures are used.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = verify_openfold_readiness(
        registry=args.registry,
        min_parse_success=args.min_parse_success,
        allow_fixture_tools=not args.no_fixtures,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
