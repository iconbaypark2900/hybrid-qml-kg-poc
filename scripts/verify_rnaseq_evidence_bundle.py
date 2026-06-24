#!/usr/bin/env python3
"""Verify integrity and claim consistency of the RNA-seq evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(
    *,
    harmonization_manifest_path: str | Path,
    benchmark_dir: str | Path,
    external_validation_path: str | Path,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, evidence: Any) -> None:
        checks.append({"check": name, "status": "pass" if condition else "fail", "evidence": evidence})

    harmonization_path = Path(harmonization_manifest_path)
    benchmark_dir = Path(benchmark_dir)
    external_path = Path(external_validation_path)
    harmonization = _read_json(harmonization_path)
    benchmark_manifest = _read_json(benchmark_dir / "benchmark_manifest.json")
    benchmark_verdict = _read_json(benchmark_dir / "quantum_value_verdict.json")
    audit = _read_json(benchmark_dir / "evidence_audit.json")
    external = _read_json(external_path)
    external_manifest_path = Path(external["outputs"]["manifest"])
    external_manifest = _read_json(external_manifest_path)

    for namespace, manifest in [
        ("harmonization", harmonization),
        ("external", external_manifest),
    ]:
        for key, descriptor in (manifest.get("inputs") or {}).items():
            path = Path(descriptor["path"])
            expected = descriptor.get("sha256")
            check(f"{namespace}_input_exists:{key}", path.exists(), str(path))
            if path.exists() and expected:
                observed = _sha256(path)
                check(
                    f"{namespace}_input_sha256:{key}",
                    observed == expected,
                    {"expected": expected, "observed": observed},
                )

    harmonized_outputs = harmonization.get("outputs") or {}
    development_counts_path = Path(harmonized_outputs["development_normalized_counts"])
    validation_counts_path = Path(harmonized_outputs["validation_normalized_counts"])
    shared_genes_path = Path(harmonized_outputs["shared_gene_universe"])
    development = pd.read_csv(development_counts_path)
    validation = pd.read_csv(validation_counts_path)
    shared_genes = pd.read_csv(shared_genes_path)["gene"].astype(str).tolist()
    development_genes = [str(column) for column in development.columns if column != "sample_id"]
    validation_genes = [str(column) for column in validation.columns if column != "sample_id"]
    check(
        "identical_ordered_gene_universe",
        development_genes == validation_genes == shared_genes,
        {
            "development_genes": len(development_genes),
            "validation_genes": len(validation_genes),
            "manifest_genes": len(shared_genes),
        },
    )
    check("development_sample_count", len(development) == 60, int(len(development)))
    check("validation_sample_count", len(validation) == 155, int(len(validation)))
    check("harmonization_labels_unused", harmonization.get("labels_used") is False, harmonization.get("labels_used"))

    check(
        "benchmark_uses_harmonized_development",
        Path(benchmark_manifest["normalized_counts"]).resolve() == development_counts_path.resolve(),
        benchmark_manifest["normalized_counts"],
    )
    external_inputs = external_manifest.get("inputs") or {}
    check(
        "external_uses_harmonized_development",
        Path(external_inputs["development_counts"]["path"]).resolve() == development_counts_path.resolve(),
        external_inputs["development_counts"]["path"],
    )
    check(
        "external_uses_harmonized_validation",
        Path(external_inputs["validation_counts"]["path"]).resolve() == validation_counts_path.resolve(),
        external_inputs["validation_counts"]["path"],
    )
    check(
        "external_links_harmonization_manifest",
        Path(external_inputs["harmonization_manifest"]["path"]).resolve() == harmonization_path.resolve(),
        external_inputs["harmonization_manifest"]["path"],
    )

    for namespace, outputs in [
        ("benchmark", benchmark_manifest.get("outputs") or {}),
        ("external", external.get("outputs") or {}),
        ("harmonization", harmonized_outputs),
    ]:
        for key, value in outputs.items():
            check(f"{namespace}_output_exists:{key}", Path(value).exists(), value)

    leakage = external.get("leakage_controls") or {}
    expected_leakage = {
        "validation_labels_used_for_feature_selection": False,
        "validation_labels_used_for_hyperparameter_selection": False,
        "validation_samples_used_for_scaler_or_pca_fit": False,
        "validation_samples_used_for_model_fit": False,
        "development_configuration_locked_before_validation": True,
        "validation_de_signature_generated": False,
        "shared_gene_universe_normalization_verified": True,
    }
    check(
        "external_leakage_controls",
        all(leakage.get(key) is value for key, value in expected_leakage.items()),
        leakage,
    )

    check("audit_review_ready", audit.get("readiness") == "review_ready", audit.get("readiness"))
    check(
        "audit_all_gates_pass",
        bool(audit.get("gates")) and all(gate.get("status") == "pass" for gate in audit["gates"]),
        {gate.get("id"): gate.get("status") for gate in audit.get("gates", [])},
    )
    check(
        "no_unsupported_quantum_claim",
        benchmark_verdict.get("quantum_adds_value") is False
        and external.get("verdict", {}).get("external_quantum_adds_value") is False,
        {
            "development_quantum_adds_value": benchmark_verdict.get("quantum_adds_value"),
            "external_quantum_adds_value": external.get("verdict", {}).get("external_quantum_adds_value"),
        },
    )

    failed = [item for item in checks if item["status"] != "pass"]
    return {
        "status": "pass" if not failed else "fail",
        "n_checks": int(len(checks)),
        "n_failed": int(len(failed)),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harmonization-manifest", required=True)
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--external-validation", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = verify_bundle(
        harmonization_manifest_path=args.harmonization_manifest,
        benchmark_dir=args.benchmark_dir,
        external_validation_path=args.external_validation,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
