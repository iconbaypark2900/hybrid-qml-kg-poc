from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path("scripts/run_rnaseq_external_validation.py")
SPEC = importlib.util.spec_from_file_location("run_rnaseq_external_validation", SCRIPT_PATH)
validation = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["run_rnaseq_external_validation"] = validation
SPEC.loader.exec_module(validation)


def test_pair_aware_permutation_preserves_matched_pair_composition() -> None:
    y = np.array([1, 0, 1, 0, 1, 0], dtype=int)
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p4"])
    rng = np.random.default_rng(7)

    for _ in range(20):
        permuted = validation._pair_aware_permutation(y, groups, rng)
        assert set(permuted[groups == "p1"].tolist()) == {0, 1}
        assert set(permuted[groups == "p2"].tolist()) == {0, 1}
        assert int(permuted.sum()) == int(y.sum())


def test_cluster_bootstrap_preserves_paired_model_comparison() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    groups = np.array(["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"])
    classical_scores = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    quantum_scores = np.array([0.2, 0.8, 0.4, 0.6, 0.6, 0.4, 0.7, 0.3])
    summary, details = validation.cluster_bootstrap(
        y,
        groups,
        {"classical": classical_scores, "qsvc_quantum": quantum_scores},
        {
            "classical": (classical_scores >= 0.5).astype(int),
            "qsvc_quantum": (quantum_scores >= 0.5).astype(int),
        },
        quantum_model="qsvc_quantum",
        classical_model="classical",
        n_bootstrap=100,
        random_state=11,
    )

    assert summary["available"] is True
    assert summary["n_clusters"] == 4
    assert len(details) == 100
    assert summary["metrics"]["delta_roc_auc"]["mean"] < 0


def test_external_validation_rejects_same_development_and_validation_cohort(tmp_path: Path) -> None:
    verdict_path = tmp_path / "verdict.json"
    manifest_path = tmp_path / "manifest.json"
    harmonization_path = tmp_path / "harmonization.json"
    verdict_path.write_text(json.dumps({}), encoding="utf-8")
    manifest_path.write_text(json.dumps({}), encoding="utf-8")
    harmonization_path.write_text(json.dumps({}), encoding="utf-8")
    args = argparse.Namespace(
        development_verdict=str(verdict_path),
        development_manifest=str(manifest_path),
        harmonization_manifest=str(harmonization_path),
        development_cohort="SAME",
        validation_cohort="SAME",
    )

    with pytest.raises(ValueError, match="must differ"):
        validation.run_external_validation(args)
