"""End-to-end integration test for scripts/run_full_repurposing_pipeline.py.

Invokes the orchestrator as a subprocess (matches how the smoke test runs it)
and asserts the artifacts are well-formed in both modes.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


_PIPELINE = "scripts/run_full_repurposing_pipeline.py"
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_pipeline(mode: str, output: Path, top_n: int = 5) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, _PIPELINE,
         "--mode", mode, "--top-n", str(top_n),
         "--output", str(output)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("mode", ["kg-only", "kg+omics"])
def test_pipeline_writes_required_artifacts(tmp_path: Path, mode: str) -> None:
    result = _run_pipeline(mode, tmp_path)
    assert result.returncode == 0, f"pipeline failed:\n{result.stderr}"

    for name in ("top_candidates.csv", "top_candidates.json",
                 "run_summary.json", "final_repurposing_report.md"):
        assert (tmp_path / name).exists(), f"missing artifact: {name}"


def test_pipeline_summary_well_formed(tmp_path: Path) -> None:
    _run_pipeline("kg+omics", tmp_path, top_n=8)
    summary = json.loads((tmp_path / "run_summary.json").read_text())

    assert summary["mode"] == "kg+omics"
    assert summary["n_candidates"] > 0
    assert summary["top_compound"] is not None
    assert 0.0 <= summary["top_score"] <= 1.0
    # All tiers must sum to n_candidates.
    tiers = summary["tier_distribution"]
    assert sum(tiers.values()) == summary["n_candidates"]


def test_pipeline_kg_only_matches_baseline_shape(tmp_path: Path) -> None:
    """kg-only mode must produce the same set of candidates as kg+omics
    (same compound-disease pairs, only the score differs)."""
    out_kg = tmp_path / "kg_only"
    out_full = tmp_path / "kg_omics"
    out_kg.mkdir()
    out_full.mkdir()

    _run_pipeline("kg-only", out_kg, top_n=12)
    _run_pipeline("kg+omics", out_full, top_n=12)

    kg = json.loads((out_kg / "top_candidates.json").read_text())
    full = json.loads((out_full / "top_candidates.json").read_text())

    assert len(kg) == len(full)
    # Same compound set (order may differ due to omics features).
    assert {c["compound"] for c in kg} == {c["compound"] for c in full}


def test_pipeline_kg_only_zeroes_reversal(tmp_path: Path) -> None:
    """In kg-only mode every candidate's reversal scores must be 0."""
    _run_pipeline("kg-only", tmp_path)
    candidates = json.loads((tmp_path / "top_candidates.json").read_text())

    for c in candidates:
        assert c["signature_reversal_score"] == 0.0
        assert c["cell_type_reversal_score"] == 0.0
        assert c["pathway_reversal_score"] == 0.0


def test_pipeline_top_n_respected(tmp_path: Path) -> None:
    _run_pipeline("kg+omics", tmp_path, top_n=3)
    candidates = json.loads((tmp_path / "top_candidates.json").read_text())
    assert len(candidates) == 3
