"""Integration tests for the Sprint 12 benchmarking & paper-builder scripts.

Covers:
- scripts/compare_pipeline_modes.py (build_comparison helper)
- scripts/aggregate_qc_summary.py (scan_qc_dir + write_outputs)
- scripts/build_signature_catalog.py (scan_signatures + write_outputs)
- scripts/build_paper_tables.py (each table builder)

Heavy subprocess paths are exercised by tests/test_pipeline_integration.py;
here we focus on the in-process helpers so the suite stays fast.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def _import_script(name: str):
    """Import a top-level script as a module without adding scripts/ to sys.path."""
    path = _SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- compare_pipeline_modes ----------------------------------------

def test_compare_pipeline_build_comparison() -> None:
    cm = _import_script("compare_pipeline_modes")

    kg = [
        {"compound": "metformin", "disease": "T2D",
         "final_score": 0.45, "confidence_tier": 4},
        {"compound": "aspirin", "disease": "diabetes",
         "final_score": 0.50, "confidence_tier": 4},
    ]
    full = [
        {"compound": "metformin", "disease": "T2D",
         "final_score": 0.72, "confidence_tier": 3,
         "signature_reversal_score": 0.78},
        {"compound": "aspirin", "disease": "diabetes",
         "final_score": 0.55, "confidence_tier": 4,
         "signature_reversal_score": 0.20},
    ]
    rows = cm.build_comparison(kg, full)
    assert len(rows) == 2

    # The row with the larger absolute delta should sort first.
    assert rows[0]["compound"] == "metformin"
    assert rows[0]["delta_score"] == pytest.approx(0.27, abs=0.001)
    assert rows[0]["tier_promoted"] == 1
    assert rows[0]["reversal_score"] == pytest.approx(0.78)

    # aspirin: tier stays at 4 → not promoted.
    assert rows[1]["tier_promoted"] == 0


def test_compare_pipeline_handles_missing_pairs() -> None:
    cm = _import_script("compare_pipeline_modes")

    kg = [{"compound": "a", "disease": "d", "final_score": 0.4, "confidence_tier": 4}]
    full = [{"compound": "b", "disease": "d", "final_score": 0.6, "confidence_tier": 3}]
    rows = cm.build_comparison(kg, full)
    # Union of pairs = 2; missing side defaults to 0/tier-4.
    assert len(rows) == 2


# ---------- aggregate_qc_summary ------------------------------------------

def test_aggregate_qc_handles_missing_dir(tmp_path: Path) -> None:
    agg = _import_script("aggregate_qc_summary")
    rows = agg.scan_qc_dir(tmp_path / "missing")
    assert rows == []


def test_aggregate_qc_parses_json_sidecar(tmp_path: Path) -> None:
    agg = _import_script("aggregate_qc_summary")
    qc_dir = tmp_path / "qc"
    qc_dir.mkdir()
    (qc_dir / "demo_qc.json").write_text(json.dumps({
        "dataset": "demo_dataset",
        "n_cells": 8000, "n_genes": 22000,
        "max_mito_pct": 20, "cells_removed": 350,
        "doublets_flagged": 42,
    }))

    rows = agg.scan_qc_dir(qc_dir)
    assert len(rows) == 1
    assert rows[0]["dataset"] == "demo_dataset"
    assert rows[0]["n_cells"] == 8000


def test_aggregate_qc_writes_outputs(tmp_path: Path) -> None:
    agg = _import_script("aggregate_qc_summary")
    rows = [{
        "dataset": "demo", "n_cells": 100, "n_genes": 2000,
        "mito_threshold": 20, "cells_removed": 5, "doublets_flagged": 2,
        "source_report": "n/a",
    }]
    agg.write_outputs(rows, tmp_path)
    assert (tmp_path / "qc_summary_table.csv").exists()
    assert (tmp_path / "qc_summary_table.md").exists()

    md = (tmp_path / "qc_summary_table.md").read_text()
    assert "demo" in md
    assert "100" in md


# ---------- build_signature_catalog ---------------------------------------

def test_signature_catalog_missing_dir_returns_empty(tmp_path: Path) -> None:
    sc = _import_script("build_signature_catalog")
    rows, full = sc.scan_signatures(tmp_path / "missing")
    assert rows == []
    assert full == []


def test_signature_catalog_parses_disease_signature(tmp_path: Path) -> None:
    sc = _import_script("build_signature_catalog")
    sig_dir = tmp_path / "signatures"
    sig_dir.mkdir()
    sig = {
        "disease": "Disease::DOID:9352", "tissue": "blood",
        "cell_type": "all_cells",
        "up_genes": ["G1", "G2", "G3"], "down_genes": ["G4", "G5"],
        "ranked_genes": [], "pathways": [],
    }
    (sig_dir / "disease_signature.json").write_text(json.dumps(sig))
    rows, _ = sc.scan_signatures(sig_dir)
    assert len(rows) == 1
    assert rows[0]["disease"] == "Disease::DOID:9352"
    assert rows[0]["n_up_genes"] == 3
    assert rows[0]["n_down_genes"] == 2


def test_signature_catalog_parses_cell_type_bundle(tmp_path: Path) -> None:
    sc = _import_script("build_signature_catalog")
    sig_dir = tmp_path / "signatures"
    sig_dir.mkdir()
    bundle = {
        "T_cell": {"disease": "D", "tissue": "blood",
                   "up_genes": ["G1", "G2"], "down_genes": ["G3"],
                   "ranked_genes": [], "pathways": []},
        "B_cell": {"disease": "D", "tissue": "blood",
                   "up_genes": ["G2", "G4"], "down_genes": ["G5"],
                   "ranked_genes": [], "pathways": []},
    }
    (sig_dir / "cell_type_signatures.json").write_text(json.dumps(bundle))
    rows, full = sc.scan_signatures(sig_dir)
    assert len(rows) == 1
    # Union of up genes across cell types = {G1, G2, G4} = 3
    assert rows[0]["n_up_genes"] == 3
    assert rows[0]["n_cell_types"] == 2
    assert full[0]["cell_types"] == ["T_cell", "B_cell"]


# ---------- build_paper_tables --------------------------------------------

def test_build_paper_tables_top_candidates(tmp_path: Path, monkeypatch) -> None:
    bpt = _import_script("build_paper_tables")
    # Stage a minimal top_candidates.csv in a fake artifacts dir.
    art_dir = tmp_path / "artifacts" / "predictions"
    art_dir.mkdir(parents=True)
    csv_path = art_dir / "top_candidates.csv"
    csv_path.write_text(
        "rank,compound,disease,final_score,confidence_tier,kg_rotate_score,qsvc_score\n"
        "1,metformin,T2D,0.7137,3,0.91,0.80\n"
        "2,aspirin,diabetes,0.6711,4,0.85,0.72\n"
    )
    # Builder reads from a relative path — chdir into tmp_path so the test
    # is hermetic.
    monkeypatch.chdir(tmp_path)

    out = bpt.build_table_8(tmp_path / "tables", top_n=20)
    assert out is not None
    tex = out.read_text(encoding="utf-8")
    assert r"\begin{table}" in tex
    assert "metformin" in tex
    assert r"0.7137" in tex


def test_build_paper_tables_mode_delta(tmp_path: Path, monkeypatch) -> None:
    bpt = _import_script("build_paper_tables")
    art_dir = tmp_path / "artifacts" / "predictions"
    art_dir.mkdir(parents=True)
    (art_dir / "mode_comparison.csv").write_text(
        "compound,disease,kg_only_score,kg_omics_score,delta_score,"
        "kg_only_tier,kg_omics_tier,tier_promoted,reversal_score\n"
        "metformin,T2D,0.45,0.71,0.26,4,3,1,0.78\n"
    )
    monkeypatch.chdir(tmp_path)

    out = bpt.build_table_9(tmp_path / "tables")
    assert out is not None
    tex = out.read_text(encoding="utf-8")
    assert r"\label{tab:mode_delta}" in tex
    assert "+0.2600" in tex


def test_build_paper_tables_skips_missing_input(tmp_path: Path, monkeypatch) -> None:
    bpt = _import_script("build_paper_tables")
    monkeypatch.chdir(tmp_path)
    # No artifacts present → builder should return None, not raise.
    assert bpt.build_table_1(tmp_path / "tables") is None
    assert bpt.build_table_4(tmp_path / "tables") is None
    assert bpt.build_table_8(tmp_path / "tables") is None
