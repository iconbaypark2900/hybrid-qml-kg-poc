"""
Resolve latest pipeline artifacts under results/.

Rules (aligned with benchmarking/dashboard.py):
- ``latest_run.csv``: single path ``<writable_results_dir>/latest_run.csv`` (same as Streamlit).
- ``optimized_results_*.json``: newest by filesystem mtime among
  ``<writable_results_dir>`` and ``<project_root>/results`` when they differ
  (e.g. read-only mount still has JSON under repo ``results/``).
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def get_results_dir() -> Path:
    """Writable results directory; falls back to temp if repo ``results/`` is not writable."""
    preferred = PROJECT_ROOT / "results"
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_check"
        probe.write_text("")
        probe.unlink()
        return preferred
    except (OSError, PermissionError):
        tmp = Path(tempfile.gettempdir()) / "hybrid_qml_kg_results"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp


def _optimized_json_search_dirs() -> List[Path]:
    rd = get_results_dir()
    out = [rd]
    pr = PROJECT_ROOT / "results"
    if pr.resolve() != rd.resolve():
        out.append(pr)
    return out


def find_latest_optimized_json_path() -> Optional[Path]:
    files: List[Path] = []
    for d in _optimized_json_search_dirs():
        if d.exists():
            files.extend(d.glob("optimized_results_*.json"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _ranking_from_blob(out: Dict[str, Any]) -> List[Dict[str, Any]]:
    ranking = out.get("ranking") or []
    if ranking:
        return ranking
    if not ("classical_results" in out or "quantum_results" in out or "ensemble_results" in out):
        return []
    ranking = []
    for name, res in (out.get("classical_results") or {}).items():
        if isinstance(res, dict) and res.get("status") == "success":
            tm = res.get("test_metrics") or {}
            ranking.append(
                {
                    "name": name,
                    "type": "classical",
                    "pr_auc": tm.get("pr_auc", 0.0),
                    "accuracy": tm.get("accuracy", 0.0),
                    "fit_time": res.get("fit_seconds", 0.0),
                }
            )
    for name, res in (out.get("quantum_results") or {}).items():
        if isinstance(res, dict) and res.get("status") == "success":
            tm = res.get("test_metrics") or {}
            ranking.append(
                {
                    "name": name,
                    "type": "quantum",
                    "pr_auc": tm.get("pr_auc", 0.0),
                    "accuracy": tm.get("accuracy", 0.0),
                    "fit_time": res.get("fit_seconds", 0.0),
                }
            )
    for name, res in (out.get("ensemble_results") or {}).items():
        if isinstance(res, dict) and res.get("status") == "success":
            tm = res.get("test_metrics") or {}
            ranking.append(
                {
                    "name": name,
                    "type": "ensemble",
                    "pr_auc": tm.get("pr_auc", 0.0),
                    "accuracy": tm.get("accuracy", 0.0),
                    "fit_time": res.get("fit_seconds", 0.0),
                }
            )
    ranking.sort(key=lambda x: x.get("pr_auc", 0.0), reverse=True)
    return ranking


def load_latest_optimized_summary() -> Optional[Dict[str, Any]]:
    """Load newest optimized_results JSON and return a small summary dict."""
    path = find_latest_optimized_json_path()
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    ranking = _ranking_from_blob(raw)
    return {
        "path": str(path.resolve()),
        "mtime_epoch": path.stat().st_mtime,
        "ranking": ranking,
        "relation": raw.get("relation"),
        "timestamp": raw.get("timestamp") or raw.get("run_timestamp"),
    }


def load_latest_csv_row() -> Optional[Dict[str, Any]]:
    """First (and typically only) row of ``latest_run.csv`` as a string-keyed dict."""
    csv_path = get_results_dir() / "latest_run.csv"
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
        if not row:
            return None
        return {str(k): (v if v != "" else None) for k, v in row.items()}
    except Exception:
        return None


def get_latest_run_snapshot() -> Dict[str, Any]:
    """Payload for ``GET /runs/latest``."""
    results_dir = str(get_results_dir().resolve())
    csv_row = load_latest_csv_row()
    csv_path = get_results_dir() / "latest_run.csv"
    json_summary = load_latest_optimized_summary()

    csv_present = csv_path.exists()
    has_any = csv_present or json_summary is not None
    out: Dict[str, Any] = {
        "status": "ok" if has_any else "empty",
        "results_dir": results_dir,
        "latest_csv": None,
        "latest_json": None,
    }
    if csv_present:
        out["latest_csv"] = {
            "path": str(csv_path.resolve()),
            "mtime_epoch": csv_path.stat().st_mtime,
            "row": csv_row,
        }
    if json_summary:
        out["latest_json"] = json_summary
    if not has_any:
        out["message"] = (
            "No latest_run.csv or optimized_results_*.json found under the configured results paths."
        )
    return out
