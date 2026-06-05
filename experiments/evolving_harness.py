#!/usr/bin/env python3
"""
Evolving experiment harness for Hybrid QML-KG link prediction.

This runner turns the project into a repeatable testing loop:

1. Load a phase from a YAML config.
2. Expand the phase grid into concrete runs.
3. Execute each run command.
4. Record command, config, runtime, return code, stdout/stderr tail.
5. Try to collect metrics from JSON/CSV artifacts.
6. Write runs.jsonl and leaderboard.csv.

The harness is intentionally conservative. It does not mutate source files or delete artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install with: pip install pyyaml") from exc


@dataclass
class RunRecord:
    run_id: str
    phase: str
    status: str
    command: str
    config: Dict[str, Any]
    config_hash: str
    started_at: str
    finished_at: str
    runtime_seconds: float
    returncode: int
    score: Optional[float]
    metrics: Dict[str, Any]
    stdout_tail: str
    stderr_tail: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def render_command(template: str, variables: Dict[str, Any]) -> str:
    return template.format(**variables)


def tail(text: str, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_csv_last_row(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return rows[-1] if rows else None
    except Exception:
        return None


def coerce_number(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def find_latest_metrics(results_dir: Path, started_epoch: float) -> Dict[str, Any]:
    """Best-effort metric collector.

    The project has multiple scripts that may write different artifact names.
    This scans recently modified JSON/CSV files and merges likely metric keys.
    """
    if not results_dir.exists():
        return {}

    candidates: List[Path] = []
    for pattern in ("*.json", "*.csv"):
        candidates.extend(results_dir.rglob(pattern))

    recent = [p for p in candidates if p.stat().st_mtime >= started_epoch - 2]
    recent.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    metric_keys = (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "runtime",
        "seconds",
        "shots",
        "qubits",
        "depth",
    )

    merged: Dict[str, Any] = {}
    for path in recent[:10]:
        payload: Optional[Dict[str, Any]]
        if path.suffix == ".json":
            payload = read_json(path)
        else:
            payload = read_csv_last_row(path)
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            low = key.lower()
            if any(token in low for token in metric_keys):
                merged[key] = coerce_number(value)
        merged.setdefault("metrics_source", str(path))
    return merged


def compute_score(metrics: Dict[str, Any], scoring: Dict[str, Any], runtime_seconds: float) -> Optional[float]:
    primary = scoring.get("primary_metric", "test_pr_auc")
    value = metrics.get(primary)
    if value is None:
        # Try common aliases.
        aliases = [
            primary,
            primary.lower(),
            f"test_{primary}",
            "test_pr_auc",
            "quantum_pr_auc",
            "classical_pr_auc",
            "pr_auc",
        ]
        for alias in aliases:
            if alias in metrics:
                value = metrics[alias]
                break
    try:
        base = float(value)
    except (TypeError, ValueError):
        return None

    runtime_penalty = float(scoring.get("runtime_penalty_per_minute", 0.0)) * (runtime_seconds / 60.0)
    score = base - runtime_penalty

    noise_drop = metrics.get("noise_drop")
    if noise_drop is not None:
        try:
            score -= float(scoring.get("noise_drop_penalty", 0.0)) * float(noise_drop)
        except (TypeError, ValueError):
            pass

    return round(score, 6)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def write_leaderboard(path: Path, records: List[RunRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        row = {
            "run_id": rec.run_id,
            "phase": rec.phase,
            "status": rec.status,
            "score": rec.score,
            "runtime_seconds": round(rec.runtime_seconds, 3),
            "returncode": rec.returncode,
            "config_hash": rec.config_hash,
        }
        for key, value in rec.config.items():
            row[f"config.{key}"] = value
        for key, value in rec.metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"metric.{key}"] = value
        rows.append(row)

    rows.sort(key=lambda r: (-1 if r["score"] is None else -float(r["score"]), r["runtime_seconds"]))
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else ["run_id"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    *,
    phase_name: str,
    command_template: str,
    variables: Dict[str, Any],
    results_dir: Path,
    scoring: Dict[str, Any],
    timeout_seconds: int,
    dry_run: bool,
) -> RunRecord:
    config_hash = stable_hash({"phase": phase_name, **variables})
    run_id = f"{phase_name}-{config_hash}"
    command = render_command(command_template, variables)
    started_at = utc_now()
    started_epoch = time.time()

    if dry_run:
        finished_at = utc_now()
        return RunRecord(
            run_id=run_id,
            phase=phase_name,
            status="dry_run",
            command=command,
            config=variables,
            config_hash=config_hash,
            started_at=started_at,
            finished_at=finished_at,
            runtime_seconds=0.0,
            returncode=0,
            score=None,
            metrics={},
            stdout_tail="",
            stderr_tail="",
        )

    proc = subprocess.run(
        shlex.split(command),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    finished_at = utc_now()
    runtime_seconds = time.time() - started_epoch
    metrics = find_latest_metrics(results_dir, started_epoch)
    score = compute_score(metrics, scoring, runtime_seconds)
    status = "completed" if proc.returncode == 0 else "failed"

    return RunRecord(
        run_id=run_id,
        phase=phase_name,
        status=status,
        command=command,
        config=variables,
        config_hash=config_hash,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=runtime_seconds,
        returncode=proc.returncode,
        score=score,
        metrics=metrics,
        stdout_tail=tail(proc.stdout),
        stderr_tail=tail(proc.stderr),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evolving Hybrid QML-KG experiments.")
    parser.add_argument("--config", default="experiments/configs/evolving_ctd.yaml")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of grid runs for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Render commands without executing.")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    phases = cfg.get("phases", {})
    if args.phase not in phases:
        valid = ", ".join(phases.keys())
        raise SystemExit(f"Unknown phase '{args.phase}'. Valid phases: {valid}")

    project = cfg.get("project", {})
    execution = cfg.get("execution", {})
    scoring = cfg.get("scoring", {})
    phase = phases[args.phase]

    results_dir = Path(project.get("results_dir", "results/evolving"))
    runs_path = results_dir / "runs.jsonl"
    leaderboard_path = results_dir / "leaderboard.csv"

    base_vars = {
        "python_bin": execution.get("python_bin", sys.executable),
        "relation": project.get("relation", "CtD"),
        "results_dir": str(results_dir),
    }

    records: List[RunRecord] = []
    grid = list(expand_grid(phase.get("grid", {})))
    if args.limit is not None:
        grid = grid[: args.limit]

    print(f"Running phase={args.phase} runs={len(grid)} results_dir={results_dir}")

    for idx, combo in enumerate(grid, start=1):
        variables = {**base_vars, **combo}
        print(f"[{idx}/{len(grid)}] {variables}")
        try:
            rec = run_one(
                phase_name=args.phase,
                command_template=phase["command_template"],
                variables=variables,
                results_dir=results_dir,
                scoring=scoring,
                timeout_seconds=int(execution.get("timeout_seconds", 3600)),
                dry_run=bool(args.dry_run or execution.get("dry_run", False)),
            )
        except subprocess.TimeoutExpired as exc:
            finished_at = utc_now()
            runtime_seconds = float(execution.get("timeout_seconds", 3600))
            config_hash = stable_hash({"phase": args.phase, **variables})
            rec = RunRecord(
                run_id=f"{args.phase}-{config_hash}",
                phase=args.phase,
                status="timeout",
                command=render_command(phase["command_template"], variables),
                config=variables,
                config_hash=config_hash,
                started_at=utc_now(),
                finished_at=finished_at,
                runtime_seconds=runtime_seconds,
                returncode=124,
                score=None,
                metrics={},
                stdout_tail=tail(exc.stdout or ""),
                stderr_tail=tail(exc.stderr or ""),
            )

        append_jsonl(runs_path, asdict(rec))
        records.append(rec)
        write_leaderboard(leaderboard_path, records)

        if rec.status == "failed" and execution.get("stop_on_error", False):
            print(f"Stopping after failed run: {rec.run_id}")
            break

    print(f"Wrote {runs_path}")
    print(f"Wrote {leaderboard_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
