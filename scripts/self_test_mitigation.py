#!/usr/bin/env python3
"""
Quick self-test to confirm mitigation logging columns are produced.

Runs a small noisy-simulator QSVC benchmark and validates that:
- ZNE columns exist
- Readout mitigation columns exist (when enabled and qubits <= max_qubits)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
LATEST = RESULTS_DIR / "latest_run.csv"


def run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(proc.returncode)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "scripts/run_optimized_pipeline.py",
        "--relation",
        "CtD",
        "--results_dir",
        "results",
        "--quantum_config_path",
        "config/quantum_config_noisy.yaml",
        "--fast_mode",
        "--quantum_only",
        "--max_entities",
        "80",
        "--qml_dim",
        "6",
    ]
    run(cmd)

    if not LATEST.exists():
        raise SystemExit(f"Missing {LATEST}")

    df = pd.read_csv(LATEST)
    if df.empty:
        raise SystemExit(f"{LATEST} is empty")

    required = [
        "obs_zne_enabled",
        # Analytical (Pauli-path) — historical primary column.
        "obs_zne_kernel_posneg_mean_C0",
        # Per preregistration §5.7 step 6 + §8.6: separate linear and Richardson
        # extrapolations must be reported alongside the analytical fit. These
        # columns were added by the ZNE Gap 3-5 fix; absence indicates a
        # regression in quantum_layer/qml_trainer.py's observable schema.
        "obs_zne_kernel_posneg_mean_C0_analytical",
        "obs_zne_kernel_posneg_mean_C0_linear",
        "obs_zne_kernel_posneg_mean_C0_richardson",
        # Backend metadata (Gap 7) — distinguishes simulator runs from hardware
        # snapshots for §8.6 sensitivity grouping.
        "obs_zne_backend_name",
        "obs_zne_backend_snapshot_id",
        # Readout mitigation + raw lambda=1.0 baseline.
        "obs_readout_mitigation_enabled",
        "obs_kernel_posneg_mean_explicit_raw_lambda1",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in latest_run.csv: {missing}")

    print("✅ Mitigation self-test passed.")
    print("Columns present:", ", ".join(required))


if __name__ == "__main__":
    main()

