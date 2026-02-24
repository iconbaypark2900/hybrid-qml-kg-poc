#!/usr/bin/env python3
"""
Quantum ablation study script.

Runs the pipeline with different quantum configurations to study:
- Feature map repetitions (reps=1 vs reps=2)
- Entanglement strategies (linear vs full)
- QML dimensions (8, 12, 16, 24 qubits)

Results are written to a CSV for analysis and can be loaded into the dashboard.
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def run_config(
    relation: str,
    qml_dim: int,
    feature_map_reps: int,
    entanglement: str,
    encoding: str = "hybrid",
    fast_mode: bool = True,
    random_state: int = 42,
    extra_args: List[str] = None,
) -> Dict[str, Any]:
    """Run the pipeline with a specific configuration and return metrics."""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "run_optimized_pipeline.py"),
        "--relation", relation,
        "--results_dir", str(RESULTS_DIR),
        "--qml_dim", str(qml_dim),
        "--qml_feature_map_reps", str(feature_map_reps),
        "--qml_entanglement", entanglement,
        "--qml_encoding", encoding,
        "--random_state", str(random_state),
    ]
    if fast_mode:
        cmd.append("--fast_mode")
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Running: dim={qml_dim}, reps={feature_map_reps}, ent={entanglement}, enc={encoding}")
    print(f"{'='*60}")
    
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    
    result = {
        "qml_dim": qml_dim,
        "feature_map_reps": feature_map_reps,
        "entanglement": entanglement,
        "encoding": encoding,
        "random_state": random_state,
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
    }
    
    # Read latest_run.csv to get metrics
    latest_run = RESULTS_DIR / "latest_run.csv"
    if latest_run.exists():
        try:
            df = pd.read_csv(latest_run)
            if not df.empty:
                row = df.iloc[-1].to_dict()
                result.update({
                    "classical_pr_auc": row.get("classical_pr_auc"),
                    "quantum_pr_auc": row.get("quantum_pr_auc"),
                    "classical_accuracy": row.get("classical_accuracy"),
                    "quantum_accuracy": row.get("quantum_accuracy"),
                    "execution_mode": row.get("execution_mode"),
                    "noise_model": row.get("noise_model"),
                })
        except Exception as e:
            result["read_error"] = str(e)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Quantum ablation study")
    parser.add_argument("--relation", type=str, default="CtD", help="Target relation")
    parser.add_argument("--dims", type=str, default="8,12,16",
                        help="Comma-separated QML dimensions to test")
    parser.add_argument("--reps", type=str, default="1,2",
                        help="Comma-separated feature map reps to test")
    parser.add_argument("--entanglements", type=str, default="linear,full",
                        help="Comma-separated entanglement strategies")
    parser.add_argument("--encodings", type=str, default="hybrid",
                        help="Comma-separated encoding strategies")
    parser.add_argument("--fast_mode", action="store_true", default=True,
                        help="Use fast mode for quicker runs")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--output", type=str, default="",
                        help="Output CSV path")
    args = parser.parse_args()
    
    dims = [int(x.strip()) for x in args.dims.split(",")]
    reps = [int(x.strip()) for x in args.reps.split(",")]
    entanglements = [x.strip() for x in args.entanglements.split(",")]
    encodings = [x.strip() for x in args.encodings.split(",")]
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    total = len(dims) * len(reps) * len(entanglements) * len(encodings)
    run_num = 0
    
    for dim in dims:
        for rep in reps:
            for ent in entanglements:
                for enc in encodings:
                    run_num += 1
                    print(f"\n[{run_num}/{total}]")
                    result = run_config(
                        relation=args.relation,
                        qml_dim=dim,
                        feature_map_reps=rep,
                        entanglement=ent,
                        encoding=enc,
                        fast_mode=args.fast_mode,
                        random_state=args.random_state,
                    )
                    results.append(result)
    
    # Write results
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_path = Path(args.output) if args.output else RESULTS_DIR / f"quantum_ablation_{ts}.csv"
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n{'='*60}")
        print(f"Wrote {len(results)} results to {output_path}")
        print(f"{'='*60}")
        
        # Summary
        print("\n=== Summary ===")
        print(df[["qml_dim", "feature_map_reps", "entanglement", "encoding", 
                  "classical_pr_auc", "quantum_pr_auc", "status"]].to_string())
    else:
        print("No results produced.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
