#!/usr/bin/env python3
"""
compare_pipeline_modes.py — KG-only vs KG+omics delta benchmarking.

Runs `run_full_repurposing_pipeline.py` in both modes (or consumes existing
outputs), then emits a per-disease comparison table showing how adding omics
evidence shifts the ranking.

Outputs:
    artifacts/predictions/mode_comparison.csv
    artifacts/predictions/mode_comparison.md

Usage:
    python scripts/compare_pipeline_modes.py
    python scripts/compare_pipeline_modes.py --kg-only-dir runs/kg_only --kg-omics-dir runs/kg_omics
    python scripts/compare_pipeline_modes.py --no-run    # use existing artifacts only
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("compare_pipeline_modes")

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _run_pipeline(mode: str, out_dir: Path, top_n: int) -> int:
    """Invoke the orchestrator as a subprocess. Returns exit code."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_full_repurposing_pipeline.py",
        "--mode", mode, "--top-n", str(top_n),
        "--output", str(out_dir),
    ]
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Pipeline failed ({mode}):\n{result.stderr}")
    return result.returncode


def _load_candidates(out_dir: Path) -> List[Dict]:
    """Load top_candidates.json — preferred over CSV because it has every field."""
    p = out_dir / "top_candidates.json"
    if not p.exists():
        raise FileNotFoundError(f"Expected pipeline output at {p}; run pipeline first.")
    return json.loads(p.read_text(encoding="utf-8"))


def _index_by_pair(candidates: List[Dict]) -> Dict[Tuple[str, str], Dict]:
    """Index candidates by (compound, disease) so we can join the two modes."""
    out: Dict[Tuple[str, str], Dict] = {}
    for c in candidates:
        key = (c.get("compound", ""), c.get("disease", ""))
        out[key] = c
    return out


def build_comparison(
    kg_only_candidates: List[Dict],
    kg_omics_candidates: List[Dict],
) -> List[Dict]:
    """
    Join kg-only and kg+omics outputs by (compound, disease) and compute deltas.
    Returns rows sorted by absolute delta descending.
    """
    kg_idx = _index_by_pair(kg_only_candidates)
    omics_idx = _index_by_pair(kg_omics_candidates)

    keys = sorted(set(kg_idx) | set(omics_idx))
    rows: List[Dict] = []
    for key in keys:
        kg = kg_idx.get(key, {})
        full = omics_idx.get(key, {})
        compound, disease = key
        kg_score = float(kg.get("final_score", 0.0))
        omics_score = float(full.get("final_score", 0.0))
        kg_tier = int(kg.get("confidence_tier", 4))
        omics_tier = int(full.get("confidence_tier", 4))
        rows.append({
            "compound": compound,
            "disease": disease,
            "kg_only_score": round(kg_score, 4),
            "kg_omics_score": round(omics_score, 4),
            "delta_score": round(omics_score - kg_score, 4),
            "kg_only_tier": kg_tier,
            "kg_omics_tier": omics_tier,
            "tier_promoted": int(omics_tier < kg_tier),  # lower number = better tier
            "reversal_score": round(float(full.get("signature_reversal_score", 0.0)), 4),
        })

    rows.sort(key=lambda r: abs(r["delta_score"]), reverse=True)
    return rows


def _write_csv(rows: List[Dict], path: Path) -> None:
    import csv
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: List[Dict], path: Path,
                    kg_summary: Dict, omics_summary: Dict) -> None:
    """Emit a markdown report comparing the two modes."""
    n = len(rows)
    promoted = sum(1 for r in rows if r["tier_promoted"])
    score_delta_mean = sum(r["delta_score"] for r in rows) / n if n else 0.0
    score_delta_max = max((r["delta_score"] for r in rows), default=0.0)
    score_delta_min = min((r["delta_score"] for r in rows), default=0.0)

    lines = [
        "# Pipeline Mode Comparison — KG-only vs KG+omics",
        "",
        "## Summary",
        "",
        f"| Metric | KG-only | KG+omics |",
        f"|--------|---------|----------|",
        f"| Top compound | {kg_summary.get('top_compound', '—')} | {omics_summary.get('top_compound', '—')} |",
        f"| Top score | {kg_summary.get('top_score', 0.0):.4f} | {omics_summary.get('top_score', 0.0):.4f} |",
        f"| Tier 1 count | {kg_summary.get('tier_distribution', {}).get('tier_1', 0)} | "
        f"{omics_summary.get('tier_distribution', {}).get('tier_1', 0)} |",
        f"| Tier 2 count | {kg_summary.get('tier_distribution', {}).get('tier_2', 0)} | "
        f"{omics_summary.get('tier_distribution', {}).get('tier_2', 0)} |",
        f"| Tier 3 count | {kg_summary.get('tier_distribution', {}).get('tier_3', 0)} | "
        f"{omics_summary.get('tier_distribution', {}).get('tier_3', 0)} |",
        f"| Tier 4 count | {kg_summary.get('tier_distribution', {}).get('tier_4', 0)} | "
        f"{omics_summary.get('tier_distribution', {}).get('tier_4', 0)} |",
        "",
        f"**Pairs compared:** {n}",
        f"**Tier-promoted by omics:** {promoted}",
        f"**Mean Δscore:** {score_delta_mean:+.4f}",
        f"**Max Δscore:** {score_delta_max:+.4f}",
        f"**Min Δscore:** {score_delta_min:+.4f}",
        "",
        "## Per-candidate delta",
        "",
        "| Compound | Disease | KG-only | KG+omics | Δ | KG tier → omics tier | Reversal |",
        "|----------|---------|---------|----------|----|----------------------|---------|",
    ]
    for r in rows:
        promo = "↑" if r["tier_promoted"] else ""
        lines.append(
            f"| {r['compound']} | {r['disease']} | "
            f"{r['kg_only_score']:.4f} | {r['kg_omics_score']:.4f} | "
            f"{r['delta_score']:+.4f} | T{r['kg_only_tier']} → T{r['kg_omics_tier']} {promo} | "
            f"{r['reversal_score']:.3f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kg-only-dir", default="artifacts/predictions/kg_only",
                        help="Output directory for kg-only run")
    parser.add_argument("--kg-omics-dir", default="artifacts/predictions/kg_omics",
                        help="Output directory for kg+omics run")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top-N candidates to compare (default 20)")
    parser.add_argument("--output", default="artifacts/predictions",
                        help="Where to write mode_comparison.{csv,md}")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip pipeline execution; use existing artifacts only")
    args = parser.parse_args()

    kg_dir = Path(args.kg_only_dir)
    omics_dir = Path(args.kg_omics_dir)

    if not args.no_run:
        if _run_pipeline("kg-only", kg_dir, args.top_n) != 0:
            return 1
        if _run_pipeline("kg+omics", omics_dir, args.top_n) != 0:
            return 1

    kg_candidates = _load_candidates(kg_dir)
    omics_candidates = _load_candidates(omics_dir)
    kg_summary = json.loads((kg_dir / "run_summary.json").read_text())
    omics_summary = json.loads((omics_dir / "run_summary.json").read_text())

    rows = build_comparison(kg_candidates, omics_candidates)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "mode_comparison.csv"
    md_path = out_dir / "mode_comparison.md"
    _write_csv(rows, csv_path)
    _write_markdown(rows, md_path, kg_summary, omics_summary)

    logger.info(f"=== Comparison complete: {len(rows)} pairs ===")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"MD:  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
