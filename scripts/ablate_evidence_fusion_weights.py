#!/usr/bin/env python3
"""Sweep evidence fusion weights on enriched repurposing candidates."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_full_repurposing_pipeline import fuse_and_rank


def _write_temp_config(
    base_path: Path,
    *,
    reversal_multiplier: float,
    zero_unmatched: bool,
    out_dir: Path,
) -> Path:
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    fusion = cfg.setdefault("evidence_fusion", {})
    matched = fusion.setdefault("matched_omics", {})
    matched["signature_reversal_multiplier"] = reversal_multiplier
    matched["zero_unmatched_reversal"] = zero_unmatched
    out = out_dir / f"fusion_config_mult_{reversal_multiplier}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out


def run_ablation(
    enriched_path: Path,
    *,
    multipliers: list[float],
    zero_unmatched: bool,
    config_path: Path,
) -> dict[str, Any]:
    candidates: list[dict] = json.loads(enriched_path.read_text(encoding="utf-8"))
    matched_compounds = {
        c["compound"]
        for c in candidates
        if c.get("creeds_match_status") in ("matched_human", "matched_non_human")
    }
    out_dir = REPO_ROOT / "results" / "repurposing_fusion_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for mult in multipliers:
        temp_cfg = _write_temp_config(
            config_path,
            reversal_multiplier=mult,
            zero_unmatched=zero_unmatched,
            out_dir=out_dir,
        )
        fused = fuse_and_rank(deepcopy(candidates), mode="kg+omics", config_path=temp_cfg.as_posix())
        top3 = [
            {
                "compound": c.compound,
                "final_score": round(c.final_score, 4),
                "signature_reversal_score": round(c.signature_reversal_score, 4),
                "kg_rotate_score": round(c.kg_rotate_score, 4),
            }
            for c in fused[:3]
        ]
        rows.append(
            {
                "signature_reversal_multiplier": mult,
                "zero_unmatched_reversal": zero_unmatched,
                "top3": top3,
                "matched_in_top3": sum(1 for row in top3 if row["compound"] in matched_compounds),
                "top_compound": fused[0].compound if fused else None,
            }
        )

    return {
        "n_candidates": len(candidates),
        "n_creeds_matched": len(matched_compounds),
        "multipliers": multipliers,
        "zero_unmatched_reversal": zero_unmatched,
        "runs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enriched",
        default="results/rnaseq_repurposing_run/repurposing_breast_bundle_human/candidates_enriched.json",
    )
    parser.add_argument(
        "--config",
        default="config/evidence_fusion_config.yaml",
    )
    parser.add_argument(
        "--multipliers",
        default="1.0,2.0,3.0,5.0",
        help="Comma-separated signature_reversal multipliers for matched drugs",
    )
    parser.add_argument(
        "--no-zero-unmatched",
        action="store_true",
        help="Keep unmatched reversal scores in fusion (default: zero them)",
    )
    parser.add_argument(
        "--output",
        default="results/repurposing_fusion_ablation/breast_human.json",
    )
    args = parser.parse_args()

    multipliers = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]
    result = run_ablation(
        Path(args.enriched),
        multipliers=multipliers,
        zero_unmatched=not args.no_zero_unmatched,
        config_path=Path(args.config),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
