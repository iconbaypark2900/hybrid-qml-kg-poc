#!/usr/bin/env python3
"""Export ranking_comparison.csv from repurposing pipeline outputs.

Bridges run_full_repurposing_pipeline.py (candidates_enriched.json) to
build_repurposing_evidence_bundle.py (ranking_comparison.csv schema).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evidence_layer.evidence_schema import EvidenceFeatures
from evidence_layer.feature_fusion import fuse_evidence


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_") or "unknown"


def candidate_id_for(row: dict[str, Any]) -> str:
    compound_id = row.get("compound_hetionet_id")
    disease_id = row.get("disease_hetionet_id")
    if compound_id and disease_id:
        return f"{compound_id}::{disease_id}"
    return f"{_slug(str(row.get('compound', '')))}::{disease_id or _slug(str(row.get('disease', '')))}"


def _to_evidence_features(row: dict[str, Any]) -> EvidenceFeatures:
    return EvidenceFeatures(
        compound=str(row.get("compound", "")),
        compound_hetionet_id=row.get("compound_hetionet_id"),
        disease=str(row.get("disease", "")),
        disease_hetionet_id=row.get("disease_hetionet_id"),
        kg_rotate_score=float(row.get("kg_rotate_score", 0.0)),
        kg_complex_score=float(row.get("kg_complex_score", 0.0)),
        graph_topology_score=float(row.get("graph_topology_score", 0.0)),
        qsvc_score=float(row.get("qsvc_score", 0.0)),
        classical_ensemble_score=float(row.get("classical_ensemble_score", 0.0)),
        signature_reversal_score=float(row.get("signature_reversal_score", 0.0)),
        cell_type_reversal_score=float(row.get("cell_type_reversal_score", 0.0)),
        pathway_reversal_score=float(row.get("pathway_reversal_score", 0.0)),
        clinical_evidence_score=float(row.get("clinical_evidence_score", 0.0)),
    )


def build_ranking_comparison_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fuse kg-only, kg+omics, and kg+omics+quantum scores per candidate."""
    fused_inputs = [_to_evidence_features(row) for row in candidates]
    match_statuses = [str(row.get("creeds_match_status", "unmatched")) for row in candidates]

    kg_only = [
        EvidenceFeatures(**{**ef.to_dict(), "qsvc_score": 0.0, "classical_ensemble_score": 0.0})
        for ef in fused_inputs
    ]
    kg_omics = [
        EvidenceFeatures(**{**ef.to_dict(), "qsvc_score": 0.0})
        for ef in fused_inputs
    ]
    kg_omics_quantum = list(fused_inputs)

    kg_only_fused = fuse_evidence(kg_only, mode="kg-only")
    kg_omics_fused = fuse_evidence(
        kg_omics, mode="kg+omics", omics_match_status=match_statuses
    )
    kg_omics_quantum_fused = fuse_evidence(
        kg_omics_quantum, mode="kg+omics", omics_match_status=match_statuses
    )

    score_by_compound = {
        "kg_only": {x.compound: x.final_score for x in kg_only_fused},
        "kg_omics": {x.compound: x.final_score for x in kg_omics_fused},
        "kg_omics_quantum": {x.compound: x.final_score for x in kg_omics_quantum_fused},
    }

    rows: list[dict[str, Any]] = []
    for row in candidates:
        compound = str(row.get("compound", ""))
        out: dict[str, Any] = {
            "candidate_id": candidate_id_for(row),
            "compound": compound,
            "profile_gene_overlap": int(row.get("creeds_profile_count") or 0),
            "signature_reversal_score": float(row.get("signature_reversal_score", 0.0)),
            "kg_only_final_score": score_by_compound["kg_only"][compound],
            "kg_omics_final_score": score_by_compound["kg_omics"][compound],
            "kg_omics_quantum_final_score": score_by_compound["kg_omics_quantum"][compound],
            "kg_rotate_score": float(row.get("kg_rotate_score", 0.0)),
            "qsvc_score": float(row.get("qsvc_score", 0.0)),
            "classical_ensemble_score": float(row.get("classical_ensemble_score", 0.0)),
            "quantum_status": "ok" if float(row.get("qsvc_score", 0.0)) != 0.0 else "skipped",
        }
        out["quantum_delta_score"] = (
            out["kg_omics_quantum_final_score"] - out["kg_omics_final_score"]
        )
        for optional in (
            "creeds_id",
            "creeds_match_status",
            "creeds_organism",
            "creeds_profile_organism",
            "creeds_geo_id",
            "creeds_cell_type",
            "disease_hetionet_id",
        ):
            if optional in row and row[optional] is not None:
                out[optional] = row[optional]
        if row.get("creeds_cell_type"):
            out["cell_type"] = row["creeds_cell_type"]
        if row.get("creeds_geo_id"):
            out["geo_id"] = row["creeds_geo_id"]
        rows.append(out)

    return rows


def load_candidates(input_dir: Path) -> list[dict[str, Any]]:
    enriched = input_dir / "candidates_enriched.json"
    if enriched.exists():
        return json.loads(enriched.read_text(encoding="utf-8"))
    top_json = input_dir / "top_candidates.json"
    if top_json.exists():
        return json.loads(top_json.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"No candidates_enriched.json or top_candidates.json in {input_dir}"
    )


def export_ranking_comparison(
    input_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    input_dir = Path(input_dir)
    candidates = load_candidates(input_dir)
    rows = build_ranking_comparison_rows(candidates)
    ranking_df = pd.DataFrame(rows).sort_values(
        "kg_omics_quantum_final_score", ascending=False
    )
    out = Path(output_path) if output_path else input_dir / "ranking_comparison.csv"
    ranking_df.to_csv(out, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Repurposing output directory containing candidates_enriched.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: {input}/ranking_comparison.csv)",
    )
    args = parser.parse_args()
    out = export_ranking_comparison(args.input, args.output)
    print(json.dumps({"status": "ok", "output": str(out), "n_rows": len(pd.read_csv(out))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
