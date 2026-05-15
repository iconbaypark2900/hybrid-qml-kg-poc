from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

from evidence_layer.evidence_schema import EvidenceFeatures

logger = logging.getLogger(__name__)


def write_evidence_report(
    candidates: List[EvidenceFeatures],
    out_dir: str = "artifacts/predictions",
    top_n: int = 50,
    disease_id: Optional[str] = None,
) -> Path:
    """
    Write top-N candidates to CSV and a markdown summary report.

    Returns path to the CSV file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    top = candidates[:top_n]

    # CSV
    csv_path = out / "top_candidates.csv"
    fieldnames = [
        "rank", "compound", "disease", "final_score", "confidence_tier",
        "kg_rotate_score", "qsvc_score", "classical_ensemble_score",
        "signature_reversal_score", "cell_type_reversal_score",
        "pathway_reversal_score", "clinical_evidence_score",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i, ef in enumerate(top, 1):
            row = ef.to_dict()
            row["rank"] = i
            writer.writerow(row)

    # JSON (full feature vectors)
    json_path = out / "top_candidates.json"
    json_path.write_text(
        json.dumps([ef.to_dict() for ef in top], indent=2), encoding="utf-8"
    )

    # Markdown
    md_path = out / "final_repurposing_report.md"
    lines = [
        "# Drug Repurposing Report",
        "",
        f"**Disease:** {disease_id or 'all'}  ",
        f"**Candidates ranked:** {len(candidates)}  ",
        f"**Top-N shown:** {top_n}",
        "",
        "| Rank | Compound | Score | Tier | KG | QSVC | Ensemble | Reversal |",
        "|------|----------|-------|------|-----|------|----------|---------|",
    ]
    for i, ef in enumerate(top, 1):
        lines.append(
            f"| {i} | {ef.compound} | {ef.final_score:.3f} | {ef.confidence_tier} "
            f"| {ef.kg_rotate_score:.3f} | {ef.qsvc_score:.3f} "
            f"| {ef.classical_ensemble_score:.3f} | {ef.signature_reversal_score:.3f} |"
        )

    if top and top[0].explanation:
        lines += ["", "## Evidence Detail — Top Candidate", "", "```"]
        lines.append(top[0].explanation)
        lines.append("```")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Evidence report written to {csv_path} and {md_path}")
    return csv_path
