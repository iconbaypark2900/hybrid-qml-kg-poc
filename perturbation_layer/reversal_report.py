from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def write_reversal_report(
    reversal_scores: Dict[str, float],
    disease_id: str,
    cell_type_scores: Optional[Dict[str, Dict[str, float]]] = None,
    out_dir: str = "artifacts/perturbations",
) -> Path:
    """
    Write reversal scores to CSV and a summary markdown report.

    Args:
        reversal_scores: {compound: overall_reversal_score}
        disease_id: Hetionet Disease node ID (for labelling)
        cell_type_scores: {compound: {cell_type: score}} (optional)
        out_dir: output directory

    Returns:
        Path to CSV file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / "reversal_scores.csv"
    sorted_scores = sorted(reversal_scores.items(), key=lambda x: x[1], reverse=True)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["compound", "disease", "overall_reversal_score"]
        if cell_type_scores:
            all_ct = sorted({ct for v in cell_type_scores.values() for ct in v})
            fieldnames += [f"reversal_{ct}" for ct in all_ct]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for compound, score in sorted_scores:
            row: Dict = {"compound": compound, "disease": disease_id,
                         "overall_reversal_score": round(score, 4)}
            if cell_type_scores and compound in cell_type_scores:
                for ct in all_ct:
                    row[f"reversal_{ct}"] = round(
                        cell_type_scores[compound].get(ct, 0.0), 4
                    )
            writer.writerow(row)

    logger.info(f"Reversal scores written to {csv_path} ({len(sorted_scores)} compounds)")

    # Summary markdown
    md_path = out / "reversal_report.md"
    lines = [
        f"# Reversal Score Report",
        f"",
        f"**Disease:** `{disease_id}`  ",
        f"**Compounds scored:** {len(sorted_scores)}",
        "",
        "## Top-20 Candidates by Reversal Score",
        "",
        "| Rank | Compound | Reversal Score |",
        "|------|----------|---------------|",
    ]
    for i, (compound, score) in enumerate(sorted_scores[:20], 1):
        lines.append(f"| {i} | {compound} | {score:.4f} |")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path
