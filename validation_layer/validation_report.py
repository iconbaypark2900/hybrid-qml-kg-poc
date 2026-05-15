from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _derive_status(pair: Dict) -> str:
    known = pair.get("known_indication", False)
    trial = pair.get("clinical_trial_found", False)
    lit = pair.get("literature_support_count", 0)

    if known:
        return "known"
    if trial:
        return "supported_novel_candidate"
    if lit >= 5:
        return "literature_supported"
    return "exploratory"


def write_validation_report(
    pairs: List[Dict],
    out_dir: str = "artifacts/validation",
    top_n: int = 50,
) -> Path:
    """
    Write validation status for top-N candidate pairs.

    Adds 'validation_status' field derived from known/trial/literature signals.
    Writes JSON + MD + CSV.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        pair.setdefault("validation_status", _derive_status(pair))

    top = pairs[:top_n]

    # JSON
    json_path = out / "validation_report.json"
    json_path.write_text(json.dumps(top, indent=2), encoding="utf-8")

    # CSV
    csv_path = out / "validation_report.csv"
    if top:
        fieldnames = list(top[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(top)

    # Markdown
    md_path = out / "validation_report.md"
    lines = [
        "# Validation Report",
        "",
        f"Candidates validated: {len(pairs)} | Shown: {len(top)}",
        "",
        "## Status Summary",
        "",
        "| Status | Count |",
        "|--------|-------|",
    ]
    from collections import Counter
    counts = Counter(p.get("validation_status", "unknown") for p in pairs)
    for status, n in sorted(counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {status} | {n} |")

    lines += [
        "",
        "## Top Candidates",
        "",
        "| Rank | Compound | Disease | Score | Status | Known | Trial | Literature |",
        "|------|----------|---------|-------|--------|-------|-------|-----------|",
    ]
    for i, p in enumerate(top, 1):
        lines.append(
            f"| {i} | {p.get('compound','')} | {p.get('disease','')} "
            f"| {p.get('final_score', 0.0):.3f} | {p.get('validation_status','')} "
            f"| {p.get('known_indication', False)} "
            f"| {p.get('trial_phase', '—')} "
            f"| {p.get('literature_support_count', 0)} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"Validation report written to {md_path}")
    return md_path
