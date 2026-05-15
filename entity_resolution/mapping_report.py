from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def write_mapping_report(
    results: Dict[str, Optional[str]],
    entity_type: str,
    out_dir: str = "artifacts/entity_resolution",
    filename: Optional[str] = None,
) -> Path:
    """
    Write a mapping quality report for a batch resolution result.

    Args:
        results: {query → hetionet_id_or_None} from any *Mapper.map_many()
        entity_type: "gene", "disease", or "compound" (used in filename/header)
        out_dir: directory to write report into
        filename: override default filename

    Returns:
        Path to the written markdown report.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fname = filename or f"{entity_type}_mapping_report.md"
    report_path = out / fname
    json_path = out / fname.replace(".md", ".json")

    resolved = {k: v for k, v in results.items() if v is not None}
    unresolved = [k for k, v in results.items() if v is None]
    total = len(results)
    n_resolved = len(resolved)
    resolution_rate = n_resolved / total if total else 0.0

    # JSON dump for downstream use
    json_path.write_text(
        json.dumps(
            {
                "entity_type": entity_type,
                "total": total,
                "resolved": n_resolved,
                "unresolved_count": len(unresolved),
                "resolution_rate": round(resolution_rate, 4),
                "mapping": results,
                "unresolved": unresolved,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Markdown report
    lines: List[str] = [
        f"# Entity Resolution Report — {entity_type.capitalize()}",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total queries | {total} |",
        f"| Resolved | {n_resolved} |",
        f"| Unresolved | {len(unresolved)} |",
        f"| Resolution rate | {resolution_rate:.1%} |",
        "",
        "## Resolved Mappings",
        "",
        "| Query | Hetionet ID |",
        "|-------|-------------|",
    ]
    for q, nid in sorted(resolved.items()):
        lines.append(f"| {q} | {nid} |")

    if unresolved:
        lines += [
            "",
            "## Unresolved Queries",
            "",
            "These identifiers could not be mapped to a Hetionet node.",
            "Consider adding them to `entity_resolution_config.yaml` aliases.",
            "",
        ]
        for q in sorted(unresolved):
            lines.append(f"- `{q}`")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(
        f"Mapping report written to {report_path} "
        f"({n_resolved}/{total} resolved, {resolution_rate:.1%})"
    )
    return report_path
