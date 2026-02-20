# benchmarking/report_generator.py

"""
Generate hypothesis-specific reports (mediation, ablation) from MetricsTracker outputs.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_mediation_report(
    hypothesis_id: str,
    results: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """
    Populate mediation template from MetricsTracker outputs and model predictions.

    Args:
        hypothesis_id: e.g., H-002
        results: Dict with keys: direct_effect, indirect_effect, total_effect,
                 mediation_proportion, lysosomal_features_included, stability_with,
                 stability_without, directional_with, directional_without
        output_path: Path to write report (default: results/reports/mediation_{id}.md)

    Returns:
        Rendered report string
    """
    template = _read_mediation_template()
    timestamp = datetime.now().isoformat()
    ctx = {
        "hypothesis_id": hypothesis_id,
        "timestamp": timestamp,
        "direct_effect": results.get("direct_effect"),
        "indirect_effect": results.get("indirect_effect"),
        "total_effect": results.get("total_effect"),
        "mediation_proportion": results.get("mediation_proportion"),
        "lysosomal_features_included": results.get("lysosomal_features_included", True),
        "stability_with": results.get("stability_with"),
        "stability_without": results.get("stability_without"),
        "directional_with": results.get("directional_with"),
        "directional_without": results.get("directional_without"),
    }
    report = _render_template(template, ctx)
    if output_path is None:
        output_path = f"results/reports/mediation_{hypothesis_id}.md"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    logger.info(f"Wrote mediation report to {output_path}")
    return report


def _read_mediation_template() -> str:
    """Read mediation template from results/reports/."""
    path = Path(__file__).parent.parent / "results" / "reports" / "mediation_template.md"
    if path.exists():
        return path.read_text()
    return """# Mediation Analysis Report: {{ hypothesis_id }}
**Generated:** {{ timestamp }}
| Metric | Value |
|--------|-------|
| Hypothesis ID | {{ hypothesis_id }} |
| Direct Effect | {{ direct_effect | default("—") }} |
| Indirect Effect | {{ indirect_effect | default("—") }} |
| Total Effect | {{ total_effect | default("—") }} |
| Mediation Proportion | {{ mediation_proportion | default("—") }} |
"""


def _render_template(template: str, ctx: Dict[str, Any]) -> str:
    """Simple template rendering (no Jinja2 dependency)."""
    import re
    out = template
    for key, val in ctx.items():
        display = str(val) if val is not None else "—"
        # Match {{ key }} or {{ key | default("...") }}
        pat = r"\{\{\s*" + re.escape(key) + r"(\s*\|\s*default\([^)]*\))?\s*\}\}"
        out = re.sub(pat, display, out)
    return out


def generate_ablation_report(
    baseline_metrics: Dict[str, float],
    mechanism_metrics: Dict[str, float],
    output_path: str = "results/reports/ablation_comparison.md",
) -> str:
    """
    Compare baseline CtD model vs mechanism-informed ranking.

    Args:
        baseline_metrics: e.g., {"pr_auc": 0.78, "ndcg": 0.5}
        mechanism_metrics: Same keys for mechanism-informed model
        output_path: Where to write report

    Returns:
        Rendered report string
    """
    report = [
        "# Ablation Comparison: Baseline vs Mechanism-Informed Ranking",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        "",
        "## Baseline CtD Model",
        "",
    ]
    for k, v in baseline_metrics.items():
        report.append(f"- {k}: {v}")
    report.extend([
        "",
        "## Mechanism-Informed Ranking",
        "",
    ])
    for k, v in mechanism_metrics.items():
        report.append(f"- {k}: {v}")
    report.extend([
        "",
        "## Comparison",
        "",
        "| Metric | Baseline | Mechanism | Difference |",
        "|--------|----------|-----------|------------|",
    ])
    for k in set(baseline_metrics) | set(mechanism_metrics):
        b = baseline_metrics.get(k, 0)
        m = mechanism_metrics.get(k, 0)
        diff = m - b
        report.append(f"| {k} | {b:.4f} | {m:.4f} | {diff:+.4f} |")
    report.append("")
    text = "\n".join(report)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)
    logger.info(f"Wrote ablation report to {output_path}")
    return text
