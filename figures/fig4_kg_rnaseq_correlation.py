"""Figure: KG prediction score vs RNA-seq reversal score correlation.

Loads the repurposing pipeline artifacts that contain both KG-based prediction
scores and RNA-seq disease-signature reversal scores for the same compound-
disease candidates, then computes Spearman rank correlation and produces a
scatter plot for the paper.

Inputs:
  - artifacts/predictions/top_candidates.json       (12 candidates, multi-modal)
  - artifacts/benchmarks/rnaseq_quantum_tcga_brca_60/ranking_comparison.csv
    (CREEDS ranking, ~60 candidates with signature_reversal_score)

Outputs:
  - figures/fig4_kg_rnaseq_correlation.png          (scatter plot)
  - figures/fig4_kg_rnaseq_correlation.svg
  - results/kg_rnaseq_correlation_metrics.json       (Spearman ρ, p-value, n)

Usage:
    .venv/bin/python figures/fig4_kg_rnaseq_correlation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]

TOP_CANDIDATES_JSON = REPO_ROOT / "artifacts" / "predictions" / "top_candidates.json"
RANKING_COMPARISON_CSV = (
    REPO_ROOT
    / "artifacts"
    / "benchmarks"
    / "rnaseq_quantum_tcga_brca_60"
    / "ranking_comparison.csv"
)
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"


def load_top_candidates() -> list[dict]:
    """Load the multi-modal candidates with KG + RNA-seq scores from CSV (has all fields)."""
    rows: list[dict] = []
    with open(TOP_CANDIDATES_JSON.with_suffix(".csv"), newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["kg_rotate_score"] = float(row["kg_rotate_score"])
            row["qsvc_score"] = float(row["qsvc_score"])
            row["classical_ensemble_score"] = float(row["classical_ensemble_score"])
            row["signature_reversal_score"] = float(row["signature_reversal_score"])
            row["cell_type_reversal_score"] = float(row["cell_type_reversal_score"])
            row["pathway_reversal_score"] = float(row["pathway_reversal_score"])
            row["clinical_evidence_score"] = float(row["clinical_evidence_score"])
            row["final_score"] = float(row["final_score"])
            row["rank"] = int(row["rank"])
            row["confidence_tier"] = int(row["confidence_tier"])
            rows.append(row)
    return rows


def load_creeds_ranking() -> list[dict]:
    """Load the CREEDS ranking comparison with signature reversal scores."""
    rows: list[dict] = []
    with open(RANKING_COMPARISON_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_correlation(
    kg_scores: np.ndarray,
    reversal_scores: np.ndarray,
) -> dict:
    """Spearman rank correlation between KG and RNA-seq reversal scores."""
    rho, pval = spearmanr(kg_scores, reversal_scores)
    return {
        "spearman_rho": float(rho),
        "spearman_p_value": float(pval),
        "n": int(len(kg_scores)),
        "kg_mean": float(np.mean(kg_scores)),
        "reversal_mean": float(np.mean(reversal_scores)),
    }


def plot_scatter(
    candidates: list[dict],
    metrics: dict,
    output_path: Path,
    title_suffix: str,
) -> None:
    """Scatter plot of KG score vs RNA-seq reversal score."""
    kg = np.array([c["kg_rotate_score"] for c in candidates])
    rev = np.array([c["signature_reversal_score"] for c in candidates])
    labels = [c["compound"] for c in candidates]
    diseases = [c["disease"] for c in candidates]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Color points: blue for general, orange for breast cancer
    colors = []
    for d in diseases:
        if "breast" in d.lower():
            colors.append("#E07B00")
        else:
            colors.append("#1f77b4")

    ax.scatter(kg, rev, c=colors, s=80, alpha=0.8, edgecolors="white", linewidth=0.5, zorder=5)

    # Annotate each point
    for i, label in enumerate(labels):
        # Shorten compound names for readability
        short = label.split(" ")[0][:12]
        ax.annotate(
            short,
            (kg[i], rev[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            alpha=0.85,
        )

    # Trend line (linear fit, not Spearman)
    if len(kg) > 2:
        z = np.polyfit(kg, rev, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(kg.min() - 0.02, kg.max() + 0.02, 100)
        ax.plot(x_fit, p(x_fit), "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlabel("KG prediction score (RotatE)", fontsize=11)
    ax.set_ylabel("RNA-seq reversal score", fontsize=11)
    ax.set_title(
        f"KG vs RNA-seq evidence: {title_suffix}\n"
        f"Spearman ρ = {metrics['spearman_rho']:.3f} (p = {metrics['spearman_p_value']:.4f}, n = {metrics['n']})",
        fontsize=10,
    )
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="General candidates"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E07B00", markersize=8, label="Breast cancer"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if str(output_path).endswith(".png"):
        svg_path = output_path.with_suffix(".svg")
        fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Panel A: Top 12 multi-modal candidates ---
    top_candidates = load_top_candidates()
    kg_top = np.array([c["kg_rotate_score"] for c in top_candidates])
    rev_top = np.array([c["signature_reversal_score"] for c in top_candidates])
    metrics_top = compute_correlation(kg_top, rev_top)

    print("=== Panel A: Top multi-modal candidates ===")
    print(f"  n = {metrics_top['n']}")
    print(f"  Spearman ρ = {metrics_top['spearman_rho']:.4f}")
    print(f"  p-value    = {metrics_top['spearman_p_value']:.4f}")
    for c in top_candidates:
        print(f"    {c['compound']:25s} -> {c['disease']:30s}  KG={c['kg_rotate_score']:.3f}  rev={c['signature_reversal_score']:.3f}")

    fig_a = FIGURES_DIR / "fig4_kg_rnaseq_correlation.png"
    plot_scatter(top_candidates, metrics_top, fig_a, "top-12 multi-modal candidates")
    print(f"  Saved: {fig_a}")

    # --- Panel B: CREEDS ranking (breast cancer DE signature) ---
    creeds_rows = load_creeds_ranking()
    # Deduplicate by compound name (keep highest reversal score per compound)
    seen: dict[str, dict] = {}
    for row in creeds_rows:
        compound = row.get("compound", "").strip()
        if not compound:
            continue
        rev_score = float(row.get("signature_reversal_score", 0))
        kg_score = float(row.get("kg_rotate_score", 0))
        if compound not in seen or rev_score > seen[compound]["signature_reversal_score"]:
            seen[compound] = {
                "compound": compound,
                "disease": "breast cancer (CREEDS)",
                "kg_rotate_score": kg_score,
                "signature_reversal_score": rev_score,
            }
    creeds_candidates = list(seen.values())
    kg_creeds = np.array([c["kg_rotate_score"] for c in creeds_candidates])
    rev_creeds = np.array([c["signature_reversal_score"] for c in creeds_candidates])
    metrics_creeds = compute_correlation(kg_creeds, rev_creeds)

    print(f"\n=== Panel B: CREEDS ranking (deduplicated, breast cancer) ===")
    print(f"  n = {metrics_creeds['n']}")
    print(f"  Spearman ρ = {metrics_creeds['spearman_rho']:.4f}")
    print(f"  p-value    = {metrics_creeds['spearman_p_value']:.4f}")
    for c in sorted(creeds_candidates, key=lambda x: x["signature_reversal_score"], reverse=True)[:10]:
        print(f"    {c['compound']:25s}  KG={c['kg_rotate_score']:.3f}  rev={c['signature_reversal_score']:.3f}")

    fig_b = FIGURES_DIR / "fig4b_creeds_kg_rnaseq_correlation.png"
    plot_scatter(creeds_candidates, metrics_creeds, fig_b, "CREEDS breast cancer ranking")
    print(f"  Saved: {fig_b}")

    # --- Save metrics ---
    results = {
        "panel_a_top_candidates": metrics_top,
        "panel_b_creeds_ranking": metrics_creeds,
        "description": (
            "Spearman rank correlation between KG-based prediction score (RotatE) "
            "and RNA-seq disease-signature reversal score. A positive ρ indicates "
            "that KG predictions align with transcriptomic reversal evidence."
        ),
    }
    metrics_path = RESULTS_DIR / "kg_rnaseq_correlation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
