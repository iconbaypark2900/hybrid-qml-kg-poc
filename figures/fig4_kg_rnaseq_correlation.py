"""Figure: KG prediction score vs RNA-seq reversal score correlation.

Loads repurposing pipeline artifacts (breast cosine CREEDS + full 200-pair run),
computes Spearman rank correlation on CREEDS-matched rows only, and writes metrics.

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

BREAST_CSV = (
    REPO_ROOT
    / "results"
    / "rnaseq_repurposing_run"
    / "repurposing_breast_bundle_human"
    / "top_candidates.csv"
)
BREAST_ENRICHED = (
    REPO_ROOT
    / "results"
    / "rnaseq_repurposing_run"
    / "repurposing_breast_bundle_human"
    / "candidates_enriched.json"
)
FULL_200_CSV = (
    REPO_ROOT
    / "results"
    / "rnaseq_repurposing_run"
    / "repurposing_full_200_cosine"
    / "top_candidates.csv"
)
FIGURES_DIR = REPO_ROOT / "figures"
RESULTS_DIR = REPO_ROOT / "results"


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["kg_rotate_score"] = float(row.get("kg_rotate_score") or 0)
            row["signature_reversal_score"] = float(row.get("signature_reversal_score") or 0)
            rows.append(row)
    return rows


def load_enriched_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    for row in data:
        row["kg_rotate_score"] = float(row.get("kg_rotate_score") or 0)
        row["signature_reversal_score"] = float(row.get("signature_reversal_score") or 0)
    return data


def matched_only(rows: list[dict]) -> list[dict]:
    return [row for row in rows if float(row.get("signature_reversal_score") or 0) > 0]


def compute_correlation(kg_scores: np.ndarray, reversal_scores: np.ndarray) -> dict:
    if len(kg_scores) < 2:
        return {
            "spearman_rho": float("nan"),
            "spearman_p_value": float("nan"),
            "n": int(len(kg_scores)),
            "kg_mean": float(np.mean(kg_scores)) if len(kg_scores) else float("nan"),
            "reversal_mean": float(np.mean(reversal_scores)) if len(reversal_scores) else float("nan"),
        }
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
    kg = np.array([c["kg_rotate_score"] for c in candidates])
    rev = np.array([c["signature_reversal_score"] for c in candidates])
    labels = [str(c.get("compound", "")) for c in candidates]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(kg, rev, c="#E07B00", s=80, alpha=0.85, edgecolors="white", linewidth=0.5)

    for idx, label in enumerate(labels):
        short = label.split(" ")[0][:12]
        ax.annotate(
            short,
            (kg[idx], rev[idx]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            alpha=0.85,
        )

    if len(kg) > 2:
        z = np.polyfit(kg, rev, 1)
        p = np.poly1d(z)
        x_fit = np.linspace(max(0, kg.min() - 0.02), min(1.0, kg.max() + 0.02), 100)
        ax.plot(x_fit, p(x_fit), "--", color="gray", alpha=0.5, linewidth=1)

    ax.set_xlabel("KG prediction score (RotatE)", fontsize=11)
    ax.set_ylabel("CREEDS reversal score (cosine)", fontsize=11)
    ax.set_title(
        f"KG vs omics evidence: {title_suffix}\n"
        f"Spearman rho = {metrics['spearman_rho']:.3f} "
        f"(p = {metrics['spearman_p_value']:.4f}, n = {metrics['n']})",
        fontsize=10,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def analyze_panel(
    rows: list[dict],
    *,
    organism_policy: str,
    label: str,
) -> dict:
    matched = matched_only(rows)
    kg = np.array([c["kg_rotate_score"] for c in matched])
    rev = np.array([c["signature_reversal_score"] for c in matched])
    metrics = compute_correlation(kg, rev)
    metrics["organism_policy"] = organism_policy
    metrics["label"] = label
    metrics["n_total"] = len(rows)
    metrics["n_matched"] = len(matched)
    return {"metrics": metrics, "matched": matched}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    breast_rows = load_enriched_rows(BREAST_ENRICHED) or load_csv_rows(BREAST_CSV)
    full_rows = load_csv_rows(FULL_200_CSV)

    panel_a = analyze_panel(
        breast_rows,
        organism_policy="human",
        label="breast_ctd_cosine_human",
    )
    panel_b = analyze_panel(
        full_rows,
        organism_policy="human",
        label="full_200_cosine_matched",
    )

    print("=== Panel A: breast CtD (CREEDS matched only) ===")
    print(json.dumps(panel_a["metrics"], indent=2))
    if panel_a["matched"]:
        fig_a = FIGURES_DIR / "fig4_kg_rnaseq_correlation.png"
        plot_scatter(panel_a["matched"], panel_a["metrics"], fig_a, "breast CREEDS-matched")
        print(f"Saved: {fig_a}")

    print("\n=== Panel B: full 200-pair (CREEDS matched only) ===")
    print(json.dumps(panel_b["metrics"], indent=2))
    if panel_b["matched"]:
        fig_b = FIGURES_DIR / "fig4b_creeds_kg_rnaseq_correlation.png"
        plot_scatter(panel_b["matched"], panel_b["metrics"], fig_b, "200-pair CREEDS-matched")
        print(f"Saved: {fig_b}")

    results = {
        "panel_a_breast_matched": panel_a["metrics"],
        "panel_b_full200_matched": panel_b["metrics"],
        "description": (
            "Spearman rank correlation between KG RotatE score and CREEDS cosine "
            "reversal score on matched candidates only (signature_reversal_score > 0)."
        ),
    }
    metrics_path = RESULTS_DIR / "kg_rnaseq_correlation_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nMetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
