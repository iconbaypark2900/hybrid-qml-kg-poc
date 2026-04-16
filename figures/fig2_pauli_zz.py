"""
Figure 2 — Pauli vs ZZ Tradeoff (The Key Quantum Finding)
Grouped bar chart showing QSVC solo vs. Ensemble PR-AUC for ZZ and Pauli feature maps.
All values verified from raw JSON result files.

Run: .venv/bin/python3 figures/fig2_pauli_zz.py
Output: figures/fig2_pauli_zz.png + figures/fig2_pauli_zz.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── Verified data ──────────────────────────────────────────────────────────────
# Source:
#   ZZ:    results/optimized_results_20260216-084359.json  (QSVC fit=908s, genuine)
#          results/optimized_results_20260216-091710.json  (stacking ensemble)
#   Pauli: results/optimized_results_20260216-100431.json  (QSVC fit=2619s, genuine)

data = {
    "ZZ":    {"qsvc": 0.7216, "ensemble": 0.7408},
    "Pauli": {"qsvc": 0.6343, "ensemble": 0.7987},
}
RF_BASELINE = 0.7838   # RandomForest-Optimized, same run

# ── Palette ────────────────────────────────────────────────────────────────────
BG        = "#0f1117"
CARD      = "#1a1d2e"
PURPLE_LT = "#c084fc"   # QSVC bars
PURPLE_DK = "#7c3aed"
TEAL_LT   = "#34d399"   # ensemble bars
TEAL_DK   = "#059669"
BASELINE  = "#f59e0b"   # RF dashed line
WHITE     = "#f1f5f9"
GRAY      = "#64748b"
GRAY_LT   = "#94a3b8"

# ── Layout ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG)
ax.set_facecolor(CARD)
for spine in ax.spines.values():
    spine.set_edgecolor("#2d3748")
    spine.set_linewidth(0.8)

labels    = list(data.keys())
x         = np.array([0, 1])
bar_w     = 0.28
gap       = 0.05

# ── Bars ───────────────────────────────────────────────────────────────────────
qsvc_vals = [data[k]["qsvc"]     for k in labels]
ens_vals  = [data[k]["ensemble"] for k in labels]

bars_q = ax.bar(x - bar_w/2 - gap/2, qsvc_vals, bar_w,
                color=PURPLE_LT, alpha=0.88,
                label="QSVC standalone",
                zorder=3, linewidth=0)
bars_e = ax.bar(x + bar_w/2 + gap/2, ens_vals, bar_w,
                color=TEAL_LT, alpha=0.88,
                label="Stacking ensemble",
                zorder=3, linewidth=0)

# Darker top edge accent
for bars, col in [(bars_q, PURPLE_DK), (bars_e, TEAL_DK)]:
    for bar in bars:
        ax.bar(bar.get_x(), bar.get_height(), bar.get_width(),
               bottom=0, color="none",
               edgecolor=col, linewidth=1.5, zorder=4)

# ── Value labels on bars ───────────────────────────────────────────────────────
for bar, val in zip(bars_q, qsvc_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
            f"{val:.4f}", ha="center", va="bottom",
            color=PURPLE_LT, fontsize=10, fontweight="bold", zorder=5)

for bar, val in zip(bars_e, ens_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
            f"{val:.4f}", ha="center", va="bottom",
            color=TEAL_LT, fontsize=10, fontweight="bold", zorder=5)

# ── RF Baseline ────────────────────────────────────────────────────────────────
ax.axhline(RF_BASELINE, color=BASELINE, lw=1.8, ls="--", zorder=2, alpha=0.9)
ax.text(1.52, RF_BASELINE + 0.003,
        f"RF baseline  {RF_BASELINE:.4f}",
        color=BASELINE, fontsize=9, va="bottom", ha="right", zorder=5)

# ── Delta annotations ──────────────────────────────────────────────────────────
# QSVC drop arrow:  ZZ(0.7216) → Pauli(0.6343)
ax.annotate(
    "", xy=(x[1] - bar_w/2 - gap/2, qsvc_vals[1] + 0.005),
    xytext=(x[0] - bar_w/2 - gap/2, qsvc_vals[0] - 0.005),
    arrowprops=dict(arrowstyle="-|>", color=PURPLE_LT, lw=1.5,
                    mutation_scale=12,
                    connectionstyle="arc3,rad=-0.45"),
    zorder=5
)
ax.text(0.5 - bar_w/2 - 0.12, (qsvc_vals[0] + qsvc_vals[1])/2,
        "−8.7 pp\n(QSVC drops)",
        color=PURPLE_LT, fontsize=8.5, ha="center", va="center",
        zorder=5,
        bbox=dict(facecolor=CARD, edgecolor="none", alpha=0.7, pad=2))

# Ensemble rise arrow: ZZ(0.7408) → Pauli(0.7987)
ax.annotate(
    "", xy=(x[1] + bar_w/2 + gap/2, ens_vals[1] - 0.005),
    xytext=(x[0] + bar_w/2 + gap/2, ens_vals[0] + 0.005),
    arrowprops=dict(arrowstyle="-|>", color=TEAL_LT, lw=1.5,
                    mutation_scale=12,
                    connectionstyle="arc3,rad=0.45"),
    zorder=5
)
ax.text(0.5 + bar_w/2 + 0.17, (ens_vals[0] + ens_vals[1])/2,
        "+5.8 pp\n(ensemble rises)",
        color=TEAL_LT, fontsize=8.5, ha="center", va="center",
        zorder=5,
        bbox=dict(facecolor=CARD, edgecolor="none", alpha=0.7, pad=2))

# ── Best result callout ────────────────────────────────────────────────────────
ax.annotate(
    "  Best result\n  PR-AUC 0.7987",
    xy=(bars_e[1].get_x() + bars_e[1].get_width()/2, 0.7987),
    xytext=(1.42, 0.782),
    color=TEAL_LT, fontsize=9, fontweight="bold", zorder=6,
    arrowprops=dict(arrowstyle="-|>", color=TEAL_LT, lw=1.2,
                    mutation_scale=10),
    bbox=dict(facecolor=CARD, edgecolor=TEAL_DK, alpha=0.9, pad=3,
              boxstyle="round,pad=0.3")
)

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(["ZZ Feature Map\n(reps=2, 16 qubits)", "Pauli Feature Map\n(reps=2, 16 qubits)"],
                   color=WHITE, fontsize=11, fontweight="bold")
ax.tick_params(axis="x", colors=WHITE, length=0)
ax.tick_params(axis="y", colors=GRAY_LT, labelsize=9)
ax.set_xlim(-0.55, 1.95)
ax.set_ylim(0.58, 0.835)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.set_ylabel("Test PR-AUC", color=WHITE, fontsize=11, labelpad=10)
ax.grid(axis="y", color="#2d3748", linewidth=0.7, zorder=1)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=PURPLE_LT, edgecolor=PURPLE_DK, linewidth=1.5,
                   label="QSVC standalone"),
    mpatches.Patch(facecolor=TEAL_LT,   edgecolor=TEAL_DK,   linewidth=1.5,
                   label="Stacking ensemble  (RF + ET + QSVC)"),
    mpatches.Patch(facecolor="none",    edgecolor=BASELINE,   linewidth=1.8,
                   linestyle="--",      label=f"Best classical baseline (RF, {RF_BASELINE:.4f})"),
]
leg = ax.legend(handles=legend_patches, loc="lower left",
                facecolor="#1e2235", edgecolor="#2d3748",
                labelcolor=WHITE, fontsize=9, framealpha=0.95)

# ── Titles ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Feature Map Selection: The Pauli Inversion Effect",
    color=WHITE, fontsize=13, fontweight="bold", pad=14
)
ax.text(
    0.5, 1.022,
    "ZZ → Pauli: QSVC solo drops 8.7 pp · Ensemble rises 5.8 pp\n"
    "Pauli kernel generates more decorrelated predictions for the stacking meta-learner",
    transform=ax.transAxes,
    ha="center", va="bottom", color=GRAY_LT, fontsize=8.5
)

# ── Background grid area ───────────────────────────────────────────────────────
fig.patch.set_facecolor(BG)

plt.tight_layout(pad=1.4)

for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig2_pauli_zz.{ext}",
                dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")

print("Saved: figures/fig2_pauli_zz.png  +  figures/fig2_pauli_zz.pdf")
