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

# ── Palette (light theme, high-contrast text for print/PDF) ────────────────────
BG        = "#f4f4f5"
CARD      = "#ffffff"
PURPLE_LT = "#9333ea"   # QSVC bars — darker for contrast on white
PURPLE_DK = "#6b21a8"
TEAL_LT   = "#059669"   # ensemble bars
TEAL_DK   = "#047857"
BASELINE  = "#b45309"   # RF dashed line (amber-700 on white)
WHITE     = "#0f172a"   # axis / title (dark)
SUBTITLE  = "#334155"    # slate-700 — not low-contrast grey
GRID      = "#cbd5e1"

# ── Layout ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6.5), facecolor=BG)
ax.set_facecolor(CARD)
for spine in ax.spines.values():
    spine.set_edgecolor("#64748b")
    spine.set_linewidth(1.0)

labels    = list(data.keys())
x         = np.array([0, 1])
bar_w     = 0.28
gap       = 0.05

# ── Bars ───────────────────────────────────────────────────────────────────────
qsvc_vals = [data[k]["qsvc"]     for k in labels]
ens_vals  = [data[k]["ensemble"] for k in labels]

bars_q = ax.bar(x - bar_w/2 - gap/2, qsvc_vals, bar_w,
                color="#c084fc", alpha=0.95,
                label="QSVC standalone",
                zorder=3, linewidth=0)
bars_e = ax.bar(x + bar_w/2 + gap/2, ens_vals, bar_w,
                color="#34d399", alpha=0.95,
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
            color=PURPLE_DK, fontsize=11, fontweight="bold", zorder=5)

for bar, val in zip(bars_e, ens_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.004,
            f"{val:.4f}", ha="center", va="bottom",
            color=TEAL_DK, fontsize=11, fontweight="bold", zorder=5)

# ── RF Baseline ────────────────────────────────────────────────────────────────
ax.axhline(RF_BASELINE, color=BASELINE, lw=1.8, ls="--", zorder=2, alpha=0.9)
ax.text(1.52, RF_BASELINE + 0.003,
        f"RF baseline  {RF_BASELINE:.4f}",
        color=BASELINE, fontsize=10, va="bottom", ha="right", fontweight="bold",
        zorder=5)

# ── Delta annotations ──────────────────────────────────────────────────────────
# QSVC drop arrow:  ZZ(0.7216) → Pauli(0.6343)
ax.annotate(
    "", xy=(x[1] - bar_w/2 - gap/2, qsvc_vals[1] + 0.005),
    xytext=(x[0] - bar_w/2 - gap/2, qsvc_vals[0] - 0.005),
    arrowprops=dict(arrowstyle="-|>", color=PURPLE_DK, lw=1.5,
                    mutation_scale=12,
                    connectionstyle="arc3,rad=-0.45"),
    zorder=5
)
ax.text(0.5 - bar_w/2 - 0.12, (qsvc_vals[0] + qsvc_vals[1])/2,
        "−8.7 pp\n(QSVC drops)",
        color=PURPLE_DK, fontsize=10, ha="center", va="center",
        zorder=5, fontweight="bold",
        bbox=dict(facecolor="#fafafa", edgecolor=PURPLE_DK, alpha=0.95,
                  pad=3, linewidth=1.0))

# Ensemble rise arrow: ZZ(0.7408) → Pauli(0.7987)
ax.annotate(
    "", xy=(x[1] + bar_w/2 + gap/2, ens_vals[1] - 0.005),
    xytext=(x[0] + bar_w/2 + gap/2, ens_vals[0] + 0.005),
    arrowprops=dict(arrowstyle="-|>", color=TEAL_DK, lw=1.5,
                    mutation_scale=12,
                    connectionstyle="arc3,rad=0.45"),
    zorder=5
)
ax.text(0.5 + bar_w/2 + 0.17, (ens_vals[0] + ens_vals[1])/2,
        "+5.8 pp\n(ensemble rises)",
        color=TEAL_DK, fontsize=10, ha="center", va="center",
        zorder=5, fontweight="bold",
        bbox=dict(facecolor="#fafafa", edgecolor=TEAL_DK, alpha=0.95,
                  pad=3, linewidth=1.0))

# ── Best result callout ────────────────────────────────────────────────────────
ax.annotate(
    "  Best result\n  PR-AUC 0.7987",
    xy=(bars_e[1].get_x() + bars_e[1].get_width()/2, 0.7987),
    xytext=(1.42, 0.782),
    color=TEAL_DK, fontsize=10, fontweight="bold", zorder=6,
    arrowprops=dict(arrowstyle="-|>", color=TEAL_DK, lw=1.4,
                    mutation_scale=11),
    bbox=dict(facecolor="#ecfdf5", edgecolor=TEAL_DK, alpha=0.98, pad=4,
              boxstyle="round,pad=0.35", linewidth=1.5)
)

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(["ZZ Feature Map\n(reps=2, 16 qubits)", "Pauli Feature Map\n(reps=2, 16 qubits)"],
                   color=WHITE, fontsize=12, fontweight="bold")
ax.tick_params(axis="x", colors=WHITE, length=0)
ax.tick_params(axis="y", colors="#475569", labelsize=10)
ax.set_xlim(-0.55, 1.95)
ax.set_ylim(0.58, 0.835)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.set_ylabel("Test PR-AUC", color=WHITE, fontsize=12, labelpad=10, fontweight="bold")
ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=1)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor="#c084fc", edgecolor=PURPLE_DK, linewidth=1.5,
                   label="QSVC standalone"),
    mpatches.Patch(facecolor="#34d399", edgecolor=TEAL_DK, linewidth=1.5,
                   label="Stacking ensemble  (RF + ET + QSVC)"),
    mpatches.Patch(facecolor="none",    edgecolor=BASELINE,   linewidth=1.8,
                   linestyle="--",      label=f"Best classical baseline (RF, {RF_BASELINE:.4f})"),
]
leg = ax.legend(handles=legend_patches, loc="lower left",
                facecolor="#fafafa", edgecolor="#94a3b8",
                labelcolor=WHITE, fontsize=10, framealpha=1.0)

# ── Titles ─────────────────────────────────────────────────────────────────────
# suptitle = main title, set_title = subtitle — no overlap
fig.suptitle(
    "Feature Map Selection: The Pauli Inversion Effect",
    color=WHITE, fontsize=15, fontweight="bold", y=1.0
)
ax.set_title(
    "ZZ → Pauli: QSVC solo drops 8.7 pp  ·  Ensemble rises 5.8 pp\n"
    "Pauli kernel generates more decorrelated predictions for the stacking meta-learner",
    color=SUBTITLE, fontsize=10.5, pad=8, linespacing=1.4
)

fig.patch.set_facecolor(BG)
plt.tight_layout(pad=1.4)
plt.subplots_adjust(top=0.88)

for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig2_pauli_zz.{ext}",
                dpi=220, bbox_inches="tight",
                facecolor=BG, edgecolor="none")

print("Saved: figures/fig2_pauli_zz.png  +  figures/fig2_pauli_zz.pdf")
