"""
Figure 3 — Score-Validity Scatter (Clinical Validation)
Plots model prediction score vs. ClinicalTrials.gov registration count,
exposing the score-validity inversion problem.

Run: .venv/bin/python3 figures/fig3_clinical.py
Output: figures/fig3_clinical.png + figures/fig3_clinical.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

# ── Verified data ──────────────────────────────────────────────────────────────
# Source: ClinicalTrials.gov query results documented in docs/RESULTS_EVIDENCE.md
# and docs/PAPER.md Section 7.2 (Table 8).

predictions = [
    {
        "label":   "Abacavir\n→ Ocular Cancer",
        "score":    0.793,
        "trials":   0,
        "color":   "#ef4444",   # red  — false positive
        "verdict": "False positive\n(graph artifact)",
        "rank":     1,
    },
    {
        "label":   "Ezetimibe\n→ Gout",
        "score":    0.693,
        "trials":   0,
        "color":   "#f59e0b",   # amber — novel hypothesis
        "verdict": "Novel plausible\nhypothesis",
        "rank":     2,
    },
    {
        "label":   "Ramipril\n→ Stomach Cancer",
        "score":    0.597,
        "trials":   0,
        "color":   "#64748b",   # gray — no support
        "verdict": "No clinical\nsupport",
        "rank":     3,
    },
    {
        "label":   "Losartan\n→ Atherosclerosis",
        "score":    0.528,
        "trials":   7,
        "color":   "#22c55e",   # green — validated
        "verdict": "Strongly validated\n(Phase 4, 7+ trials)",
        "rank":     4,
    },
    {
        "label":   "Mitomycin\n→ Liver Cancer",
        "score":    0.525,
        "trials":   7,
        "color":   "#22c55e",   # green — validated
        "verdict": "Strongly validated\n(TACE, 7 trials)",
        "rank":     5,
    },
    {
        "label":   "Salmeterol\n→ Liver Cancer",
        "score":    0.520,
        "trials":   0,
        "color":   "#64748b",   # gray — no support
        "verdict": "No clinical\nsupport",
        "rank":     6,
    },
]

# ── Palette ────────────────────────────────────────────────────────────────────
BG      = "#0f1117"
CARD    = "#1a1d2e"
WHITE   = "#f1f5f9"
GRAY_LT = "#94a3b8"
GRAY    = "#64748b"
RED     = "#ef4444"
AMBER   = "#f59e0b"
GREEN   = "#22c55e"
BLUE    = "#4a9eff"

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5), facecolor=BG)
ax.set_facecolor(CARD)
for spine in ax.spines.values():
    spine.set_edgecolor("#2d3748")
    spine.set_linewidth(0.8)

# ── "Ideal zone" shading ────────────────────────────────────────────────────────
ideal = FancyBboxPatch((0.72, 5.5), 0.12, 2.3,
                       boxstyle="round,pad=0.02",
                       facecolor=GREEN, edgecolor="none", alpha=0.08, zorder=1)
ax.add_patch(ideal)
ax.text(0.775, 7.95, "Ideal zone\n(high score + high trials)",
        ha="center", va="center", color=GREEN, fontsize=7.5,
        alpha=0.65, zorder=2, style="italic")

# ── Grid ───────────────────────────────────────────────────────────────────────
ax.grid(axis="y", color="#2d3748", linewidth=0.7, zorder=1)
ax.grid(axis="x", color="#2d3748", linewidth=0.4, zorder=1, alpha=0.5)

# ── Plot points ────────────────────────────────────────────────────────────────
# Size proportional to trial count (with a minimum for zero-trial points)
for p in predictions:
    size = max(p["trials"] * 90 + 120, 120)
    ax.scatter(p["score"], p["trials"],
               s=size, color=p["color"],
               edgecolors="white", linewidths=1.0,
               zorder=5, alpha=0.92)

# ── Point labels ───────────────────────────────────────────────────────────────
label_offsets = {
    "Abacavir\n→ Ocular Cancer":    (-0.038,  0.55),
    "Ezetimibe\n→ Gout":            (-0.038,  0.55),
    "Ramipril\n→ Stomach Cancer":   (-0.038,  0.55),
    "Losartan\n→ Atherosclerosis":  ( 0.013,  0.55),
    "Mitomycin\n→ Liver Cancer":    (-0.073, -1.05),
    "Salmeterol\n→ Liver Cancer":   (-0.038,  0.55),
}

for p in predictions:
    dx, dy = label_offsets[p["label"]]
    ax.text(p["score"] + dx, p["trials"] + dy,
            p["label"],
            color=p["color"], fontsize=8.5, fontweight="bold",
            ha="center", va="bottom", zorder=6,
            bbox=dict(facecolor=CARD, edgecolor="none", alpha=0.6,
                      boxstyle="round,pad=0.2"))

# ── Verdict chips ───────────────────────────────────────────────────────────────
verdict_x = 0.800
verdict_positions = {
    "Abacavir\n→ Ocular Cancer":    (0.757,  -0.55),
    "Ezetimibe\n→ Gout":            (0.693, -0.55),
    "Losartan\n→ Atherosclerosis":  (0.534,  6.7),
    "Mitomycin\n→ Liver Cancer":    (0.527,  5.9),
}
for p in predictions:
    if p["label"] in verdict_positions:
        vx, vy = verdict_positions[p["label"]]
        ax.text(vx, vy, p["verdict"],
                color=p["color"], fontsize=7.5, ha="left", va="center",
                zorder=6, alpha=0.85, style="italic")

# ── Score-validity inversion arrow ─────────────────────────────────────────────
# Draw a curved arrow from Abacavir (high score, 0 trials) toward Losartan (low score, 7 trials)
ax.annotate(
    "",
    xy=(0.532, 6.5),
    xytext=(0.785, 0.45),
    arrowprops=dict(
        arrowstyle="-|>",
        color="#94a3b8",
        lw=1.6,
        mutation_scale=14,
        connectionstyle="arc3,rad=0.35",
        linestyle="dashed"
    ),
    zorder=4
)
ax.text(0.635, 3.8,
        "score-validity\ninversion",
        color=GRAY_LT, fontsize=9, ha="center", va="center",
        rotation=-52, zorder=5, style="italic",
        bbox=dict(facecolor=CARD, edgecolor="none", alpha=0.7, pad=2))

# ── Axes formatting ────────────────────────────────────────────────────────────
ax.set_xlim(0.46, 0.86)
ax.set_ylim(-0.8, 9.2)
ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8])
ax.tick_params(axis="both", colors=GRAY_LT, labelsize=9)
ax.set_xlabel("Model Prediction Score  (CtD probability estimate)",
              color=WHITE, fontsize=11, labelpad=10)
ax.set_ylabel("ClinicalTrials.gov Registrations  (any phase/status)",
              color=WHITE, fontsize=11, labelpad=10)

# ── Rank strip on right ─────────────────────────────────────────────────────────
ax2 = ax.twinx()
ax2.set_ylim(-0.8, 9.2)
ax2.set_yticks([])
ax2.set_facecolor("none")
for spine in ax2.spines.values():
    spine.set_visible(False)

# ── Legend ─────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=RED,   edgecolor="white", linewidth=0.8,
                   label="False positive  (0 trials, high score)"),
    mpatches.Patch(facecolor=AMBER, edgecolor="white", linewidth=0.8,
                   label="Novel plausible hypothesis  (0 trials, indirect evidence)"),
    mpatches.Patch(facecolor=GREEN, edgecolor="white", linewidth=0.8,
                   label="Strongly validated  (7+ trials, Phase 2–4)"),
    mpatches.Patch(facecolor=GRAY,  edgecolor="white", linewidth=0.8,
                   label="No clinical support"),
]
leg = ax.legend(handles=legend_patches, loc="upper left",
                facecolor="#1e2235", edgecolor="#2d3748",
                labelcolor=WHITE, fontsize=8.5, framealpha=0.95)

# ── Size legend ─────────────────────────────────────────────────────────────────
for n_trials, label in [(0, "0 trials"), (7, "7 trials")]:
    size = max(n_trials * 90 + 120, 120)
    ax.scatter([], [], s=size, color=GRAY_LT, alpha=0.5,
               label=label, edgecolors="white", linewidths=0.8)
leg2 = ax.legend(
    *[*zip(*[(plt.scatter([], [], s=max(n*90+120,120), color=GRAY_LT,
                          alpha=0.5, edgecolors="white", linewidths=0.8), f"{n} trials")
              for n in [0, 7]])],
    loc="center left",
    title="Point size", title_fontsize=8,
    facecolor="#1e2235", edgecolor="#2d3748",
    labelcolor=WHITE, fontsize=8.5, framealpha=0.95
)
ax.add_artist(leg)

# ── Titles ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Score-Validity Inversion: Clinical Trial Validation of Top Predictions",
    color=WHITE, fontsize=12.5, fontweight="bold", pad=14
)
ax.text(
    0.5, 1.022,
    "Highest model score ≠ most clinically validated  ·  "
    "Motivates the mechanism-of-action (MoA) feature module",
    transform=ax.transAxes,
    ha="center", va="bottom", color=GRAY_LT, fontsize=8.5
)

plt.tight_layout(pad=1.4)

for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig3_clinical.{ext}",
                dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")

print("Saved: figures/fig3_clinical.png  +  figures/fig3_clinical.pdf")
