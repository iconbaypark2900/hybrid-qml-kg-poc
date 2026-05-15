"""
Figure 1 — Pipeline Architecture
Hybrid quantum-classical pipeline for CtD link prediction on Hetionet.
Light theme + large type for print/PDF accessibility.

Run: .venv/bin/python3 figures/fig1_pipeline.py
Output: figures/fig1_pipeline.png + figures/fig1_pipeline.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── Light palette (WCAG-friendly contrast on white/off-white) ─────────────────
BG = "#f4f4f5"
# Pastel fills + dark text (readable without relying on low-contrast grey)
ORANGE = "#ffedd5"
BLUE = "#dbeafe"
PURPLE = "#ede9fe"
GREEN = "#d1fae5"
EDGE_ORANGE = "#c2410c"
EDGE_BLUE = "#1d4ed8"
EDGE_PURPLE = "#6d28d9"
EDGE_GREEN = "#047857"
TEXT = "#0f172a"       # slate-900 — main labels
SUBTEXT = "#334155"    # slate-700 — sublabels (not grey-on-pastel)
ACCENT = "#0f172a"
ARROW_COL = "#475569"
LEGEND_BG = "#ffffff"

fig = plt.figure(figsize=(14, 9), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_facecolor(BG)


def box(ax, x, y, w, h, face, edge, label, sublabel=None,
        fontsize=12, alpha=1.0):
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=face, edgecolor=edge,
        alpha=alpha, zorder=3,
        linewidth=2.0,
    )
    ax.add_patch(patch)
    sub_fs = max(fontsize - 1, 10.5)
    if sublabel:
        ax.text(x, y + 0.13, label, ha="center", va="center", color=TEXT,
                fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center", color=SUBTEXT,
                fontsize=sub_fs, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center", color=TEXT,
                fontsize=fontsize, fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, color=ARROW_COL, lw=2.0):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=15),
        zorder=2,
    )


def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", color=color,
            fontsize=11, fontstyle="italic", fontweight="bold", zorder=4)


CX = 2.5

box(ax, CX, 8.1, 3.65, 0.72, ORANGE, EDGE_ORANGE,
    "Hetionet v1.0",
    "47,031 entities · 2.25 M edges · 24 relation types",
    fontsize=12)

arrow(ax, CX, 7.74, CX, 7.14)

box(ax, CX, 6.8, 3.65, 0.64, ORANGE, EDGE_ORANGE,
    "Full-Graph RotatE Embeddings",
    "128D or 256D · 200–250 epochs · PyKEEN",
    fontsize=11.5)

arrow(ax, CX, 6.49, CX, 5.89)

box(ax, CX, 5.55, 3.65, 0.64, ORANGE, EDGE_ORANGE,
    "CtD Pair Construction",
    "604 train pos + 604 hard neg = 1,208 pairs · 80/20 split",
    fontsize=11.5)

arrow(ax, CX, 5.24, CX, 4.64)

box(ax, CX, 4.3, 3.65, 0.64, ORANGE, EDGE_ORANGE,
    "Pair Feature Engineering",
    "concat · diff · Hadamard · graph topology  →  ~299 features",
    fontsize=11.5)

split_y = 3.7
arrow(ax, CX, 3.99, CX, split_y + 0.02)

ax.annotate("", xy=(5.5, split_y), xytext=(CX, split_y),
            arrowprops=dict(arrowstyle="-", color=EDGE_BLUE, lw=2.2), zorder=2)
ax.annotate("", xy=(CX, split_y), xytext=(9.5, split_y),
            arrowprops=dict(arrowstyle="-", color=EDGE_PURPLE, lw=2.2), zorder=2)

CCX = 5.5
section_label(ax, CCX, 3.38, "Classical Path", EDGE_BLUE)

arrow(ax, CCX, split_y, CCX, 3.1 + 0.04)

box(ax, CCX, 2.77, 3.15, 0.60, BLUE, EDGE_BLUE,
    "Classical Models",
    "RandomForest · ExtraTrees · LogisticRegression",
    fontsize=11.5)

arrow(ax, CCX, 2.48, CCX, 1.88)

box(ax, CCX, 1.55, 3.15, 0.60, BLUE, EDGE_BLUE,
    "GridSearchCV Tuning",
    "5-fold CV · hyperparameter optimization",
    fontsize=11.5)

QCX = 9.5
section_label(ax, QCX, 3.38, "Quantum Path", EDGE_PURPLE)

arrow(ax, QCX, split_y, QCX, 3.1 + 0.04)

box(ax, QCX, 2.77, 3.15, 0.60, PURPLE, EDGE_PURPLE,
    "Dimensionality Reduction",
    "PCA  299D → 24D → 16 qubits",
    fontsize=11.5)

arrow(ax, QCX, 2.48, QCX, 1.88)

box(ax, QCX, 1.55, 3.15, 0.60, PURPLE, EDGE_PURPLE,
    "Pauli Feature Map  ·  QSVC",
    "16 qubits · reps=2 · fidelity kernel · C=0.1 · 2,619 s",
    fontsize=11.5)

MX = 7.5
merge_y = 0.85

ax.annotate("", xy=(MX - 0.3, merge_y + 0.38), xytext=(CCX, 1.26),
            arrowprops=dict(arrowstyle="-|>", color=EDGE_BLUE, lw=2.0,
                            mutation_scale=14, connectionstyle="arc3,rad=-0.25"),
            zorder=2)

ax.annotate("", xy=(MX + 0.3, merge_y + 0.38), xytext=(QCX, 1.26),
            arrowprops=dict(arrowstyle="-|>", color=EDGE_PURPLE, lw=2.0,
                            mutation_scale=14, connectionstyle="arc3,rad=0.25"),
            zorder=2)

box(ax, MX, merge_y, 3.65, 0.64, GREEN, EDGE_GREEN,
    "Stacking Ensemble",
    "LR meta-learner on [RF, ET, QSVC] out-of-fold predictions",
    fontsize=12)

arrow(ax, MX, merge_y - 0.31, MX, 0.2)
ax.text(MX, 0.08, "PR-AUC  0.7987  (primary)  ·  0.8581  (extended)",
        ha="center", va="center", color=EDGE_GREEN,
        fontsize=12, fontweight="bold", zorder=4)

# ── Title block — left half of top strip (x 0–9.8) ─────────────────────────────
ax.text(4.9, 8.76, "Hybrid Quantum-Classical Pipeline for Drug Repurposing",
        ha="center", va="center", color=ACCENT,
        fontsize=15.5, fontweight="bold", zorder=4)
ax.text(4.9, 8.36, "Hetionet CtD Link Prediction  ·  RotatE + QSVC + Stacking Ensemble",
        ha="center", va="center", color=SUBTEXT,
        fontsize=11, zorder=4)

# ── Legend chips — right half of top strip (x 10.0–14) ──────────────────────────
legend_items = [
    (ORANGE, EDGE_ORANGE, "Data / KG layer"),
    (BLUE, EDGE_BLUE, "Classical path"),
    (PURPLE, EDGE_PURPLE, "Quantum path"),
    (GREEN, EDGE_GREEN, "Ensemble / output"),
]
for i, (face, edge, lbl) in enumerate(legend_items):
    lx = 10.1 + (i % 2) * 1.9
    ly = 8.70 - (i // 2) * 0.46
    patch = FancyBboxPatch((lx - 0.13, ly - 0.13), 0.26, 0.26,
                           boxstyle="round,pad=0.04",
                           facecolor=face, edgecolor=edge, linewidth=1.5,
                           alpha=1.0, zorder=4)
    ax.add_patch(patch)
    ax.text(lx + 0.24, ly, lbl, va="center", color=TEXT,
            fontsize=10, zorder=4)

for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig1_pipeline.{ext}",
                dpi=220, bbox_inches="tight",
                facecolor=BG, edgecolor="none")

print("Saved: figures/fig1_pipeline.png  +  figures/fig1_pipeline.pdf")
