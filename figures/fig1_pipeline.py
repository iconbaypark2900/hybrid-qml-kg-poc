"""
Figure 1 — Pipeline Architecture
Hybrid quantum-classical pipeline for CtD link prediction on Hetionet.
Run: .venv/bin/python3 figures/fig1_pipeline.py
Output: figures/fig1_pipeline.png + figures/fig1_pipeline.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0f1117"
CARD_DARK   = "#1a1d2e"
CARD_MID    = "#1e2235"
BLUE        = "#4a9eff"      # classical path
PURPLE      = "#a855f7"      # quantum path
GREEN       = "#22c55e"      # ensemble / output
ORANGE      = "#f59e0b"      # data / KG
GRAY_TEXT   = "#94a3b8"
WHITE       = "#f1f5f9"
ARROW_COL   = "#475569"

fig = plt.figure(figsize=(14, 9), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_facecolor(BG)

# ── Helper: rounded box ────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel=None, text_color=WHITE, fontsize=10, alpha=0.92):
    patch = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor=color,
        alpha=alpha, zorder=3,
        linewidth=1.5
    )
    ax.add_patch(patch)
    if sublabel:
        ax.text(x, y + 0.13, label,   ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight="bold", zorder=4)
        ax.text(x, y - 0.22, sublabel, ha="center", va="center", color=GRAY_TEXT,
                fontsize=fontsize - 1.5, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight="bold", zorder=4)

# ── Helper: arrow ──────────────────────────────────────────────────────────────
def arrow(ax, x1, y1, x2, y2, color=ARROW_COL, lw=1.8):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=14),
        zorder=2
    )

# ── Helper: section label ──────────────────────────────────────────────────────
def section_label(ax, x, y, text, color):
    ax.text(x, y, text, ha="center", va="center", color=color,
            fontsize=8.5, fontstyle="italic", zorder=4, alpha=0.8)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 1 — DATA LAYER  (x ≈ 2.5)
# ══════════════════════════════════════════════════════════════════════════════
CX = 2.5   # centre x

box(ax, CX, 8.1, 3.6, 0.7, ORANGE,
    "Hetionet v1.0",
    "47,031 entities · 2.25 M edges · 24 relation types",
    fontsize=9.5)

arrow(ax, CX, 7.74, CX, 7.14)

box(ax, CX, 6.8, 3.6, 0.62, ORANGE,
    "Full-Graph RotatE Embeddings",
    "128D or 256D · 200–250 epochs · PyKEEN",
    fontsize=9)

arrow(ax, CX, 6.49, CX, 5.89)

box(ax, CX, 5.55, 3.6, 0.62, ORANGE,
    "CtD Pair Construction",
    "604 train pos + 604 hard neg = 1,208 pairs · 80/20 split",
    fontsize=9)

arrow(ax, CX, 5.24, CX, 4.64)

box(ax, CX, 4.3, 3.6, 0.62, ORANGE,
    "Pair Feature Engineering",
    "concat · diff · Hadamard · graph topology  →  ~299 features",
    fontsize=9)

# ── Split arrow ────────────────────────────────────────────────────────────────
# trunk down to split point
split_y = 3.7
arrow(ax, CX, 3.99, CX, split_y + 0.02)

# horizontal branch lines
ax.annotate("", xy=(5.5, split_y),  xytext=(CX, split_y),
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=2.0), zorder=2)
ax.annotate("", xy=(CX, split_y),   xytext=(9.5, split_y),
            arrowprops=dict(arrowstyle="-", color=PURPLE, lw=2.0), zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 2 — CLASSICAL PATH  (x ≈ 5.5)
# ══════════════════════════════════════════════════════════════════════════════
CCX = 5.5
section_label(ax, CCX, 3.38, "Classical Path", BLUE)

arrow(ax, CCX, split_y, CCX, 3.1 + 0.04)

box(ax, CCX, 2.77, 3.1, 0.58, BLUE,
    "Classical Models",
    "RandomForest · ExtraTrees · LogisticRegression",
    fontsize=9)

arrow(ax, CCX, 2.48, CCX, 1.88)

box(ax, CCX, 1.55, 3.1, 0.58, BLUE,
    "GridSearchCV Tuning",
    "5-fold CV · hyperparameter optimization",
    fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN 3 — QUANTUM PATH  (x ≈ 9.5)
# ══════════════════════════════════════════════════════════════════════════════
QCX = 9.5
section_label(ax, QCX, 3.38, "Quantum Path", PURPLE)

arrow(ax, QCX, split_y, QCX, 3.1 + 0.04)

box(ax, QCX, 2.77, 3.1, 0.58, PURPLE,
    "Dimensionality Reduction",
    "PCA  299D → 24D → 16 qubits",
    fontsize=9)

arrow(ax, QCX, 2.48, QCX, 1.88)

box(ax, QCX, 1.55, 3.1, 0.58, PURPLE,
    "Pauli Feature Map  ·  QSVC",
    "16 qubits · reps=2 · fidelity kernel · C=0.1 · 2,619 s",
    fontsize=9)

# ══════════════════════════════════════════════════════════════════════════════
# MERGE — STACKING ENSEMBLE  (x ≈ 7.5)
# ══════════════════════════════════════════════════════════════════════════════
MX = 7.5
merge_y = 0.85

# classical → merge
ax.annotate("", xy=(MX - 0.3, merge_y + 0.38),  xytext=(CCX, 1.26),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.0,
                            mutation_scale=13, connectionstyle="arc3,rad=-0.25"),
            zorder=2)

# quantum → merge
ax.annotate("", xy=(MX + 0.3, merge_y + 0.38),  xytext=(QCX, 1.26),
            arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=2.0,
                            mutation_scale=13, connectionstyle="arc3,rad=0.25"),
            zorder=2)

box(ax, MX, merge_y, 3.6, 0.62, GREEN,
    "Stacking Ensemble",
    "LR meta-learner on [RF, ET, QSVC] out-of-fold predictions",
    fontsize=9.5)

# ── output arrow + metric ──────────────────────────────────────────────────────
arrow(ax, MX, merge_y - 0.31, MX, 0.2)
ax.text(MX, 0.08, "PR-AUC  0.7987  (primary)  ·  0.8581  (extended)",
        ha="center", va="center", color=GREEN,
        fontsize=9.5, fontweight="bold", zorder=4)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND CHIPS
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (ORANGE, "Data / KG layer"),
    (BLUE,   "Classical path"),
    (PURPLE, "Quantum path"),
    (GREEN,  "Ensemble / output"),
]
for i, (col, lbl) in enumerate(legend_items):
    lx = 10.3 + (i % 2) * 1.9
    ly = 8.55 - (i // 2) * 0.42
    patch = FancyBboxPatch((lx - 0.13, ly - 0.13), 0.26, 0.26,
                           boxstyle="round,pad=0.04",
                           facecolor=col, edgecolor="none", alpha=0.9, zorder=4)
    ax.add_patch(patch)
    ax.text(lx + 0.22, ly, lbl, va="center", color=GRAY_TEXT,
            fontsize=8, zorder=4)

# ── title ──────────────────────────────────────────────────────────────────────
ax.text(7.0, 8.7, "Hybrid Quantum-Classical Pipeline for Drug Repurposing",
        ha="center", va="center", color=WHITE,
        fontsize=13, fontweight="bold", zorder=4)
ax.text(7.0, 8.35, "Hetionet CtD Link Prediction  ·  RotatE + QSVC + Stacking Ensemble",
        ha="center", va="center", color=GRAY_TEXT,
        fontsize=9, zorder=4)

# ── save ──────────────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    plt.savefig(f"figures/fig1_pipeline.{ext}",
                dpi=200, bbox_inches="tight",
                facecolor=BG, edgecolor="none")

print("Saved: figures/fig1_pipeline.png  +  figures/fig1_pipeline.pdf")
