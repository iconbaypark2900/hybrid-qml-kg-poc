#!/usr/bin/env python3
"""Render all three paper figures as PDFs (and PNGs) in figures/.

Produces:
  figures/fig1_pipeline.pdf  - two-column pipeline architecture flowchart
  figures/fig2_pauli_zz.pdf  - grouped bar chart: ZZ vs Pauli feature maps
  figures/fig3_clinical.pdf  - score-validity scatter (6 predictions)

Run:  .venv/bin/python scripts/render_figures.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = REPO_ROOT / "figures"

FIGURE_SCRIPTS = [
    "fig1_pipeline.py",
    "fig2_pauli_zz.py",
    "fig3_clinical.py",
]

REQUIRED_PDFS = [
    FIGURES_DIR / "fig1_pipeline.pdf",
    FIGURES_DIR / "fig2_pauli_zz.pdf",
    FIGURES_DIR / "fig3_clinical.pdf",
]


def main() -> int:
    if str(FIGURES_DIR) not in sys.path:
        sys.path.insert(0, str(FIGURES_DIR))

    for script in FIGURE_SCRIPTS:
        path = FIGURES_DIR / script
        if not path.exists():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1
        print(f"Rendering {script} ...")
        runpy.run_path(str(path), run_name="__main__")

    missing = [p for p in REQUIRED_PDFS if not p.exists()]
    if missing:
        print(f"ERROR: missing PDFs: {missing}", file=sys.stderr)
        return 1

    print("\nAll 3 figures rendered:")
    for p in REQUIRED_PDFS:
        print(f"  {p.relative_to(REPO_ROOT)}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
