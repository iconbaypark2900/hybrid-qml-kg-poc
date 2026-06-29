# Scientific Poster

A0 portrait poster for the hybrid QML-KG drug repurposing platform.
Compiles to a single PDF (`poster.pdf`) suitable for conference printing.

## Layout

```
+--------------------------------------------------------------+
|     TITLE  · Authors · Quantum Global Group                  |
+----------------+----------------+----------------------------+
| 1. Motivation  | 2. Architecture | 3. Headline result        |
|                |   (pipeline      |   (PR-AUC = 0.7805)       |
|                |    diagram)      |                           |
+----------------+----------------+----------------------------+
| 4. Methods                      | 5. Per-disease results    |
|    (KG, QML, signatures,        |    (compound table +      |
|     reversal score, fusion)     |     validation hits)      |
+----------------+----------------+----------------------------+
| 6. Reproducibility | 7. Limitations | 8. Future work        |
+--------------------+----------------+------------------------+
```

## Build

```bash
# From the repo root
bash scripts/build_poster.sh
```

Or manually:

```bash
cd docs/poster
pdflatex poster && pdflatex poster
```

Output: `docs/poster/poster.pdf` (A0 portrait, ~1.5 MB).

## Dependencies

Requires `pdflatex` (TeX Live) plus the `tikzposter`, `tikz`, `tcolorbox`,
`booktabs`, and `lmodern` packages.

On Ubuntu / Debian:
```bash
sudo apt-get install texlive-latex-extra texlive-pictures
```

On macOS (MacTeX):
```bash
sudo tlmgr install tikzposter tcolorbox
```

## Customizing

| What | Where to edit |
|------|---------------|
| Title / authors | `\title{...}`, `\author{...}`, `\institute{...}` near the top |
| Color theme | `\definecolor{qggblue}{RGB}{20, 70, 140}` block |
| Section content | Each `\block{N. Title}{...}` |
| Pipeline diagram | TikZ block in section 2 |
| Headline number | The `\Huge\bfseries PR-AUC = ...` line in section 3 |
| Per-disease table | Section 5 — replace placeholder rows with real `top_candidates.csv` data |
| Conference logo | Add to `\titlegraphic{\includegraphics{...}}` in the title block |

## Updating from real artifacts

The headline CtD numbers in section 3 come from
[`results/multiseed/TABLE3.md`](../../results/multiseed/TABLE3.md).
Section 5 breast-cancer rows are sourced from
[`artifacts/repurposing/brca_external_validation/repurposing_evidence_bundle.json`](../../artifacts/repurposing/brca_external_validation/repurposing_evidence_bundle.json).

```bash
# 1. Refresh repurposing bundles (optional)
./scripts/run_repurposing_workbench_refresh.sh

# 2. Edit poster.tex section 5 if bundle scores change, then rebuild
bash scripts/build_poster.sh
```

For full pipeline demos (kg-only vs kg+omics mode comparison), see
`scripts/run_full_repurposing_pipeline.py` and `scripts/compare_pipeline_modes.py`.

## Print specs

- **Paper size**: A0 (841 × 1189 mm)
- **Orientation**: portrait
- **Color profile**: sRGB
- **Bleed**: not required (margins built into tikzposter)
- **Recommended printer**: any A0-capable plotter (HP DesignJet, Canon iPF)
- **File**: PDF/A-1b compatible after `qpdf --linearize poster.pdf poster_print.pdf`
