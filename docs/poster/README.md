# Scientific Poster

A0 portrait poster for the hybrid QML-KG drug repurposing platform.
Compiles to a single PDF (`poster.pdf`) suitable for conference printing.

## Layout

```
+--------------------------------------------------------------+
|     TITLE  · Authors · Quantum Global Group                  |
+----------------+----------------+----------------------------+
| 1. Motivation  | 2. Architecture | 3. Headline result        |
|                |   (pipeline      |   (PR-AUC = 0.7987)       |
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

The headline numbers in sections 3 and 5 should be regenerated from the
latest pipeline run before printing:

```bash
# 1. Run the pipeline
python scripts/run_full_repurposing_pipeline.py --mode kg+omics --validate --top-n 12

# 2. Build the comparison delta
python scripts/compare_pipeline_modes.py --top-n 12

# 3. Build LaTeX tables (table 8 & 9)
python scripts/build_paper_tables.py --tables 8,9

# 4. Update poster.tex section 5 to import or paste the real numbers,
#    then rebuild
bash scripts/build_poster.sh
```

Section 5 currently has hand-curated placeholder rows; future work
(Sprint 14) will auto-inject from `artifacts/predictions/mode_comparison.csv`.

## Print specs

- **Paper size**: A0 (841 × 1189 mm)
- **Orientation**: portrait
- **Color profile**: sRGB
- **Bleed**: not required (margins built into tikzposter)
- **Recommended printer**: any A0-capable plotter (HP DesignJet, Canon iPF)
- **File**: PDF/A-1b compatible after `qpdf --linearize poster.pdf poster_print.pdf`
