#!/usr/bin/env bash
# build_papers.sh — Compile LaTeX papers to PDF.
#
# Outputs:
#   docs/qgg_biomedical.pdf          (full arXiv / reference)
#   docs/qgg_biomedical_qce.pdf      (QCE short-track, ~6 pages)
#   docs/paper_qGG_full_2026.pdf
#
# Uses host pdflatex when available; otherwise Docker (texlive/texlive).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_latex_sequence() {
  local workdir="$1"
  local basename="$2"
  local use_bibtex="${3:-yes}"

  cd "$workdir"
  echo "=== pdflatex pass 1: $basename ==="
  pdflatex -interaction=nonstopmode -halt-on-error "$basename.tex"
  if [[ "$use_bibtex" == "yes" ]]; then
    echo "=== bibtex: $basename ==="
    bibtex "$basename"
  fi
  echo "=== pdflatex pass 2: $basename ==="
  pdflatex -interaction=nonstopmode -halt-on-error "$basename.tex"
  echo "=== pdflatex pass 3: $basename ==="
  pdflatex -interaction=nonstopmode -halt-on-error "$basename.tex"
  ls -lh "$basename.pdf"
}

docker_latex_sequence() {
  local workdir_rel="$1"
  local basename="$2"
  local use_bibtex="${3:-yes}"

  docker run --rm \
    -u "$(id -u):$(id -g)" \
    -v "$ROOT:/work" \
    -w "/work/$workdir_rel" \
    texlive/texlive:latest \
    bash -lc "
      set -euo pipefail
      pdflatex -interaction=nonstopmode -halt-on-error '$basename.tex'
      if [[ '$use_bibtex' == 'yes' ]]; then bibtex '$basename'; fi
      pdflatex -interaction=nonstopmode -halt-on-error '$basename.tex'
      pdflatex -interaction=nonstopmode -halt-on-error '$basename.tex'
      ls -lh '$basename.pdf'
    "
}

build_one() {
  local basename="$1"
  local use_bibtex="$2"
  if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    run_latex_sequence "$ROOT/docs" "$basename" "$use_bibtex"
  else
    docker_latex_sequence "docs" "$basename" "$use_bibtex"
  fi
}

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  echo "[info] Using host TeX Live"
else
  echo "[info] Host pdflatex not found; using Docker texlive/texlive:latest"
  if ! command -v docker >/dev/null 2>&1; then
    echo "[error] Neither pdflatex nor docker available." >&2
    exit 1
  fi
fi

build_one "qgg_biomedical" "yes"
build_one "qgg_biomedical_qce" "yes"
build_one "paper_qGG_full_2026" "no"

echo ""
echo "=== Papers built ==="
ls -lh docs/qgg_biomedical.pdf docs/qgg_biomedical_qce.pdf docs/paper_qGG_full_2026.pdf
