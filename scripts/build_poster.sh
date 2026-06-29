#!/usr/bin/env bash
# build_poster.sh — Compile docs/poster/poster.tex to PDF.
#
# Output: docs/poster/poster.pdf
#
# Requires: pdflatex (TeX Live) with the tikzposter package. On Ubuntu:
#   sudo apt-get install texlive-latex-extra texlive-pictures
set -euo pipefail

POSTER_DIR="docs/poster"
POSTER_NAME="poster"

if [ ! -f "$POSTER_DIR/$POSTER_NAME.tex" ]; then
    echo "[error] $POSTER_DIR/$POSTER_NAME.tex not found." >&2
    exit 1
fi

if ! command -v pdflatex >/dev/null 2>&1; then
    ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    if command -v docker >/dev/null 2>&1; then
        echo "[info] pdflatex not found; building poster via Docker texlive/texlive:latest"
        docker run --rm \
            -u "$(id -u):$(id -g)" \
            -v "$ROOT:/work" \
            -w "/work/$POSTER_DIR" \
            texlive/texlive:latest \
            bash -lc "
                set -euo pipefail
                pdflatex -interaction=nonstopmode -halt-on-error '$POSTER_NAME.tex'
                pdflatex -interaction=nonstopmode -halt-on-error '$POSTER_NAME.tex'
                ls -lh '$POSTER_NAME.pdf'
            "
        exit 0
    fi
    echo "[error] pdflatex not found. Install with:" >&2
    echo "  sudo apt-get install texlive-latex-extra texlive-pictures" >&2
    exit 1
fi

cd "$POSTER_DIR"

echo "=== Compiling $POSTER_NAME.tex (pass 1/2) ==="
pdflatex -interaction=nonstopmode -halt-on-error "$POSTER_NAME.tex" >/tmp/poster_pass1.log 2>&1 || {
    echo "[error] pass 1 failed; tail of log:" >&2
    tail -40 /tmp/poster_pass1.log >&2
    exit 1
}

echo "=== Compiling $POSTER_NAME.tex (pass 2/2) ==="
pdflatex -interaction=nonstopmode -halt-on-error "$POSTER_NAME.tex" >/tmp/poster_pass2.log 2>&1 || {
    echo "[error] pass 2 failed; tail of log:" >&2
    tail -40 /tmp/poster_pass2.log >&2
    exit 1
}

# Tidy up auxiliary files the PDF doesn't need
rm -f "$POSTER_NAME.aux" "$POSTER_NAME.log" "$POSTER_NAME.out" "$POSTER_NAME.toc" "$POSTER_NAME.nav" "$POSTER_NAME.snm"

echo ""
echo "=== Poster built: $POSTER_DIR/$POSTER_NAME.pdf ==="
ls -lh "$POSTER_NAME.pdf"
