#!/usr/bin/env bash
# prepare_osf_bundle.sh — Build the OSF preprint upload bundle.
#
# Collects everything an external reviewer needs to reproduce the headline
# result + browse the supplementary material:
#   - artifacts/predictions/*  (sealed test metrics, top candidates, comparisons)
#   - artifacts/signatures/    (signature catalog if present)
#   - artifacts/single_cell/   (QC summary if present)
#   - docs/poster/poster.pdf
#   - docs/tables/*.tex
#   - docs/paper_alignment/*.md
#   - docs/RELEASE_NOTES_v1.0.0-preprint.md
#   - CHANGELOG.md
#   - preregistration/*
#
# Output: osf_bundle_<version>.tar.gz + osf_bundle_<version>.SHA256
#
# Usage:
#   bash scripts/prepare_osf_bundle.sh
#   bash scripts/prepare_osf_bundle.sh --version v1.0.0-preprint
#   bash scripts/prepare_osf_bundle.sh --rebuild   # regenerate artifacts first
set -euo pipefail

VERSION="v1.0.0-preprint"
REBUILD=0

for arg in "$@"; do
    case $arg in
        --version) shift; VERSION="$1"; shift ;;
        --version=*) VERSION="${arg#*=}" ;;
        --rebuild) REBUILD=1 ;;
        -h|--help)
            sed -n '2,20p' "$0"
            exit 0
            ;;
    esac
done

echo "=== Building OSF bundle: $VERSION ==="

if [ "$REBUILD" -eq 1 ]; then
    echo ""
    echo "=== Regenerating artifacts before bundling ==="
    python3 scripts/run_full_repurposing_pipeline.py --mode kg+omics --top-n 50
    python3 scripts/compare_pipeline_modes.py --top-n 20 --no-run
    python3 scripts/aggregate_qc_summary.py || true
    python3 scripts/build_signature_catalog.py || true
    python3 scripts/build_paper_tables.py --all || true
    if command -v pdflatex >/dev/null 2>&1; then
        bash scripts/build_poster.sh || echo "[warn] poster build failed; continuing"
    fi
fi

STAGING="osf_bundle_${VERSION}"
OUTPUT="osf_bundle_${VERSION}.tar.gz"
MANIFEST="osf_bundle_${VERSION}.SHA256"

# Clean any prior staging directory
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Copy reproducibility-critical paths only — no caches, no .pnpm-store, no IDE files
copy_if_exists() {
    local src="$1"
    if [ -e "$src" ]; then
        echo "  [+] $src"
        if [ -d "$src" ]; then
            mkdir -p "$STAGING/$(dirname "$src")"
            cp -r "$src" "$STAGING/$(dirname "$src")/"
        else
            mkdir -p "$STAGING/$(dirname "$src")"
            cp "$src" "$STAGING/$src"
        fi
    else
        echo "  [-] $src (skipped, not present)"
    fi
}

echo ""
echo "=== Collecting bundle contents ==="
copy_if_exists "CHANGELOG.md"
copy_if_exists "README.md"
copy_if_exists "LICENSE"
copy_if_exists "docs/RELEASE_NOTES_v1.0.0-preprint.md"
copy_if_exists "docs/poster/poster.pdf"
copy_if_exists "docs/poster/poster.tex"
copy_if_exists "docs/poster/README.md"
copy_if_exists "docs/paper_alignment"
copy_if_exists "docs/tables"
copy_if_exists "docs/results"
copy_if_exists "docs/deployment/DGX_RUNBOOK.md"
copy_if_exists "preregistration"
copy_if_exists "charter"
copy_if_exists "artifacts/predictions/sealed_test_metrics.json"
copy_if_exists "artifacts/predictions/top_candidates.csv"
copy_if_exists "artifacts/predictions/top_candidates.json"
copy_if_exists "artifacts/predictions/run_summary.json"
copy_if_exists "artifacts/predictions/final_repurposing_report.md"
copy_if_exists "artifacts/predictions/mode_comparison.csv"
copy_if_exists "artifacts/predictions/mode_comparison.md"
copy_if_exists "artifacts/predictions/bootstrap_ci.json"
copy_if_exists "artifacts/data/hetionet_stats.json"
copy_if_exists "artifacts/signatures/signature_catalog.csv"
copy_if_exists "artifacts/signatures/signature_catalog.md"
copy_if_exists "artifacts/single_cell/qc/qc_summary_table.csv"
copy_if_exists "artifacts/single_cell/qc/qc_summary_table.md"

echo ""
echo "=== Computing SHA256 manifest ==="
( cd "$STAGING" && find . -type f | LC_ALL=C sort | xargs sha256sum ) > "$MANIFEST"
wc -l "$MANIFEST"

echo ""
echo "=== Creating tarball ==="
tar -czf "$OUTPUT" "$STAGING"
ls -lh "$OUTPUT" "$MANIFEST"

echo ""
echo "=== OSF bundle ready ==="
echo "  Tarball:  $OUTPUT"
echo "  Manifest: $MANIFEST"
echo ""
echo "Next steps:"
echo "  1. Upload $OUTPUT to OSF as the primary artifact"
echo "  2. Upload $MANIFEST as the SHA256 verification file"
echo "  3. Tag the git release: git tag $VERSION && git push --tags"
echo "  4. Submit OSF preprint with link to the GitHub release"
