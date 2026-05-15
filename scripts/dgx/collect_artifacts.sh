#!/usr/bin/env bash
# collect_artifacts.sh — Bundle artifacts/ with SHA256 manifest.
set -euo pipefail

BUNDLE="artifacts_$(date +%Y%m%d_%H%M%S).tar.gz"
MANIFEST="artifacts/MANIFEST_SHA256.txt"

echo "Computing SHA256 checksums …"
find artifacts/ -type f | sort | xargs sha256sum > "$MANIFEST"

echo "Bundling artifacts/ → $BUNDLE"
tar -czf "$BUNDLE" artifacts/

echo "Bundle written: $BUNDLE"
echo "Manifest: $MANIFEST"
