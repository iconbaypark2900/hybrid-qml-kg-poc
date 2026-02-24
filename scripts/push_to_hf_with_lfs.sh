#!/bin/bash
# Push this repo to Hugging Face Spaces using Git LFS for binary files.
# Prerequisite: Install Git LFS once (requires sudo):
#   Ubuntu/WSL: sudo apt-get install git-lfs
#   macOS:      brew install git-lfs
#
# Then run from project root:  bash scripts/push_to_hf_with_lfs.sh

set -e
cd "$(dirname "$0")/.."

if ! command -v git-lfs &>/dev/null; then
  echo "Git LFS is not installed. Install it first:"
  echo "  Ubuntu/WSL: sudo apt-get install git-lfs"
  echo "  macOS:      brew install git-lfs"
  exit 1
fi

git lfs install

# Commit .gitattributes if present and not yet committed
if [[ -f .gitattributes ]]; then
  git add .gitattributes
  git diff --cached --quiet || git commit -m "Track *.png and *.pdf with Git LFS for HF Spaces" || true
fi

# Rewrite history so existing binaries become LFS pointers
echo "Migrating existing *.png and *.pdf to LFS (rewriting history)..."
git lfs migrate import --include="*.png,*.pdf" --everything

# Push to Hugging Face (remote must be named 'hf')
BRANCH=$(git branch --show-current)
echo "Pushing $BRANCH to hf main..."
git push hf "$BRANCH:main" --force

echo "Done. Space URL: https://huggingface.co/spaces/rocRevyAreGoals15/QGG-HYBRID-PROJECT"
