"""Assemble a clean hetqml-next-original/ folder by combining:
  - Files recovered from JSONL transcripts (./recovered_originals/)
  - Files I never edited, copied from the current on-disk hetqml-next/

For globals.css specifically: the transcript Read was partial (only 49 lines
out of 522). I never appended permanently to the file — my appended block
was stripped out during the revert. So the current on-disk globals.css IS
the original; it gets copied verbatim.

Output: C:\\Users\\Jon B\\Downloads\\hetqml-next-original\\
Plus a MANIFEST.txt with per-file provenance (recovered vs on-disk).

Files I created in this session (ActiveModelBadge, ErrorBoundary, etc.) are
NOT included — they weren't in your upload.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CURRENT = Path("/mnt/c/Users/Jon B/Downloads/hetqml-next")
RECOVERED = ROOT / "output" / "recovered_originals"
TARGET = Path("/mnt/c/Users/Jon B/Downloads/hetqml-next-original")

# Files I never touched — copy from current on-disk (Apr 30 timestamps confirm origin).
ON_DISK_UNTOUCHED = [
    "components/shared/HelpHint.tsx",
    "components/shared/PageStub.tsx",
    "components/initialize/CandidateContext.tsx",
    "components/initialize/LiveKgPreview.tsx",
    "app/page.tsx",
    "app/initialize/page.tsx",
    "app/globals.css",  # current on-disk = original (my appended block was reverted)
    "next.config.mjs",
    "tsconfig.json",
    ".eslintrc.json",
    ".gitignore",
]

# Files I created — explicitly EXCLUDED. (Listed for completeness in the manifest.)
EXCLUDED_NEW_FILES = [
    "lib/api.ts",
    "lib/use-manifest.ts",
    "lib/use-evaluations.ts",
    "lib/use-live-algorithms.ts",
    "components/providers/QueryProvider.tsx",
    "components/shared/ActiveModelBadge.tsx",
    "components/shared/ErrorBoundary.tsx",
    "components/shared/OnboardingChecklist.tsx",
    "components/shared/Skeleton.tsx",
    "components/settings/SettingsPage.tsx",
    "components/experiment/ExperimentPage.tsx",
    "components/experiment/MetricStrip.tsx",
    "components/experiment/ModelLeaderboard.tsx",
    "components/experiment/DetailedMetrics.tsx",
    "components/validate/ValidatePage.tsx",
    "components/visualize/VisualizePage.tsx",
    "components/visualize/KGForceGraph.tsx",
    "components/visualize/MoleculeViewer.tsx",
    "components/operations/OperationsPage.tsx",
    "playwright.config.ts",
    "tests/e2e/smoke.spec.ts",
    ".claude/launch.json",
]


def main() -> int:
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)

    manifest_lines = [
        "hetqml-next-original — provenance manifest",
        "=" * 60,
        f"Assembled at: {Path.cwd()}",
        "",
        "## Files recovered from Claude Code transcripts",
        "",
    ]

    # 1. Copy from recovered_originals (everything except globals.css which we'll
    #    take from on-disk because the transcript Read was partial).
    n_recovered = 0
    for src in sorted(RECOVERED.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(RECOVERED).as_posix()
        if rel == "MANIFEST.txt":
            continue
        if rel == "app/globals.css":
            continue  # use on-disk version instead (full file)
        target = TARGET / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest_lines.append(f"  RECOVERED  {rel}  ({target.stat().st_size} bytes)")
        n_recovered += 1

    # 2. Copy untouched files from current on-disk
    manifest_lines += ["", "## Files copied from on-disk (untouched by me)", ""]
    n_disk = 0
    for rel in ON_DISK_UNTOUCHED:
        src = CURRENT / rel
        if not src.exists():
            manifest_lines.append(f"  MISSING    {rel}  (not on disk!)")
            continue
        target = TARGET / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest_lines.append(f"  ON-DISK    {rel}  ({target.stat().st_size} bytes)")
        n_disk += 1

    # 3. Note exclusions
    manifest_lines += ["", "## Files I created in this session — EXCLUDED", ""]
    for rel in EXCLUDED_NEW_FILES:
        present = (CURRENT / rel).exists()
        manifest_lines.append(f"  EXCLUDED   {rel}  (still in working folder: {present})")

    # 4. Make hooks/ and public/ dirs (originally empty per README)
    (TARGET / "hooks").mkdir(exist_ok=True)
    (TARGET / "public").mkdir(exist_ok=True)
    manifest_lines += ["", "## Empty directories preserved", "  hooks/", "  public/"]

    # 5. Save manifest
    (TARGET / "_RECOVERY_MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n",
                                                     encoding="utf-8")

    print(f"Recovered {n_recovered} files from transcripts")
    print(f"Copied {n_disk} files from on-disk")
    print(f"Total in target: {sum(1 for _ in TARGET.rglob('*') if _.is_file())}")
    print(f"Target: {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
