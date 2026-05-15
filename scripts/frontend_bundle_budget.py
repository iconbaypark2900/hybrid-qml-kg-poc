#!/usr/bin/env python3
"""Lightweight Next.js bundle budget checks for the v2 dashboard.

Run after `frontend/next build`.
By default this reports warnings and exits 0. Set BUNDLE_BUDGET_STRICT=1 to fail
when a hard budget is exceeded.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
NEXT_DIR = FRONTEND / ".next"
STATIC_DIR = NEXT_DIR / "static"
STRICT = os.environ.get("BUNDLE_BUDGET_STRICT", "0").strip() in {"1", "true", "yes"}

GLOBAL_JS_BUDGET = int(os.environ.get("FRONTEND_GLOBAL_JS_BUDGET", "12000000"))
HEAVY_IMPORTS = ("3dmol", "three", "chart.js", "d3")
ALLOWED_HEAVY_IMPORT_FILES = {
    "frontend/components/v2/molecule-viewer.tsx",
    "frontend/components/kg-graph.tsx",
    "frontend/components/v2/kg-evidence-graph.tsx",
    "frontend/components/quantum-circuit.tsx",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def static_js_size() -> int:
    if not STATIC_DIR.exists():
        return 0
    return sum(path.stat().st_size for path in STATIC_DIR.rglob("*.js"))


def heavy_import_findings() -> list[str]:
    findings: list[str] = []
    for path in (FRONTEND).rglob("*.tsx"):
        rel = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        for marker in HEAVY_IMPORTS:
            if f"from \"{marker}\"" in text or f"import(\"{marker}\")" in text:
                if rel not in ALLOWED_HEAVY_IMPORT_FILES:
                    findings.append(f"{rel} imports heavy library {marker}")
    return findings


def main() -> int:
    failures: list[str] = []
    size = static_js_size()
    print("bundle_budget: global static JS size")
    print(f"  static JS: {size} bytes / budget {GLOBAL_JS_BUDGET}")
    if size > GLOBAL_JS_BUDGET:
        failures.append(f"global static JS exceeds budget: {size} > {GLOBAL_JS_BUDGET}")

    for finding in heavy_import_findings():
        failures.append(finding)

    if failures:
        print("bundle_budget: findings")
        for failure in failures:
            print(f"  - {failure}")
        return 1 if STRICT else 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
