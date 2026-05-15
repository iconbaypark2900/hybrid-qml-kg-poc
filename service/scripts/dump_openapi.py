"""Dump the live OpenAPI 3 spec to stdout (or a file).

Usage:
    python -m service.scripts.dump_openapi              # → stdout
    python -m service.scripts.dump_openapi --out openapi.json
    python -m service.scripts.dump_openapi --pretty

Used to keep the generated frontend types in sync with the backend schemas.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from service.app import create_app  # noqa: E402
from service.settings import Settings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None,
                   help="Write to file instead of stdout")
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = p.parse_args(argv)

    # Build the app without entering lifespan — openapi() only needs routes
    app = create_app(settings=Settings())
    spec = app.openapi()

    blob = json.dumps(spec, indent=2 if args.pretty else None, sort_keys=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(blob, encoding="utf-8")
        print(f"wrote {len(blob)} bytes to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(blob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
