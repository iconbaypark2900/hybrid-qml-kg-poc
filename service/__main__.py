"""Boot the service via `python -m service`.

Uses uvicorn programmatically against `service.app:create_app` (factory mode),
so the lifespan runs the same way it does under the smoke harness — but bound
to a real socket.

Examples:
    python -m service                                     # 0.0.0.0:8000
    python -m service --port 8080
    python -m service --reload                            # dev autoreload
    HETQML_LOG_FORMAT=text python -m service --log-level debug

Multi-worker note: because the in-memory job queue (service.jobs.JobQueue)
isn't shared across worker processes, `--workers` is fixed at 1 in v1.
For multi-worker deployments, replace JobQueue with a Redis/RQ-backed queue.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m service", description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="autoreload on code changes (dev only)")
    p.add_argument("--log-level", default="info",
                   choices=["critical", "error", "warning", "info", "debug", "trace"])
    p.add_argument("--root-path", default="",
                   help="Mount under a path prefix (when behind a reverse proxy)")
    args = p.parse_args(argv)

    import uvicorn

    uvicorn.run(
        "service.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        root_path=args.root_path,
        workers=1,  # see module docstring
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
