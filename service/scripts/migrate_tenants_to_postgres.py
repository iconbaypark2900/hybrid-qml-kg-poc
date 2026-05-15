"""One-shot migration: secrets/tenants.yaml -> Postgres `tenants` table.

Usage:
    HETQML_TENANTS_DSN=postgresql://user:pass@host/db \\
        python -m service.scripts.migrate_tenants_to_postgres \\
            --tenants-path secrets/tenants.yaml

Idempotent on tenant_id: re-running upserts.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from service.tenants import TenantStore  # noqa: E402
from service.tenants_postgres import PostgresTenantStore  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tenants-path", required=True, type=Path)
    p.add_argument("--dsn", default=os.environ.get("HETQML_TENANTS_DSN"))
    args = p.parse_args()

    if not args.dsn:
        print("HETQML_TENANTS_DSN env var or --dsn flag required", file=sys.stderr)
        return 1

    file_store = TenantStore.from_yaml(args.tenants_path)
    pg = PostgresTenantStore(dsn=args.dsn)
    if not pg._loaded:
        print("postgres connection failed; check DSN and psycopg install", file=sys.stderr)
        return 1

    written = 0
    for t in file_store.all_tenants():
        if t.is_system:
            continue
        pg.upsert(t)
        written += 1
        print(f"upserted tenant_id={t.tenant_id}")
    print(f"\nmigration complete. {written} tenants written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
