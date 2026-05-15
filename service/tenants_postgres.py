"""Postgres-backed tenant store for multi-worker deployments.

Schema:
    CREATE TABLE tenants (
        tenant_id           text PRIMARY KEY,
        name                text NOT NULL,
        api_key_sha256      text NOT NULL UNIQUE,
        previous_api_key_sha256 text,
        previous_key_expires_at double precision,
        quota               jsonb NOT NULL,
        created_at          double precision NOT NULL,
        is_system           boolean NOT NULL DEFAULT false,
        updated_at          double precision NOT NULL
    );
    CREATE INDEX tenants_prev_sha_idx ON tenants(previous_api_key_sha256)
        WHERE previous_api_key_sha256 IS NOT NULL;

Activation:
    HETQML_TENANTS_BACKEND=postgres
    HETQML_TENANTS_DSN=postgresql://user:pass@host/db

If either env is unset, the file-backed TenantStore (tenants.py) is used.

Migration: `python -m service.scripts.migrate_tenants_to_postgres` reads the
existing secrets/tenants.yaml and writes rows. Idempotent on tenant_id.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

from .schemas import Tenant, TenantQuota

log = logging.getLogger(__name__)


_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id text PRIMARY KEY,
    name text NOT NULL,
    api_key_sha256 text NOT NULL UNIQUE,
    previous_api_key_sha256 text,
    previous_key_expires_at double precision,
    quota jsonb NOT NULL,
    created_at double precision NOT NULL,
    is_system boolean NOT NULL DEFAULT false,
    updated_at double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS tenants_prev_sha_idx ON tenants(previous_api_key_sha256)
    WHERE previous_api_key_sha256 IS NOT NULL;
"""


def is_enabled() -> bool:
    return (
        os.environ.get("HETQML_TENANTS_BACKEND", "").lower() == "postgres"
        and bool(os.environ.get("HETQML_TENANTS_DSN"))
    )


class PostgresTenantStore:
    """Drop-in replacement for service.tenants.TenantStore.

    Lazy connection: opens psycopg pool on first call; degrades to file
    fallback if psycopg isn't installed.
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn or os.environ.get("HETQML_TENANTS_DSN")
        self._loaded = True
        self._pool = None
        self._init_pool()
        self._ensure_schema()

    def _init_pool(self) -> None:
        try:
            import psycopg_pool  # type: ignore[import-not-found]
        except ImportError:
            log.warning(
                "HETQML_TENANTS_BACKEND=postgres but psycopg_pool not installed; "
                "falling back to file-backed store. "
                "Install: pip install 'psycopg[binary,pool]>=3.1'"
            )
            self._loaded = False
            return
        if self.dsn is None:
            log.warning("HETQML_TENANTS_DSN not set; degrading to file-backed store")
            self._loaded = False
            return
        try:
            self._pool = psycopg_pool.ConnectionPool(
                self.dsn, min_size=1, max_size=5, kwargs={"autocommit": True},
            )
        except Exception as e:
            log.warning("postgres connection failed: %s; degrading to file fallback", e)
            self._loaded = False
            self._pool = None

    def _ensure_schema(self) -> None:
        if self._pool is None:
            return
        try:
            with self._pool.connection() as conn:
                conn.execute(_SCHEMA_DDL)
        except Exception as e:
            log.error("failed to apply tenants schema: %s", e)
            self._loaded = False

    async def find_by_key_sha(self, sha: str) -> Optional[Tenant]:
        if not self._loaded or self._pool is None:
            return None
        # Look up either by current or previous (non-expired) key.
        sql = """
            SELECT tenant_id, name, api_key_sha256, previous_api_key_sha256,
                   previous_key_expires_at, quota, created_at, is_system
              FROM tenants
             WHERE api_key_sha256 = %s
                OR (previous_api_key_sha256 = %s
                    AND (previous_key_expires_at IS NULL
                         OR previous_key_expires_at > %s))
             LIMIT 1
        """
        try:
            with self._pool.connection() as conn:
                row = conn.execute(sql, (sha, sha, time.time())).fetchone()
        except Exception as e:
            log.warning("postgres find_by_key_sha failed: %s", e)
            return None
        if row is None:
            return None
        (tid, name, sha_, prev_sha, prev_exp, quota_json, created_at, is_system) = row
        if is_system:
            return None
        quota_data = quota_json if isinstance(quota_json, dict) else json.loads(quota_json)
        return Tenant(
            tenant_id=tid,
            name=name,
            api_key_sha256=sha_,
            previous_api_key_sha256=prev_sha,
            previous_key_expires_at=prev_exp,
            quota=TenantQuota(**quota_data),
            created_at=created_at,
            is_system=is_system,
        )

    def get(self, tenant_id: str) -> Optional[Tenant]:
        if not self._loaded or self._pool is None:
            return None
        sql = """
            SELECT tenant_id, name, api_key_sha256, previous_api_key_sha256,
                   previous_key_expires_at, quota, created_at, is_system
              FROM tenants WHERE tenant_id = %s LIMIT 1
        """
        try:
            with self._pool.connection() as conn:
                row = conn.execute(sql, (tenant_id,)).fetchone()
        except Exception as e:
            log.warning("postgres get failed: %s", e)
            return None
        if row is None:
            return None
        (tid, name, sha_, prev_sha, prev_exp, quota_json, created_at, is_system) = row
        quota_data = quota_json if isinstance(quota_json, dict) else json.loads(quota_json)
        return Tenant(
            tenant_id=tid, name=name, api_key_sha256=sha_,
            previous_api_key_sha256=prev_sha, previous_key_expires_at=prev_exp,
            quota=TenantQuota(**quota_data),
            created_at=created_at, is_system=is_system,
        )

    def all_tenants(self) -> list[Tenant]:
        if not self._loaded or self._pool is None:
            return []
        sql = """
            SELECT tenant_id, name, api_key_sha256, previous_api_key_sha256,
                   previous_key_expires_at, quota, created_at, is_system
              FROM tenants ORDER BY tenant_id
        """
        out: list[Tenant] = []
        try:
            with self._pool.connection() as conn:
                rows = conn.execute(sql).fetchall()
        except Exception as e:
            log.warning("postgres all_tenants failed: %s", e)
            return []
        for row in rows:
            (tid, name, sha_, prev_sha, prev_exp, quota_json, created_at, is_system) = row
            quota_data = quota_json if isinstance(quota_json, dict) else json.loads(quota_json)
            out.append(Tenant(
                tenant_id=tid, name=name, api_key_sha256=sha_,
                previous_api_key_sha256=prev_sha, previous_key_expires_at=prev_exp,
                quota=TenantQuota(**quota_data),
                created_at=created_at, is_system=is_system,
            ))
        return out

    def upsert(self, tenant: Tenant) -> None:
        if not self._loaded or self._pool is None:
            raise RuntimeError("postgres store not available")
        sql = """
            INSERT INTO tenants (
                tenant_id, name, api_key_sha256,
                previous_api_key_sha256, previous_key_expires_at,
                quota, created_at, is_system, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id) DO UPDATE SET
                name = EXCLUDED.name,
                api_key_sha256 = EXCLUDED.api_key_sha256,
                previous_api_key_sha256 = EXCLUDED.previous_api_key_sha256,
                previous_key_expires_at = EXCLUDED.previous_key_expires_at,
                quota = EXCLUDED.quota,
                is_system = EXCLUDED.is_system,
                updated_at = EXCLUDED.updated_at
        """
        with self._pool.connection() as conn:
            conn.execute(sql, (
                tenant.tenant_id, tenant.name, tenant.api_key_sha256,
                tenant.previous_api_key_sha256, tenant.previous_key_expires_at,
                json.dumps(tenant.quota.model_dump()), tenant.created_at,
                tenant.is_system, time.time(),
            ))


def maybe_build_store():
    """Returns a PostgresTenantStore if env-enabled and reachable, else None."""
    if not is_enabled():
        return None
    store = PostgresTenantStore()
    if not store._loaded:
        return None
    return store
