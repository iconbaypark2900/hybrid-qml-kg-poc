"""
SQLite-backed storage for tenant-scoped external integrations.

IBM Quantum credentials are stored per tenant and never returned through the
API. Set INTEGRATION_ENCRYPTION_KEY to a Fernet key in shared environments.
Without it, secrets are base64-encoded for local/dev only.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "results" / "research_state.db"
DB_PATH = Path(
    os.environ.get(
        "INTEGRATIONS_DB_PATH",
        os.environ.get("RESEARCH_DB_PATH", str(DEFAULT_DB_PATH)),
    )
).expanduser()
IBM_PROVIDER = "ibm_quantum"
DEFAULT_CHANNEL = "ibm_quantum_platform"


class IntegrationStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tenant_integration_secrets (
                    tenant_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    token_secret TEXT NOT NULL,
                    token_storage TEXT NOT NULL,
                    token_preview TEXT NOT NULL,
                    token_sha256 TEXT NOT NULL,
                    instance_crn TEXT,
                    channel TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_verified_at REAL,
                    PRIMARY KEY (tenant_id, provider)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tenant_integration_provider
                ON tenant_integration_secrets(provider, updated_at DESC)
                """
            )
            conn.commit()

    def save_ibm_quantum_credentials(
        self,
        *,
        tenant_id: str,
        token: str,
        instance_crn: Optional[str] = None,
        channel: str = DEFAULT_CHANNEL,
    ) -> Dict[str, Any]:
        cleaned_tenant = _clean_tenant_id(tenant_id)
        cleaned_token = token.strip()
        if not cleaned_token:
            raise ValueError("IBM Quantum token is required.")

        cleaned_instance = _clean_optional(instance_crn)
        cleaned_channel = (channel or DEFAULT_CHANNEL).strip() or DEFAULT_CHANNEL
        secret, storage = _encode_secret(cleaned_token)
        ts = time.time()

        with self._lock, self._connect() as conn:
            existing = conn.execute(
                """
                SELECT created_at FROM tenant_integration_secrets
                WHERE tenant_id = ? AND provider = ?
                """,
                (cleaned_tenant, IBM_PROVIDER),
            ).fetchone()
            conn.execute(
                """
                INSERT OR REPLACE INTO tenant_integration_secrets
                (tenant_id, provider, token_secret, token_storage, token_preview,
                 token_sha256, instance_crn, channel, created_at, updated_at, last_verified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cleaned_tenant,
                    IBM_PROVIDER,
                    secret,
                    storage,
                    _token_preview(cleaned_token),
                    hashlib.sha256(cleaned_token.encode("utf-8")).hexdigest(),
                    cleaned_instance,
                    cleaned_channel,
                    existing["created_at"] if existing else ts,
                    ts,
                    None,
                ),
            )
            conn.commit()

        metadata = self.get_ibm_quantum_metadata(cleaned_tenant)
        if metadata is None:
            raise RuntimeError("Saved credentials could not be read back.")
        return metadata

    def get_ibm_quantum_metadata(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        row = self._get_row(tenant_id, IBM_PROVIDER)
        return _row_to_metadata(row) if row else None

    def get_ibm_quantum_credentials(self, tenant_id: str) -> Optional[Dict[str, Optional[str]]]:
        row = self._get_row(tenant_id, IBM_PROVIDER)
        if row is None:
            return None
        return {
            "token": _decode_secret(row["token_secret"], row["token_storage"]),
            "instance_crn": row["instance_crn"],
            "channel": row["channel"],
        }

    def mark_ibm_quantum_verified(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        cleaned_tenant = _clean_tenant_id(tenant_id)
        ts = time.time()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE tenant_integration_secrets
                SET last_verified_at = ?, updated_at = ?
                WHERE tenant_id = ? AND provider = ?
                """,
                (ts, ts, cleaned_tenant, IBM_PROVIDER),
            )
            conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get_ibm_quantum_metadata(cleaned_tenant)

    def _get_row(self, tenant_id: str, provider: str) -> Optional[sqlite3.Row]:
        cleaned_tenant = _clean_tenant_id(tenant_id)
        with self._lock, self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM tenant_integration_secrets
                WHERE tenant_id = ? AND provider = ?
                """,
                (cleaned_tenant, provider),
            ).fetchone()


def _row_to_metadata(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "configured": True,
        "tenant_id": row["tenant_id"],
        "provider": row["provider"],
        "instance_crn": row["instance_crn"],
        "channel": row["channel"],
        "token_preview": row["token_preview"],
        "secret_storage": row["token_storage"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_verified_at": row["last_verified_at"],
    }


def _encode_secret(value: str) -> tuple[str, str]:
    fernet_key = os.environ.get("INTEGRATION_ENCRYPTION_KEY", "").strip()
    if fernet_key:
        from cryptography.fernet import Fernet

        return Fernet(fernet_key.encode("utf-8")).encrypt(value.encode("utf-8")).decode(
            "utf-8"
        ), "fernet"
    return base64.b64encode(value.encode("utf-8")).decode("ascii"), "base64-dev"


def _decode_secret(secret: str, storage: str) -> str:
    if storage == "fernet":
        from cryptography.fernet import Fernet

        key = os.environ.get("INTEGRATION_ENCRYPTION_KEY", "").strip()
        if not key:
            raise RuntimeError("INTEGRATION_ENCRYPTION_KEY is required to decrypt this secret.")
        return Fernet(key.encode("utf-8")).decrypt(secret.encode("utf-8")).decode("utf-8")
    if storage == "base64-dev":
        return base64.b64decode(secret.encode("ascii")).decode("utf-8")
    raise RuntimeError(f"Unsupported secret storage type: {storage}")


def _token_preview(token: str) -> str:
    if len(token) <= 8:
        return "[REDACTED]"
    return f"{token[:3]}...{token[-3:]}"


def _clean_optional(value: Optional[str]) -> Optional[str]:
    cleaned = (value or "").strip()
    return cleaned or None


def _clean_tenant_id(value: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError("Tenant ID is required.")
    if len(cleaned) > 128:
        raise ValueError("Tenant ID must be 128 characters or fewer.")
    return cleaned


integration_store = IntegrationStore(DB_PATH)
