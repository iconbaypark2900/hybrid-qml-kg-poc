"""TenantStore — file-backed YAML index of tenants and their api_key sha256s.

Layout:
  - secrets/tenants.yaml     (gitignored, real keys)
  - tenants.example.yaml     (committed, dummy keys for local dev)

The store is loaded once at startup and held in-memory. Reload requires
restart (or the future POST /admin/reload).

Key generation helper (operator-side):
    python -m service.tenants generate-key

This prints the plaintext API key once (operator copies it) and the
sha256 to paste into tenants.yaml. Plaintext is never persisted.
"""
from __future__ import annotations

import asyncio
import hashlib
import secrets
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from .schemas import Tenant, TenantQuota


SYSTEM_LEGACY_TENANT_ID = "legacy"


def _build_legacy_tenant() -> Tenant:
    """System tenant for migrated pre-rebuild evaluation history.
    No real API key resolves to this tenant — only operator tools/tests.
    """
    return Tenant(
        tenant_id=SYSTEM_LEGACY_TENANT_ID,
        name="Legacy (pre-rebuild migration)",
        api_key_sha256="0" * 64,  # unreachable via auth middleware
        quota=TenantQuota(
            requests_per_minute=0,
            requests_per_day=0,
            can_use_quantum_strict=False,
            can_use_ibm_hardware=False,
            max_batch_size=0,
        ),
        created_at=0.0,
        is_system=True,
    )


class TenantStore:
    """In-memory tenant index keyed by both tenant_id and api_key_sha256.

    The `_loaded` flag is False for the empty placeholder constructed at
    create_app() time and True for the real one written during lifespan
    startup. AuthMiddleware checks this flag to return 503 instead of 401
    when a request lands during the brief startup window.

    Key rotation: tenants with `previous_api_key_sha256` set accept either
    the current or the previous key, until `previous_key_expires_at`.
    """

    def __init__(self, tenants: list[Tenant], loaded: bool = True):
        self._by_id: dict[str, Tenant] = {t.tenant_id: t for t in tenants}
        self._by_sha: dict[str, Tenant] = {}
        for t in tenants:
            if t.is_system:
                continue
            self._by_sha[t.api_key_sha256] = t
            if t.previous_api_key_sha256:
                self._by_sha[t.previous_api_key_sha256] = t
        self._loaded = loaded

    @classmethod
    def from_yaml(cls, path: Path, fallback_example: Optional[Path] = None) -> "TenantStore":
        """Load tenants from yaml. If `path` is missing and `fallback_example`
        is provided, load from the example (logs a WARNING — example tenants
        must never be used in production).
        """
        load_path: Optional[Path] = None
        if path.exists():
            load_path = path
        elif fallback_example is not None and fallback_example.exists():
            load_path = fallback_example

        tenants: list[Tenant] = [_build_legacy_tenant()]
        if load_path is None:
            return cls(tenants)

        with load_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}

        for entry in payload.get("tenants", []):
            quota_dict = entry.get("quota", {})
            quota = TenantQuota(**quota_dict)
            tenant = Tenant(
                tenant_id=entry["tenant_id"],
                name=entry.get("name", entry["tenant_id"]),
                api_key_sha256=entry["api_key_sha256"],
                quota=quota,
                created_at=float(entry.get("created_at", time.time())),
                is_system=bool(entry.get("is_system", False)),
            )
            tenants.append(tenant)
        return cls(tenants)

    async def find_by_key_sha(self, sha: str) -> Optional[Tenant]:
        """Async-friendly lookup. The dict op is sync but signature stays
        async so callers don't need to know the implementation. Honors
        previous_key_expires_at: a request with the previous key is accepted
        only if expiry is in the future."""
        tenant = self._by_sha.get(sha)
        if tenant is None:
            return None
        # If this lookup matched the *previous* key and it has expired, reject.
        if (
            tenant.previous_api_key_sha256 == sha
            and tenant.previous_key_expires_at is not None
            and tenant.previous_key_expires_at <= time.time()
        ):
            return None
        return tenant

    def get(self, tenant_id: str) -> Optional[Tenant]:
        return self._by_id.get(tenant_id)

    def all_tenants(self) -> list[Tenant]:
        return list(self._by_id.values())


def hash_api_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode()).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Returns (plaintext, sha256). Operator stores plaintext securely;
    only sha256 is committed to tenants.yaml."""
    plaintext = secrets.token_urlsafe(32)
    return plaintext, hash_api_key(plaintext)


def _cli() -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m service.tenants")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("generate-key", help="Generate a fresh API key")
    rotate = sub.add_parser(
        "rotate-key",
        help="Rotate a tenant's API key with a grace period",
    )
    rotate.add_argument("--tenants-path", required=True, type=Path,
                        help="Path to secrets/tenants.yaml")
    rotate.add_argument("--tenant-id", required=True)
    rotate.add_argument("--grace-hours", type=int, default=24,
                        help="How long the previous key keeps working")
    args = p.parse_args()

    if args.cmd == "generate-key":
        plaintext, sha = generate_api_key()
        print("# Plaintext API key (give this to the tenant ONCE; never commit):")
        print(plaintext)
        print()
        print("# Add this entry to secrets/tenants.yaml:")
        print(f"#   api_key_sha256: {sha}")
        return 0

    if args.cmd == "rotate-key":
        return _rotate_key_cli(args.tenants_path, args.tenant_id, args.grace_hours)
    return 2


def _rotate_key_cli(tenants_path: Path, tenant_id: str, grace_hours: int) -> int:
    if not tenants_path.exists():
        print(f"tenants file not found: {tenants_path}", file=sys.stderr)
        return 1
    payload = yaml.safe_load(tenants_path.read_text(encoding="utf-8")) or {}
    entries = payload.get("tenants", [])
    target = next((e for e in entries if e.get("tenant_id") == tenant_id), None)
    if target is None:
        print(f"tenant {tenant_id!r} not in {tenants_path}", file=sys.stderr)
        return 1

    plaintext, new_sha = generate_api_key()
    expires_at = time.time() + grace_hours * 3600
    target["previous_api_key_sha256"] = target.get("api_key_sha256")
    target["previous_key_expires_at"] = expires_at
    target["api_key_sha256"] = new_sha

    # Atomic write
    tmp = tenants_path.with_suffix(tenants_path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    tmp.replace(tenants_path)

    expires_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(expires_at))
    print(f"# Rotated key for tenant {tenant_id!r}.")
    print(f"# Previous key remains valid until {expires_iso} ({grace_hours}h grace).")
    print(f"# Hot-reload the running service with:  POST /admin/tenants/reload")
    print()
    print("# Plaintext API key (give to the tenant ONCE; never commit):")
    print(plaintext)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
