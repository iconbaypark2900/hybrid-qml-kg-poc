"""Per-tenant usage accounting (quantum-seconds, jobs, est cost).

Single-process JSON file under artifacts/usage/<tenant_id>.json. When the
queue is replaced by Redis, swap to a hash there.

Cost is computed as quantum_seconds * IBM_PRICE_PER_SECOND_USD; the rate
constant is conservative ($0.0016/sec for ibm_torino base rate as of
2025) and should be overridden via env when partner contracts change it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from threading import Lock

from .schemas import JobCost, TenantUsage


IBM_PRICE_PER_SECOND_USD = float(os.environ.get("HETQML_IBM_PRICE_PER_SECOND_USD", "0.0016"))


class UsageStore:
    """Tenant usage counters with file-backed persistence."""

    def __init__(self, root: Path):
        self.root = root
        self._lock = Lock()

    def _path(self, tenant_id: str) -> Path:
        return self.root / "usage" / f"{tenant_id}.json"

    def _load(self, tenant_id: str) -> TenantUsage:
        p = self._path(tenant_id)
        if not p.exists():
            return TenantUsage(tenant_id=tenant_id, period_start=time.time())
        try:
            return TenantUsage.model_validate_json(p.read_text(encoding="utf-8"))
        except Exception:
            return TenantUsage(tenant_id=tenant_id, period_start=time.time())

    def _save(self, usage: TenantUsage) -> None:
        p = self._path(usage.tenant_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(usage.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(p)

    def record(self, tenant_id: str, cost: JobCost) -> TenantUsage:
        with self._lock:
            usage = self._load(tenant_id)
            usage = usage.model_copy(update={
                "quantum_seconds_used": usage.quantum_seconds_used + cost.quantum_seconds,
                "quantum_jobs_run": usage.quantum_jobs_run + 1,
                "estimated_cost_usd": usage.estimated_cost_usd
                    + (cost.estimated_cost_usd or 0.0),
            })
            self._save(usage)
            return usage

    def get(self, tenant_id: str) -> TenantUsage:
        with self._lock:
            return self._load(tenant_id)

    def reset(self, tenant_id: str) -> TenantUsage:
        with self._lock:
            usage = TenantUsage(tenant_id=tenant_id, period_start=time.time())
            self._save(usage)
            return usage


def estimate_cost(quantum_seconds: float) -> float:
    return round(quantum_seconds * IBM_PRICE_PER_SECOND_USD, 6)
