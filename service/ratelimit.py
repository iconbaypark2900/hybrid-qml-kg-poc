"""Per-tenant token-bucket rate limiter.

Single-process, in-memory. For multi-worker deployments, replace with Redis.
"""
from __future__ import annotations

import time
from threading import Lock

from .schemas import Tenant


class TokenBucket:
    __slots__ = ("capacity", "refill", "tokens", "last", "_lock")

    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self.tokens = float(capacity)
        self.last = time.monotonic()
        self._lock = Lock()

    def take(self, n: float = 1.0) -> bool:
        with self._lock:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill)
            self.last = now
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False

    def retry_after_seconds(self, n: float = 1.0) -> float:
        """Return seconds until n tokens are available, or 0 if available now."""
        with self._lock:
            if self.tokens >= n:
                return 0.0
            if self.refill <= 0:
                return float("inf")
            return (n - self.tokens) / self.refill


class RateLimiter:
    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], TokenBucket] = {}
        self._lock = Lock()

    def _bucket_for(self, tenant: Tenant, scope: str) -> TokenBucket:
        key = (tenant.tenant_id, scope)
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                # capacity = full minute budget; refill = budget/60s
                capacity = max(tenant.quota.requests_per_minute, 1)
                bucket = TokenBucket(
                    capacity=capacity,
                    refill_per_sec=tenant.quota.requests_per_minute / 60.0,
                )
                self._buckets[key] = bucket
            return bucket

    def check(self, tenant: Tenant, scope: str = "default") -> bool:
        if tenant.is_system or tenant.quota.requests_per_minute <= 0:
            # System tenants and zero-quota tenants are always denied.
            # is_system tenants should never reach this code path through
            # normal auth, but defense in depth.
            return False
        return self._bucket_for(tenant, scope).take(1.0)

    def retry_after(self, tenant: Tenant, scope: str = "default") -> float:
        if tenant.is_system or tenant.quota.requests_per_minute <= 0:
            return float("inf")
        return self._bucket_for(tenant, scope).retry_after_seconds(1.0)
