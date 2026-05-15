"""Per-tenant rate limiting via TokenBucket."""
from __future__ import annotations

import time

from service.ratelimit import RateLimiter, TokenBucket
from service.schemas import Tenant, TenantQuota


def _tenant(tenant_id: str, rpm: int) -> Tenant:
    return Tenant(
        tenant_id=tenant_id,
        name=tenant_id,
        api_key_sha256="x" * 64,
        quota=TenantQuota(
            requests_per_minute=rpm,
            requests_per_day=rpm * 60,
            can_use_quantum_strict=False,
            can_use_ibm_hardware=False,
            max_batch_size=10,
        ),
        created_at=time.time(),
    )


def test_token_bucket_initial_capacity():
    b = TokenBucket(capacity=5, refill_per_sec=0.1)
    for _ in range(5):
        assert b.take(1) is True
    assert b.take(1) is False


def test_token_bucket_refills_over_time(monkeypatch):
    b = TokenBucket(capacity=2, refill_per_sec=10.0)
    assert b.take(1) and b.take(1)
    assert b.take(1) is False
    # After 0.2s we should have 2 tokens back
    time.sleep(0.21)
    assert b.take(1) is True
    assert b.take(1) is True


def test_rate_limiter_per_tenant_independence():
    rl = RateLimiter()
    a = _tenant("a", rpm=2)
    b = _tenant("b", rpm=2)
    # Tenant A burns its budget
    assert rl.check(a) and rl.check(a)
    assert rl.check(a) is False
    # Tenant B's bucket is unaffected
    assert rl.check(b)
    assert rl.check(b)


def test_rate_limiter_zero_quota_always_denied():
    rl = RateLimiter()
    t = _tenant("zero", rpm=0)
    assert rl.check(t) is False


def test_rate_limiter_returns_retry_after_for_exhausted_bucket():
    rl = RateLimiter()
    t = _tenant("a", rpm=1)
    assert rl.check(t)  # capacity 1 used
    assert rl.check(t) is False
    retry = rl.retry_after(t)
    # 1 rpm = 1/60 tokens per sec, so retry_after ≈ 60s; allow some slack
    assert 0 < retry <= 60.0
