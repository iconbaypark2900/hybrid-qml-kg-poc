"""FastAPI dependency providers."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from .jobs import JobQueue
from .observability import StartupTracker
from .orchestration import Orchestrator
from .ratelimit import RateLimiter
from .schemas import ErrorCode, ErrorResponse, Tenant
from .settings import Settings
from .tenants import TenantStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_tracker(request: Request) -> StartupTracker:
    return request.app.state.tracker


def get_tenant_store(request: Request) -> TenantStore:
    return request.app.state.tenant_store


def get_rate_limiter(request: Request) -> RateLimiter:
    return request.app.state.rate_limiter


def get_job_queue(request: Request) -> JobQueue:
    return request.app.state.job_queue


def require_ready(tracker: StartupTracker = Depends(get_tracker)) -> StartupTracker:
    """Used by /predict, /jobs — never on /status."""
    if not tracker.is_ready():
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                code=ErrorCode.SERVICE_NOT_READY,
                message=f"loading: {tracker.current_step}",
                detail={"step": tracker.current_step,
                        "loaded": tracker.loaded,
                        "total": tracker.total},
            ).model_dump(),
        )
    return tracker


def get_orchestrator(
    request: Request, _ready: StartupTracker = Depends(require_ready)
) -> Orchestrator:
    return request.app.state.orchestrator


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def get_tenant(request: Request) -> Tenant:
    """Pulled out of request.state by AuthMiddleware. Public routes don't
    have AuthMiddleware run on them — those routes don't depend on this."""
    tenant = getattr(request.state, "tenant", None)
    if tenant is None:
        raise HTTPException(
            status_code=401,
            detail=ErrorResponse(
                code=ErrorCode.AUTH_MISSING, message="tenant context missing"
            ).model_dump(),
        )
    return tenant


def enforce_rate_limit(
    tenant: Tenant = Depends(get_tenant),
    rl: RateLimiter = Depends(get_rate_limiter),
) -> None:
    if not rl.check(tenant):
        retry = rl.retry_after(tenant)
        raise HTTPException(
            status_code=429,
            headers={"Retry-After": str(int(retry) + 1)},
            detail=ErrorResponse(
                code=ErrorCode.RATE_LIMITED,
                message=f"rate limit exceeded for tenant {tenant.tenant_id}",
                detail={"retry_after_seconds": retry},
            ).model_dump(),
        )
