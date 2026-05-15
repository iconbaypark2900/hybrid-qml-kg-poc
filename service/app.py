"""FastAPI app factory with lifespan-managed eager startup.

Startup order:
    1. settings / logging / tracker
    2. async pools (thread + process)
    3. tenant store
    4. rate limiter
    5. manifest chain (LATEST.txt → DAG)
    6. orchestrator (entity resolver, feature_prep, predictors, quantum executor)
    7. job queue + workers
    8. mark_ready

If any step fails, the tracker is marked failed but the app still serves /status
honestly. Routes that depend on require_ready return 503 with a structured
ErrorResponse pointing at the failing step.

Tests override the orchestrator / quantum executor via app.dependency_overrides
or by passing `state_overrides` to create_app — see tests/conftest.py.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .async_runtime import init_pools, shutdown_pools
from .jobs import JobQueue
from .middleware import (
    AuthMiddleware,
    CorrelationIdMiddleware,
    StructuredLoggingMiddleware,
)
from . import tracing
from .observability import StartupTracker, setup_logging
from .persistence import load_active_manifest_chain, load_active_quantum_chain
from .ratelimit import RateLimiter
from .routes import ALL_ROUTERS
from .settings import Settings
from .tenants import TenantStore

log = logging.getLogger(__name__)

LIFESPAN_STEPS = (
    "logging", "async_pools", "tenant_store", "rate_limiter",
    "manifest_chain", "orchestrator", "job_workers", "ready",
)


StateBuilder = Callable[[FastAPI, Settings, StartupTracker], Awaitable[None]]


@asynccontextmanager
async def _default_lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    tracker: StartupTracker = app.state.tracker
    state_builder: Optional[StateBuilder] = getattr(app.state, "_state_builder", None)

    try:
        # 1. logging — already done by create_app(); count as a step for visibility
        tracker.step("logging")

        # 2. async pools
        init_pools(
            thread_workers=settings.thread_pool_workers,
            process_workers=settings.process_pool_workers,
        )
        tracker.step("async_pools")

        # 3. tenant store — Postgres if env-configured, else file-backed
        from .tenants_postgres import maybe_build_store as _pg
        pg_store = _pg()
        if pg_store is not None:
            app.state.tenant_store = pg_store
            log.info("tenant_store: postgres backend (%d tenants)",
                     len(pg_store.all_tenants()))
        else:
            store = TenantStore.from_yaml(
                settings.tenants_path, fallback_example=settings.tenants_example_path
            )
            app.state.tenant_store = store
            log.info("tenant_store: file-backed (%d tenants)",
                     len(store.all_tenants()))
        tracker.step("tenant_store")

        # 4. rate limiter
        app.state.rate_limiter = RateLimiter()
        tracker.step("rate_limiter")

        # 5. manifest chains (classical + optional quantum)
        chain = load_active_manifest_chain(settings.artifacts_dir)
        app.state.manifest_chain = chain
        app.state.quantum_chain = load_active_quantum_chain(settings.artifacts_dir)
        tracker.step("manifest_chain")

        # 6. orchestrator + quantum executor
        if state_builder is not None:
            await state_builder(app, settings, tracker)
        else:
            await _default_orchestrator_builder(app, settings, tracker)
        tracker.step("orchestrator")

        # 7. job queue (with usage store) — Redis if env-configured, else in-memory
        from .usage import UsageStore
        from .jobs_redis import maybe_build_queue as _redis_q
        usage_store = UsageStore(settings.artifacts_dir)
        app.state.usage_store = usage_store
        redis_q = _redis_q(settings.artifacts_dir, usage_store=usage_store)
        if redis_q is not None:
            app.state.job_queue = redis_q
            log.info("job_queue: redis backend")
            if hasattr(app.state, "orchestrator"):
                await redis_q.start_workers(settings.job_workers, app.state.orchestrator)
        else:
            queue = JobQueue(settings.artifacts_dir, usage_store=usage_store)
            app.state.job_queue = queue
            log.info("job_queue: in-memory backend")
            if hasattr(app.state, "orchestrator"):
                await queue.start_workers(settings.job_workers, app.state.orchestrator)
        tracker.step("job_workers")

        # 8. ready
        tracker.step("ready")
        tracker.mark_ready()
        log.info("service_ready", extra={"git_sha": settings.git_sha,
                                         "config_hash": settings.config_hash})
    except Exception as e:
        log.exception("startup_failed")
        tracker.mark_failed(f"{type(e).__name__}: {e}")

    try:
        yield
    finally:
        # Shutdown
        try:
            queue = getattr(app.state, "job_queue", None)
            if queue is not None:
                await queue.stop()
        except Exception:
            log.exception("queue_stop_failed")
        try:
            shutdown_pools(wait=False)
        except Exception:
            log.exception("pool_shutdown_failed")


async def _default_orchestrator_builder(
    app: FastAPI, settings: Settings, tracker: StartupTracker
) -> None:
    """Default orchestrator wiring against the kept library code.

    This is intentionally tolerant: if any kept module fails to import, we
    leave app.state.orchestrator unset and the /predict route returns 503
    with a clear reason. /status still works and reports the failure.
    """
    try:
        from .legacy_adapter import build_orchestrator_from_legacy
    except Exception as e:
        log.warning("legacy_adapter import failed: %s", e)
        return
    try:
        orch, qe = await build_orchestrator_from_legacy(
            settings,
            app.state.manifest_chain,
            getattr(app.state, "quantum_chain", None),
        )
        app.state.orchestrator = orch
        app.state.entity_resolver = orch.resolver
        app.state.embedder = getattr(orch, "_embedder", None)
        app.state.quantum_executor = qe
    except Exception as e:
        log.exception("orchestrator_build_failed")
        # Surface via tracker so /status reports the cause
        tracker.mark_failed(f"orchestrator: {type(e).__name__}: {e}")


def create_app(
    settings: Optional[Settings] = None,
    state_builder: Optional[StateBuilder] = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        settings: optional pre-built Settings; defaults to Settings.from_env().
        state_builder: optional callable for tests to install fake orchestrators.
            Signature: async (app, settings, tracker) -> None.
            Should set app.state.orchestrator (and optionally embedder,
            quantum_executor, entity_resolver).
    """
    settings = settings or Settings.from_env()
    setup_logging(settings)
    tracing.setup_tracing("hetqml-service")

    tracker = StartupTracker(total_steps=len(LIFESPAN_STEPS))

    app = FastAPI(
        title="hetqml-service",
        version=settings.version,
        lifespan=_default_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = settings
    app.state.tracker = tracker
    if state_builder is not None:
        app.state._state_builder = state_builder  # type: ignore[attr-defined]

    # Tenant store has to exist before AuthMiddleware. We add a placeholder
    # marked loaded=False so AuthMiddleware can return 503 (not 401) for any
    # request that lands during the lifespan startup window. The real store
    # with loaded=True is installed by the lifespan handler.
    app.state.tenant_store = TenantStore(tenants=[], loaded=False)

    # Middleware order matters — outermost first; CORS wraps everything,
    # then correlation id (so logs always have it), then auth, then logging.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-Id"],
        expose_headers=["X-Request-Id"],
    )
    app.add_middleware(AuthMiddlewareProxy)  # reads store dynamically from app.state
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    for r in ALL_ROUTERS:
        app.include_router(r)

    tracing.instrument_app(app)

    return app


class AuthMiddlewareProxy(AuthMiddleware):
    """AuthMiddleware that reads the live tenant_store from app.state per request.

    Needed because the real store is built during lifespan, after middleware
    is constructed. Without this, AuthMiddleware would capture the empty
    placeholder store at __init__ time.
    """

    def __init__(self, app, **_kwargs):
        # Don't call super().__init__ with a store — we'll read it dynamically.
        from starlette.middleware.base import BaseHTTPMiddleware
        BaseHTTPMiddleware.__init__(self, app)
        self._public = AuthMiddleware.DEFAULT_PUBLIC

    async def dispatch(self, request, call_next):
        # Pull the live store off app.state for every request.
        self._store = request.app.state.tenant_store
        return await super().dispatch(request, call_next)
