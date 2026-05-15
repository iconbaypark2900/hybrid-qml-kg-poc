"""HTTP middleware: correlation IDs, tenant auth, structured request logging."""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from typing import Awaitable, Callable, Iterable

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .observability import request_id_ctx, tenant_id_ctx
from .schemas import ErrorCode, ErrorResponse
from .tenants import TenantStore

log = logging.getLogger("service.request")


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Generate or accept X-Request-Id; pin it to contextvars for logging."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get("X-Request-Id") or uuid.uuid4().hex
        request.state.request_id = rid
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["X-Request-Id"] = rid
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, latency_ms, request_id, tenant_id."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000
            log.exception(
                "request_failed",
                extra={"method": request.method, "path": request.url.path,
                       "status": 500, "latency_ms": round(elapsed_ms, 2)},
            )
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.info(
            "request",
            extra={"method": request.method, "path": request.url.path,
                   "status": response.status_code, "latency_ms": round(elapsed_ms, 2)},
        )
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """Bearer-token → Tenant resolver. Skips public paths."""

    DEFAULT_PUBLIC: tuple[str, ...] = (
        "/healthz", "/openapi.json", "/docs", "/redoc", "/docs/oauth2-redirect",
    )

    def __init__(
        self,
        app: ASGIApp,
        tenant_store: TenantStore,
        public_paths: Iterable[str] = DEFAULT_PUBLIC,
    ):
        super().__init__(app)
        self._store = tenant_store
        self._public = set(public_paths)

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # CORS preflights (OPTIONS) must bypass auth — the browser sends them
        # without an Authorization header on purpose, and CORSMiddleware (which
        # runs after this in the stack) needs to answer them.
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in self._public or path.startswith("/docs/"):
            return await call_next(request)

        # Startup race guard: lifespan replaces the placeholder TenantStore
        # with the real one. A request that lands in the gap sees an empty
        # store and would 401 against a valid key. Return 503 with a clear
        # signal so clients retry once startup completes.
        store = getattr(request.app.state, "tenant_store", None)
        if store is None or not getattr(store, "_loaded", False):
            return _error_response(503, ErrorCode.SERVICE_NOT_READY,
                                   "tenant store still loading",
                                   request_id=getattr(request.state, "request_id", None))

        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return _error_response(401, ErrorCode.AUTH_MISSING,
                                   "Bearer token required",
                                   request_id=getattr(request.state, "request_id", None))

        token = auth[7:].strip()
        if not token:
            return _error_response(401, ErrorCode.AUTH_MISSING,
                                   "Bearer token empty",
                                   request_id=getattr(request.state, "request_id", None))

        sha = hashlib.sha256(token.encode()).hexdigest()
        tenant = await self._store.find_by_key_sha(sha)
        if tenant is None or tenant.is_system:
            return _error_response(401, ErrorCode.AUTH_INVALID,
                                   "unknown API key",
                                   request_id=getattr(request.state, "request_id", None))

        request.state.tenant = tenant
        token_ctx = tenant_id_ctx.set(tenant.tenant_id)
        try:
            response = await call_next(request)
        finally:
            tenant_id_ctx.reset(token_ctx)
        return response


def _error_response(status: int, code: ErrorCode, message: str,
                    request_id: str | None = None) -> JSONResponse:
    body = ErrorResponse(code=code, message=message, request_id=request_id)
    return JSONResponse(status_code=status, content=body.model_dump())
