"""All API routers in one file.

Endpoints:
    GET  /healthz                                  liveness; no auth
    GET  /status                                   honest liveness; auth required
    POST /predict                                  sync predict (classical / sim quantum)
    POST /predict/batch
    POST /jobs                                     async predict (IBM hardware)
    GET  /jobs/{job_id}
    GET  /manifest/active
    GET  /manifest/embedding/{manifest_id}
    GET  /manifest/feature-pipeline/{manifest_id}
    GET  /manifest/model/{manifest_id}
    GET  /evaluations
    GET  /evaluations/{evaluation_id}
"""
from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel as _PydanticBase  # noqa: F401  (re-export safety)

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .deps import (
    enforce_rate_limit,
    get_job_queue,
    get_orchestrator,
    get_request_id,
    get_settings,
    get_tenant,
    get_tracker,
)
from .observability import (
    derive_overall,
    probe_all_components,
    probe_quantum_status,
)
from .orchestration import (
    Orchestrator,
    QuantumUnavailableError,
    UnknownEntityError,
    UseJobsEndpointError,
)
from .persistence import (
    list_evaluations,
    load_embedding_manifest,
    load_feature_pipeline_manifest,
    load_model_manifest,
)
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    EmbeddingManifest,
    ErrorCode,
    ErrorResponse,
    EvaluationRecord,
    FeaturePipelineManifest,
    JobRecord,
    JobSubmitRequest,
    ManifestChain,
    ModelManifest,
    PredictRequest,
    PredictResponse,
    StatusResponse,
    TenantUsage,
    TenantView,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public health (no auth)
# ---------------------------------------------------------------------------


public_router = APIRouter(tags=["health"])


@public_router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness probe: process is alive. Doesn't check downstream deps.
    Use with k8s `livenessProbe` to restart a hung process."""
    return {"status": "ok"}


@public_router.get("/readyz")
async def readyz(request: Request) -> dict[str, object]:
    """Readiness probe: process is up AND ready to serve real traffic.
    Differs from /healthz in that it fails (503) until lifespan completes
    AND all critical components are non-unavailable. Use with k8s
    `readinessProbe` to gate traffic routing."""
    tracker = getattr(request.app.state, "tracker", None)
    if tracker is None or not tracker.is_ready():
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "reason": tracker.current_step if tracker else "no tracker",
            },
        )
    # Critical component check — if classical_model or entity_mappings is
    # unavailable, serving a /predict will 5xx, so we shouldn't claim ready.
    from .observability import probe_classical_model, probe_entity_mappings
    classical = await probe_classical_model(request.app.state)
    entities = await probe_entity_mappings(request.app.state)
    blockers = [c for c in (classical, entities) if c.state.value == "unavailable"]
    if blockers:
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "reason": "critical components unavailable",
                "blockers": [{"name": c.name, "detail": c.detail} for c in blockers],
            },
        )
    return {"ready": True}


# ---------------------------------------------------------------------------
# /status (auth required, but no readiness gate)
# ---------------------------------------------------------------------------


status_router = APIRouter(prefix="/status", tags=["status"])


@status_router.get("", response_model=StatusResponse)
async def get_status(
    request: Request,
    tenant=Depends(get_tenant),
    tracker=Depends(get_tracker),
    settings=Depends(get_settings),
) -> StatusResponse:
    components = await probe_all_components(request.app.state)
    quantum = probe_quantum_status(request.app.state, tenant=tenant)
    return StatusResponse(
        service_version=settings.version,
        git_sha=settings.git_sha,
        config_hash=settings.config_hash,
        overall=derive_overall(tracker, components),
        startup=tracker.snapshot(),
        components=components,
        quantum=quantum,
        active_manifest_chain=getattr(request.app.state, "manifest_chain", None),
        active_quantum_manifest_chain=getattr(request.app.state, "quantum_chain", None),
        tenant=TenantView(tenant_id=tenant.tenant_id, name=tenant.name, quota=tenant.quota),
    )


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------


predict_router = APIRouter(prefix="/predict", tags=["predict"])


@predict_router.post("", response_model=PredictResponse,
                     dependencies=[Depends(enforce_rate_limit)])
async def predict(
    body: PredictRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    tenant=Depends(get_tenant),
    request_id: str = Depends(get_request_id),
) -> PredictResponse:
    if (body.method.value == "quantum_strict"
            and orch.router.would_use_ibm_hardware()):
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code=ErrorCode.USE_JOBS_ENDPOINT,
                message="IBM hardware predictions must go through POST /jobs",
                request_id=request_id,
            ).model_dump(),
        )
    try:
        return await orch.predict(body, tenant=tenant, request_id=request_id)
    except UnknownEntityError as e:
        raise HTTPException(
            status_code=422,
            detail=ErrorResponse(
                code=ErrorCode.UNKNOWN_ENTITY, message=str(e),
                request_id=request_id, detail={"kind": e.kind, "value": e.value},
            ).model_dump(),
        ) from None
    except QuantumUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail=ErrorResponse(
                code=ErrorCode.QUANTUM_UNAVAILABLE, message=str(e),
                request_id=request_id,
            ).model_dump(),
        ) from None
    except UseJobsEndpointError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code=ErrorCode.USE_JOBS_ENDPOINT, message=str(e),
                request_id=request_id,
            ).model_dump(),
        ) from None


@predict_router.post("/batch", response_model=BatchPredictResponse,
                     dependencies=[Depends(enforce_rate_limit)])
async def predict_batch(
    body: BatchPredictRequest,
    orch: Orchestrator = Depends(get_orchestrator),
    tenant=Depends(get_tenant),
    request_id: str = Depends(get_request_id),
) -> BatchPredictResponse:
    if len(body.pairs) > tenant.quota.max_batch_size:
        raise HTTPException(
            status_code=413,
            detail=ErrorResponse(
                code=ErrorCode.INVALID_REQUEST,
                message=f"batch size {len(body.pairs)} exceeds tenant max {tenant.quota.max_batch_size}",
                request_id=request_id,
            ).model_dump(),
        )
    return await orch.predict_batch(body, tenant=tenant, request_id=request_id)


# ---------------------------------------------------------------------------
# /jobs (async / IBM hardware)
# ---------------------------------------------------------------------------


jobs_router = APIRouter(prefix="/jobs", tags=["jobs"])


@jobs_router.post("", response_model=JobRecord,
                  dependencies=[Depends(enforce_rate_limit)])
async def submit_job(
    body: JobSubmitRequest,
    queue=Depends(get_job_queue),
    tenant=Depends(get_tenant),
    request_id: str = Depends(get_request_id),
) -> JobRecord:
    if not tenant.quota.can_use_ibm_hardware:
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                code=ErrorCode.FEATURE_DISABLED,
                message="tenant lacks can_use_ibm_hardware quota",
                request_id=request_id, detail={"feature": "ibm_hardware"},
            ).model_dump(),
        )
    try:
        return await queue.submit(tenant, body.request, callback_url=body.callback_url)
    except PermissionError as e:
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                code=ErrorCode.FEATURE_DISABLED,
                message=str(e), request_id=request_id,
            ).model_dump(),
        ) from None


@jobs_router.get("/{job_id}", response_model=JobRecord)
async def get_job(
    job_id: str,
    queue=Depends(get_job_queue),
    tenant=Depends(get_tenant),
    request_id: str = Depends(get_request_id),
) -> JobRecord:
    rec = await queue.get(tenant, job_id)
    if rec is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code=ErrorCode.NOT_FOUND, message="job not found",
                request_id=request_id,
            ).model_dump(),
        )
    return rec


# ---------------------------------------------------------------------------
# /manifest
# ---------------------------------------------------------------------------


manifest_router = APIRouter(prefix="/manifest", tags=["manifest"])


@manifest_router.get("/active", response_model=ManifestChain)
async def get_active_chain(
    request: Request,
    request_id: str = Depends(get_request_id),
    _tenant=Depends(get_tenant),
) -> ManifestChain:
    chain = getattr(request.app.state, "manifest_chain", None)
    if chain is None:
        raise HTTPException(
            status_code=404,
            detail=ErrorResponse(
                code=ErrorCode.MANIFEST_MISSING, message="no active manifest chain",
                request_id=request_id,
            ).model_dump(),
        )
    return chain


@manifest_router.get("/embedding/{manifest_id}", response_model=EmbeddingManifest)
async def read_embedding_manifest(
    manifest_id: str,
    settings=Depends(get_settings),
    request_id: str = Depends(get_request_id),
    _tenant=Depends(get_tenant),
) -> EmbeddingManifest:
    try:
        return load_embedding_manifest(settings.artifacts_dir, manifest_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=ErrorResponse(
            code=ErrorCode.NOT_FOUND, message="embedding manifest not found",
            request_id=request_id).model_dump()) from None


@manifest_router.get("/feature-pipeline/{manifest_id}",
                     response_model=FeaturePipelineManifest)
async def read_feature_pipeline_manifest(
    manifest_id: str,
    settings=Depends(get_settings),
    request_id: str = Depends(get_request_id),
    _tenant=Depends(get_tenant),
) -> FeaturePipelineManifest:
    try:
        return load_feature_pipeline_manifest(settings.artifacts_dir, manifest_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=ErrorResponse(
            code=ErrorCode.NOT_FOUND, message="feature pipeline manifest not found",
            request_id=request_id).model_dump()) from None


@manifest_router.get("/model/{manifest_id}", response_model=ModelManifest)
async def read_model_manifest(
    manifest_id: str,
    settings=Depends(get_settings),
    request_id: str = Depends(get_request_id),
    _tenant=Depends(get_tenant),
) -> ModelManifest:
    try:
        return load_model_manifest(settings.artifacts_dir, manifest_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=ErrorResponse(
            code=ErrorCode.NOT_FOUND, message="model manifest not found",
            request_id=request_id).model_dump()) from None


# ---------------------------------------------------------------------------
# /evaluations
# ---------------------------------------------------------------------------


usage_router = APIRouter(prefix="/usage", tags=["usage"])


@usage_router.get("", response_model=TenantUsage)
async def get_usage(
    request: Request,
    tenant=Depends(get_tenant),
) -> TenantUsage:
    store = getattr(request.app.state, "usage_store", None)
    if store is None:
        # Service was built without a usage store; return zeros.
        import time
        return TenantUsage(tenant_id=tenant.tenant_id, period_start=time.time())
    return store.get(tenant.tenant_id)


eval_router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@eval_router.get("", response_model=list[EvaluationRecord])
async def list_evals(
    settings=Depends(get_settings),
    tenant=Depends(get_tenant),
    model_manifest_id: Optional[str] = None,
    limit: int = 50,
) -> list[EvaluationRecord]:
    limit = max(1, min(limit, 500))
    return await list_evaluations(
        settings.artifacts_dir, tenant.tenant_id,
        model_manifest_id=model_manifest_id, limit=limit,
    )


@eval_router.get("/{evaluation_id}", response_model=EvaluationRecord)
async def read_evaluation(
    evaluation_id: str,
    settings=Depends(get_settings),
    tenant=Depends(get_tenant),
    request_id: str = Depends(get_request_id),
) -> EvaluationRecord:
    # Linear scan over the tenant's JSONL — cheap enough for low-volume case;
    # add an index if usage grows.
    records = await list_evaluations(
        settings.artifacts_dir, tenant.tenant_id, limit=1_000_000
    )
    for rec in records:
        if rec.evaluation_id == evaluation_id:
            return rec
    raise HTTPException(404, detail=ErrorResponse(
        code=ErrorCode.NOT_FOUND, message="evaluation not found",
        request_id=request_id).model_dump())


# ---------------------------------------------------------------------------
# /admin (privileged operations; gated to tenants with admin quota)
# ---------------------------------------------------------------------------


admin_router = APIRouter(prefix="/admin", tags=["admin"])


class ReloadResult(BaseModel):
    classical_chain: Optional[ManifestChain] = None
    quantum_chain: Optional[ManifestChain] = None
    classical_changed: bool
    quantum_changed: bool
    note: Optional[str] = None


def _require_admin(tenant, request_id: str) -> None:
    if not (tenant.quota.is_admin or tenant.quota.can_use_quantum_strict):
        raise HTTPException(
            status_code=403,
            detail=ErrorResponse(
                code=ErrorCode.FEATURE_DISABLED,
                message="admin operations require is_admin quota",
                request_id=request_id,
            ).model_dump(),
        )


@admin_router.post("/reload")
async def admin_reload(
    request: Request,
    request_id: str = Depends(get_request_id),
    tenant=Depends(get_tenant),
) -> ReloadResult:
    """Re-read LATEST.txt + LATEST_QUANTUM.txt and reload predictors if changed."""
    _require_admin(tenant, request_id)

    from .legacy_adapter import build_orchestrator_from_legacy
    from .persistence import (
        load_active_manifest_chain,
        load_active_quantum_chain,
    )

    settings = request.app.state.settings
    old_classical = getattr(request.app.state, "manifest_chain", None)
    old_quantum = getattr(request.app.state, "quantum_chain", None)

    new_classical = load_active_manifest_chain(settings.artifacts_dir)
    new_quantum = load_active_quantum_chain(settings.artifacts_dir)

    classical_changed = new_classical != old_classical
    quantum_changed = new_quantum != old_quantum

    if classical_changed or quantum_changed:
        # Rebuild the orchestrator. This is the same path lifespan uses.
        try:
            orch, qe = await build_orchestrator_from_legacy(
                settings, new_classical, new_quantum,
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    code=ErrorCode.INTERNAL,
                    message=f"reload failed: {type(e).__name__}: {e}",
                    request_id=request_id,
                ).model_dump(),
            ) from None
        request.app.state.manifest_chain = new_classical
        request.app.state.quantum_chain = new_quantum
        request.app.state.orchestrator = orch
        request.app.state.entity_resolver = orch.resolver
        request.app.state.embedder = getattr(orch, "_embedder", None)
        request.app.state.quantum_executor = qe

    return ReloadResult(
        classical_chain=new_classical,
        quantum_chain=new_quantum,
        classical_changed=classical_changed,
        quantum_changed=quantum_changed,
        note=None if (classical_changed or quantum_changed)
              else "no chain change detected; orchestrator left in place",
    )


@admin_router.post("/usage/reset", response_model=TenantUsage)
async def admin_usage_reset(
    request: Request,
    target_tenant_id: str,
    tenant=Depends(get_tenant),
    request_id: str = Depends(get_request_id),
) -> TenantUsage:
    """Reset a target tenant's usage counters at the start of a billing period."""
    _require_admin(tenant, request_id)
    store = request.app.state.usage_store
    return store.reset(target_tenant_id)


class TenantsReloadResult(BaseModel):
    tenant_count: int
    note: Optional[str] = None


@admin_router.post("/tenants/reload", response_model=TenantsReloadResult)
async def admin_tenants_reload(
    request: Request,
    tenant=Depends(get_tenant),
    request_id: str = Depends(get_request_id),
) -> TenantsReloadResult:
    """Re-read secrets/tenants.yaml without restarting the service. Use
    after adding/rotating a tenant. Existing in-flight requests against
    the previous tenant key continue to authenticate against whatever
    state was loaded when their request started."""
    _require_admin(tenant, request_id)
    from .tenants import TenantStore
    settings = request.app.state.settings
    new_store = TenantStore.from_yaml(
        settings.tenants_path, fallback_example=settings.tenants_example_path
    )
    request.app.state.tenant_store = new_store
    return TenantsReloadResult(
        tenant_count=len(new_store.all_tenants()),
        note="tenant_store hot-reloaded from disk",
    )


ALL_ROUTERS = (
    public_router,
    status_router,
    predict_router,
    jobs_router,
    manifest_router,
    eval_router,
    usage_router,
    admin_router,
)
