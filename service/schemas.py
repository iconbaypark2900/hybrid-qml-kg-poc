"""All Pydantic v2 schemas for the service.

One file by design — under ~300 LOC, easier to OpenAPI-generate frontend
types from one stable surface than from a sprawl of submodules.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Liveness / status
# ---------------------------------------------------------------------------


class HealthState(str, Enum):
    OK = "ok"
    LOADING = "loading"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class ComponentHealth(BaseModel):
    name: str
    state: HealthState
    detail: Optional[str] = None
    last_checked: float


class QuantumMode(str, Enum):
    SIMULATOR = "simulator"
    GPU_SIMULATOR = "gpu_simulator"
    IBM_HERON = "ibm_heron"
    UNAVAILABLE = "unavailable"


class QuantumStatus(BaseModel):
    default_mode: QuantumMode
    available_modes: list[QuantumMode]
    ibm_runtime_reachable: bool
    last_calibration_check: Optional[float] = None
    note: Optional[str] = None  # e.g. "weights not persisted; quantum_strict will 503"


class StartupProgress(BaseModel):
    state: Literal["starting", "ready", "failed"]
    components_loaded: int
    components_total: int
    elapsed_seconds: float
    last_step: str


# ---------------------------------------------------------------------------
# Manifest DAG
# ---------------------------------------------------------------------------


class ArtifactRef(BaseModel):
    path: str
    sha256: str
    size_bytes: int


class HetionetSource(BaseModel):
    """Identifies the Hetionet release the embedding was trained on.

    Pinning this prevents silent drift if upstream Hetionet ships a new
    release. `release_tag` is the GitHub tag/release; `nodes_sha256` and
    `edges_sha256` are content hashes of the actual files used.
    """
    release_tag: str = "v1.0"
    release_url: Optional[str] = "https://github.com/hetio/hetionet/tree/main/hetnet/json"
    nodes_sha256: Optional[str] = None
    edges_sha256: Optional[str] = None
    snapshot_path: Optional[str] = None


class EmbeddingManifest(BaseModel):
    manifest_id: str
    kind: Literal["embedding"] = "embedding"
    created_at: float
    git_sha: str
    seed: int
    relation: str
    max_entities: int  # 0 means uncapped; the 300 footgun, surfaced
    method: str        # TransE | ComplEx | RotatE | DistMult | fallback
    dim: int
    epochs: int
    artifacts: dict[str, ArtifactRef]
    config_files: dict[str, str]
    hetionet: HetionetSource = HetionetSource()


class FeaturePipelineManifest(BaseModel):
    manifest_id: str
    kind: Literal["feature_pipeline"] = "feature_pipeline"
    parent_embedding: str
    created_at: float
    qml_dim: int
    mode: Literal["diff", "hadamard", "both", "classical_only"]
    artifacts: dict[str, ArtifactRef]


class ModelManifest(BaseModel):
    manifest_id: str
    kind: Literal["classical", "quantum"]
    parent_feature_pipeline: str
    created_at: float
    model_type: str
    quantum_execution_mode_at_train: Optional[QuantumMode] = None
    artifacts: dict[str, ArtifactRef]


class ManifestChain(BaseModel):
    """Resolved chain: model → feature_pipeline → embedding."""
    embedding_id: str
    feature_pipeline_id: str
    model_id: str


# ---------------------------------------------------------------------------
# Tenancy
# ---------------------------------------------------------------------------


class TenantQuota(BaseModel):
    model_config = ConfigDict(frozen=True)
    requests_per_minute: int = 60
    requests_per_day: int = 5000
    can_use_quantum_strict: bool = False
    can_use_ibm_hardware: bool = False
    max_batch_size: int = 50
    is_admin: bool = False
    pinned_classical_model_id: Optional[str] = None
    pinned_quantum_model_id: Optional[str] = None
    monthly_quantum_seconds_budget: Optional[float] = None


class TenantUsage(BaseModel):
    """Cumulative usage counters per tenant. Reset monthly by ops."""
    tenant_id: str
    period_start: float
    quantum_seconds_used: float = 0.0
    quantum_jobs_run: int = 0
    estimated_cost_usd: float = 0.0


class Tenant(BaseModel):
    """Internal tenant record. api_key_sha256 is NEVER returned over HTTP.

    During key rotation, both `api_key_sha256` (the new key) and
    `previous_api_key_sha256` (the old one, valid until `previous_key_expires_at`)
    accept incoming requests. Once expiry passes, only the new key works.
    """
    model_config = ConfigDict(frozen=True)
    tenant_id: str
    name: str
    api_key_sha256: str
    quota: TenantQuota
    created_at: float
    is_system: bool = False  # legacy/system tenants for migration data
    previous_api_key_sha256: Optional[str] = None
    previous_key_expires_at: Optional[float] = None


class TenantView(BaseModel):
    """Public-facing tenant projection. No api_key_sha256."""
    tenant_id: str
    name: str
    quota: TenantQuota


# ---------------------------------------------------------------------------
# Status (top-level)
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    service_version: str
    git_sha: str
    config_hash: str
    overall: HealthState
    startup: StartupProgress
    components: list[ComponentHealth]
    quantum: QuantumStatus
    active_manifest_chain: Optional[ManifestChain] = None
    active_quantum_manifest_chain: Optional[ManifestChain] = None
    tenant: TenantView


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class PredictMethod(str, Enum):
    CLASSICAL = "classical"
    QUANTUM_STRICT = "quantum_strict"        # 503 if quantum unavailable
    QUANTUM_PREFERRED = "quantum_preferred"  # fall back, tag response
    AUTO = "auto"


class PredictRequest(BaseModel):
    drug_id: Annotated[str, Field(min_length=1, max_length=64,
                                  description="Exact DrugBank ID. No fuzzy matching.")]
    disease_id: Annotated[str, Field(min_length=1, max_length=64,
                                     description="Exact DOID identifier.")]
    method: PredictMethod = PredictMethod.AUTO


class PredictResponse(BaseModel):
    drug_id: str
    disease_id: str
    probability: float
    method_requested: PredictMethod
    method_used: PredictMethod
    quantum_mode_used: Optional[QuantumMode] = None
    fallback_reason: Optional[str] = None
    manifest_chain: ManifestChain
    request_id: str
    tenant_id: str


class BatchPredictRequest(BaseModel):
    pairs: Annotated[list[PredictRequest], Field(min_length=1, max_length=500)]


class BatchItemResult(BaseModel):
    request: PredictRequest
    response: Optional[PredictResponse] = None
    error: Optional["ErrorResponse"] = None


class BatchPredictResponse(BaseModel):
    results: list[BatchItemResult]
    summary: dict[str, int]  # ok, failed, fallback_used


# ---------------------------------------------------------------------------
# Async jobs (IBM hardware predictions)
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobSubmitRequest(BaseModel):
    request: PredictRequest
    callback_url: Optional[str] = None  # reserved; not implemented in v1


class JobCost(BaseModel):
    """Resource accounting for a single job. `quantum_seconds` is the IBM
    Runtime billable time; `shots` is the total quantum shot count."""
    quantum_seconds: float = 0.0
    shots: int = 0
    backend: Optional[str] = None
    estimated_cost_usd: Optional[float] = None


class JobRecord(BaseModel):
    job_id: str
    tenant_id: str
    status: JobStatus
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    request: PredictRequest
    response: Optional[PredictResponse] = None
    error: Optional["ErrorResponse"] = None
    quantum_mode: QuantumMode
    cost: Optional[JobCost] = None
    callback_url: Optional[str] = None
    callback_status: Optional[str] = None  # "pending" | "delivered" | "failed:<reason>"


# ---------------------------------------------------------------------------
# Evaluations (separate from manifest — re-evaluating doesn't mint a manifest)
# ---------------------------------------------------------------------------


class EvaluationRecord(BaseModel):
    evaluation_id: str
    tenant_id: str
    manifest_chain: ManifestChain
    created_at: float
    test_set_hash: str
    metrics: dict[str, float]
    cv_folds: Optional[int] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorCode(str, Enum):
    AUTH_MISSING = "auth_missing"
    AUTH_INVALID = "auth_invalid"
    RATE_LIMITED = "rate_limited"
    UNKNOWN_ENTITY = "unknown_entity"
    INVALID_REQUEST = "invalid_request"
    QUANTUM_UNAVAILABLE = "quantum_unavailable"
    SERVICE_NOT_READY = "service_not_ready"
    MANIFEST_MISSING = "manifest_missing"
    FEATURE_DISABLED = "feature_disabled"
    USE_JOBS_ENDPOINT = "use_jobs_endpoint"
    NOT_FOUND = "not_found"
    INTERNAL = "internal"


class ErrorResponse(BaseModel):
    code: ErrorCode
    message: str
    request_id: Optional[str] = None
    detail: Optional[dict[str, Any]] = None


# Forward ref resolution
BatchItemResult.model_rebuild()
JobRecord.model_rebuild()
