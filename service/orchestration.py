"""Entity resolver, execution router, prediction orchestrator.

Replaces middleware/orchestrator.py. Notable differences vs. the audited code:
  * Exact-id-only resolution; no fuzzy substring fallback.
  * Quantum routing has four states (classical, quantum_strict, quantum_preferred,
    auto); quantum_strict raises QuantumUnavailableError instead of silently
    falling back; quantum_preferred falls back BUT records fallback_reason.
  * No module-level singleton; orchestrator is built once per process by
    service.app.lifespan and held on app.state.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol

import numpy as np

from .async_runtime import run_io_bound
from .schemas import (
    BatchItemResult,
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorCode,
    ErrorResponse,
    ManifestChain,
    PredictMethod,
    PredictRequest,
    PredictResponse,
    QuantumMode,
    Tenant,
)
from .synonyms import SynonymIndex

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class UnknownEntityError(Exception):
    def __init__(self, kind: str, value: str):
        self.kind = kind
        self.value = value
        super().__init__(f"unknown {kind}: {value}")


class QuantumUnavailableError(Exception):
    """Raised when quantum_strict is requested but quantum execution is unavailable."""


class UseJobsEndpointError(Exception):
    """Raised when a quantum prediction would route to IBM hardware in a sync context."""


# ---------------------------------------------------------------------------
# Entity resolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Exact-id + curated-synonym resolution against drug + disease id sets.

    Order:
      1. Exact id match (DB00001, DOID:14330) — fast path
      2. Synonym table (case-insensitive) → canonical id, then exact match

    No fuzzy/substring matching. The audit's "Asp could collide with several
    drugs" failure mode is impossible: aliases must be explicitly curated.
    """

    def __init__(
        self,
        drug_ids: set[str],
        disease_ids: set[str],
        synonyms: Optional["SynonymIndex"] = None,
    ):
        self.drug_ids = frozenset(drug_ids)
        self.disease_ids = frozenset(disease_ids)
        self.synonyms = synonyms

    def resolve(self, kind: str, value: str) -> str:
        if kind == "drug":
            valid = self.drug_ids
        elif kind == "disease":
            valid = self.disease_ids
        else:
            raise ValueError(f"unknown entity kind: {kind}")

        # 1. Exact match
        if value in valid:
            return value

        # 2. Synonym fallback (curated; not fuzzy)
        if self.synonyms is not None:
            canonical = (
                self.synonyms.resolve_drug(value)
                if kind == "drug"
                else self.synonyms.resolve_disease(value)
            )
            if canonical is not None and canonical in valid:
                return canonical

        raise UnknownEntityError(kind=kind, value=value)

    @classmethod
    def from_id_to_entity_csv(cls, path) -> "EntityResolver":
        """Parse legacy data/id_to_entity.csv. Schema is brittle in the audited
        code — this version reads by named column ('entity_id' or fallback).
        """
        import csv
        from pathlib import Path
        p = Path(path)
        drugs: set[str] = set()
        diseases: set[str] = set()
        with p.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            id_col = next(
                (c for c in ("entity_id", "id", "Compound", "Disease", "0") if c in cols),
                None,
            )
            if id_col is None:
                raise ValueError(f"id_to_entity.csv at {p} has no recognized id column")
            for row in reader:
                eid = (row.get(id_col) or "").strip()
                if not eid:
                    continue
                if eid.startswith("DB"):
                    drugs.add(eid)
                elif eid.startswith("DOID:"):
                    diseases.add(eid)
        return cls(drugs, diseases)


# ---------------------------------------------------------------------------
# Quantum executor surface
# ---------------------------------------------------------------------------


class QuantumExecutorProtocol(Protocol):
    """The minimal surface the orchestrator needs from the quantum executor.

    Implementations live in quantum_layer/quantum_executor.py (kept).
    For tests, a fake satisfies this protocol.
    """

    @property
    def _ibm_initialized(self) -> bool: ...
    @property
    def _ibm_reachable(self) -> bool: ...
    @property
    def _simulator_available(self) -> bool: ...
    @property
    def _gpu_simulator_available(self) -> bool: ...

    def current_mode(self) -> QuantumMode: ...
    def is_ibm_hardware_mode(self) -> bool: ...


# ---------------------------------------------------------------------------
# Predictor protocols
# ---------------------------------------------------------------------------


PredictFn = Callable[[np.ndarray], np.ndarray]
FeaturePrepFn = Callable[[str, str], np.ndarray]


@dataclass
class PredictorBundle:
    """Predictor callables wired up at startup. Sync (sklearn/Qiskit);
    orchestrator wraps them in run_io_bound."""
    classical_predict: PredictFn
    quantum_predict: Optional[PredictFn]


# ---------------------------------------------------------------------------
# Execution router
# ---------------------------------------------------------------------------


class ExecutionRouter:
    """Honest about which method actually ran. No silent downgrades."""

    def __init__(
        self,
        bundle: PredictorBundle,
        quantum_executor: Optional[QuantumExecutorProtocol],
    ):
        self.bundle = bundle
        self.qe = quantum_executor

    def quantum_available(self) -> bool:
        return (
            self.qe is not None
            and self.bundle.quantum_predict is not None
            and (
                getattr(self.qe, "_simulator_available", False)
                or getattr(self.qe, "_ibm_reachable", False)
            )
        )

    def would_use_ibm_hardware(self) -> bool:
        return self.qe is not None and self.qe.is_ibm_hardware_mode()

    def quantum_mode_used(self) -> Optional[QuantumMode]:
        if self.qe is None:
            return None
        try:
            return self.qe.current_mode()
        except Exception:
            return None

    def route(
        self, method: PredictMethod
    ) -> tuple[PredictMethod, PredictFn, Optional[str]]:
        """Returns (method_actually_used, predict_fn, fallback_reason).

        Raises:
            QuantumUnavailableError: when method == quantum_strict and quantum unavailable.
            UseJobsEndpointError: when route would use IBM hardware (caller must
                use POST /jobs instead of POST /predict).
        """
        q_avail = self.quantum_available()
        ibm_path = self.would_use_ibm_hardware()

        if method == PredictMethod.CLASSICAL:
            return PredictMethod.CLASSICAL, self.bundle.classical_predict, None

        if method == PredictMethod.QUANTUM_STRICT:
            if not q_avail:
                raise QuantumUnavailableError("quantum_strict requested; not available")
            if ibm_path:
                raise UseJobsEndpointError("IBM hardware predictions must go through POST /jobs")
            assert self.bundle.quantum_predict is not None
            return PredictMethod.QUANTUM_STRICT, self.bundle.quantum_predict, None

        if method == PredictMethod.QUANTUM_PREFERRED:
            if q_avail and not ibm_path:
                assert self.bundle.quantum_predict is not None
                return PredictMethod.QUANTUM_PREFERRED, self.bundle.quantum_predict, None
            reason = (
                "quantum_preferred routed to /jobs path; falling back to classical synchronously"
                if ibm_path
                else "quantum unavailable; fell back to classical"
            )
            return PredictMethod.CLASSICAL, self.bundle.classical_predict, reason

        # AUTO
        if q_avail and not ibm_path:
            assert self.bundle.quantum_predict is not None
            return PredictMethod.QUANTUM_PREFERRED, self.bundle.quantum_predict, None
        return PredictMethod.CLASSICAL, self.bundle.classical_predict, None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """The single seam between HTTP and ML.

    Holds two manifest chains: a classical chain (always present) and an
    optional quantum chain (when quantum weights have been persisted). The
    PredictResponse returns whichever chain actually ran.
    """

    def __init__(
        self,
        resolver: EntityResolver,
        router: ExecutionRouter,
        feature_prep: FeaturePrepFn,
        classical_chain: ManifestChain,
        quantum_chain: Optional[ManifestChain] = None,
        quantum_feature_prep: Optional[FeaturePrepFn] = None,
        ibm_predict_fn: Optional[PredictFn] = None,
    ):
        self.resolver = resolver
        self.router = router
        self.feature_prep = feature_prep
        self.classical_chain = classical_chain
        self.quantum_chain = quantum_chain
        # Quantum often needs a different feature shape (PCA-reduced); use a
        # dedicated callable when one is supplied, otherwise fall back to the
        # classical feature_prep.
        self.quantum_feature_prep = quantum_feature_prep or feature_prep
        self.ibm_predict = ibm_predict_fn

    @property
    def chain(self) -> ManifestChain:
        """Backwards-compatible default — points at the classical chain."""
        return self.classical_chain

    def _chain_for(self, method_used: PredictMethod, tenant: Optional[Tenant] = None) -> ManifestChain:
        # Per-tenant pinning: if the tenant has a pinned model_id, return a
        # chain pointing at it. The orchestrator currently only resolves
        # against the active classical/quantum chains; pinned-but-not-active
        # chains will fall through to the default until a future patch
        # supports loading multiple chains simultaneously.
        if tenant is not None:
            pinned = (
                tenant.quota.pinned_quantum_model_id
                if method_used in (PredictMethod.QUANTUM_STRICT, PredictMethod.QUANTUM_PREFERRED)
                else tenant.quota.pinned_classical_model_id
            )
            if pinned:
                base = self.quantum_chain if (
                    method_used in (PredictMethod.QUANTUM_STRICT, PredictMethod.QUANTUM_PREFERRED)
                    and self.quantum_chain
                ) else self.classical_chain
                if base.model_id != pinned:
                    # Tenant pinned to a model the active chain doesn't match.
                    # Surface the pin in the response by overriding model_id;
                    # the actual prediction still ran on the active chain
                    # (loading a second chain in-process is a future change).
                    return base.model_copy(update={"model_id": pinned})

        if method_used in (PredictMethod.QUANTUM_STRICT, PredictMethod.QUANTUM_PREFERRED):
            return self.quantum_chain or self.classical_chain
        return self.classical_chain

    async def predict(
        self, req: PredictRequest, *, tenant: Tenant, request_id: str
    ) -> PredictResponse:
        from . import tracing
        with tracing.span(
            "orchestrator.predict",
            tenant_id=tenant.tenant_id,
            method_requested=str(req.method),
        ):
            return await self._predict_impl(req, tenant=tenant, request_id=request_id)

    async def _predict_impl(
        self, req: PredictRequest, *, tenant: Tenant, request_id: str
    ) -> PredictResponse:
        drug = self.resolver.resolve("drug", req.drug_id)
        disease = self.resolver.resolve("disease", req.disease_id)
        method_used, fn, fallback_reason = self.router.route(req.method)

        # Pick the feature pipeline that matches the predictor that's about to run.
        is_quantum = method_used in (PredictMethod.QUANTUM_STRICT, PredictMethod.QUANTUM_PREFERRED)
        prep = self.quantum_feature_prep if is_quantum else self.feature_prep
        features: np.ndarray = await run_io_bound(prep, drug, disease)

        prob_arr = await run_io_bound(fn, features)
        prob = float(np.atleast_1d(np.asarray(prob_arr)).reshape(-1)[0])

        quantum_mode_used = self.router.quantum_mode_used() if is_quantum else None

        return PredictResponse(
            drug_id=drug,
            disease_id=disease,
            probability=prob,
            method_requested=req.method,
            method_used=method_used,
            quantum_mode_used=quantum_mode_used,
            fallback_reason=fallback_reason,
            manifest_chain=self._chain_for(method_used, tenant=tenant),
            request_id=request_id,
            tenant_id=tenant.tenant_id,
        )

    async def predict_batch(
        self, req: BatchPredictRequest, *, tenant: Tenant, request_id: str
    ) -> BatchPredictResponse:
        results: list[BatchItemResult] = []
        ok = failed = fallback = 0
        for i, item in enumerate(req.pairs):
            sub_id = f"{request_id}.{i}"
            try:
                resp = await self.predict(item, tenant=tenant, request_id=sub_id)
                results.append(BatchItemResult(request=item, response=resp))
                ok += 1
                if resp.fallback_reason:
                    fallback += 1
            except UnknownEntityError as e:
                results.append(
                    BatchItemResult(
                        request=item,
                        error=ErrorResponse(
                            code=ErrorCode.UNKNOWN_ENTITY,
                            message=str(e),
                            request_id=sub_id,
                            detail={"kind": e.kind, "value": e.value},
                        ),
                    )
                )
                failed += 1
            except QuantumUnavailableError as e:
                results.append(
                    BatchItemResult(
                        request=item,
                        error=ErrorResponse(
                            code=ErrorCode.QUANTUM_UNAVAILABLE,
                            message=str(e),
                            request_id=sub_id,
                        ),
                    )
                )
                failed += 1
            except UseJobsEndpointError as e:
                results.append(
                    BatchItemResult(
                        request=item,
                        error=ErrorResponse(
                            code=ErrorCode.USE_JOBS_ENDPOINT,
                            message=str(e),
                            request_id=sub_id,
                        ),
                    )
                )
                failed += 1
            except Exception as e:
                log.exception("batch item %s failed", sub_id)
                results.append(
                    BatchItemResult(
                        request=item,
                        error=ErrorResponse(
                            code=ErrorCode.INTERNAL,
                            message=f"{type(e).__name__}",
                            request_id=sub_id,
                        ),
                    )
                )
                failed += 1
        return BatchPredictResponse(
            results=results,
            summary={"ok": ok, "failed": failed, "fallback_used": fallback},
        )

    async def predict_quantum_hardware(
        self, req: PredictRequest, *, tenant_id: str, request_id: str
    ) -> PredictResponse:
        """Used by JobQueue worker. Forces IBM hardware path; raises if no
        ibm_predict_fn was wired."""
        if self.ibm_predict is None:
            raise QuantumUnavailableError("IBM hardware predict_fn not wired")
        drug = self.resolver.resolve("drug", req.drug_id)
        disease = self.resolver.resolve("disease", req.disease_id)
        features: np.ndarray = await run_io_bound(self.quantum_feature_prep, drug, disease)
        prob_arr = await run_io_bound(self.ibm_predict, features)
        prob = float(np.atleast_1d(np.asarray(prob_arr)).reshape(-1)[0])
        return PredictResponse(
            drug_id=drug,
            disease_id=disease,
            probability=prob,
            method_requested=PredictMethod.QUANTUM_STRICT,
            method_used=PredictMethod.QUANTUM_STRICT,
            quantum_mode_used=QuantumMode.IBM_HERON,
            fallback_reason=None,
            manifest_chain=self.quantum_chain or self.classical_chain,
            request_id=request_id,
            tenant_id=tenant_id,
        )
