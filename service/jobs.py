"""Async job queue for IBM hardware predictions.

Predictions on real quantum hardware can take seconds to minutes — they cannot
be served from a sync /predict route without blocking the event loop and
violating any sane HTTP timeout. POST /jobs returns immediately with a job_id;
clients poll GET /jobs/{job_id}.

This implementation is single-process, in-memory + disk-backed for persistence
across restarts. For multi-worker deployments, replace with Redis/RQ.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .persistence import job_path, load_job, persist_job
from .schemas import (
    ErrorCode,
    ErrorResponse,
    JobCost,
    JobRecord,
    JobStatus,
    PredictRequest,
    QuantumMode,
    Tenant,
)
from .usage import UsageStore, estimate_cost

log = logging.getLogger(__name__)


def _new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:16]}"


class JobQueue:
    """Tenant-isolated async job queue."""

    def __init__(self, root: Path, usage_store: Optional[UsageStore] = None):
        self.root = root
        self.usage = usage_store or UsageStore(root)
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._records: dict[tuple[str, str], JobRecord] = {}
        self._workers: list[asyncio.Task] = []
        self._stopped = asyncio.Event()

    async def submit(
        self,
        tenant: Tenant,
        req: PredictRequest,
        callback_url: Optional[str] = None,
    ) -> JobRecord:
        if not tenant.quota.can_use_ibm_hardware:
            raise PermissionError("tenant lacks can_use_ibm_hardware quota")

        # Enforce monthly quantum-seconds budget if set.
        if tenant.quota.monthly_quantum_seconds_budget is not None:
            usage = self.usage.get(tenant.tenant_id)
            if usage.quantum_seconds_used >= tenant.quota.monthly_quantum_seconds_budget:
                raise PermissionError(
                    f"tenant {tenant.tenant_id} exceeded monthly quantum-seconds "
                    f"budget ({usage.quantum_seconds_used:.1f}/"
                    f"{tenant.quota.monthly_quantum_seconds_budget:.1f})"
                )

        record = JobRecord(
            job_id=_new_job_id(),
            tenant_id=tenant.tenant_id,
            status=JobStatus.QUEUED,
            submitted_at=time.time(),
            request=req,
            quantum_mode=QuantumMode.IBM_HERON,
            callback_url=callback_url,
            callback_status="pending" if callback_url else None,
        )
        await persist_job(record, self.root)
        self._records[(tenant.tenant_id, record.job_id)] = record
        await self._queue.put((tenant.tenant_id, record.job_id))
        return record

    async def get(self, tenant: Tenant, job_id: str) -> Optional[JobRecord]:
        # Look up in-memory first (fast); fall back to disk for jobs that
        # outlived a restart.
        rec = self._records.get((tenant.tenant_id, job_id))
        if rec is None:
            rec = await load_job(self.root, tenant.tenant_id, job_id)
        if rec is None:
            return None
        if rec.tenant_id != tenant.tenant_id:
            # Tenant isolation: never leak across tenants.
            return None
        return rec

    async def start_workers(self, n: int, orchestrator) -> None:
        for i in range(n):
            t = asyncio.create_task(self._worker_loop(i, orchestrator))
            self._workers.append(t)

    async def stop(self, timeout: float = 5.0) -> None:
        self._stopped.set()
        for t in self._workers:
            t.cancel()
        # Drain pending tasks
        for t in self._workers:
            try:
                await asyncio.wait_for(t, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._workers.clear()

    async def _worker_loop(self, worker_id: int, orchestrator) -> None:
        log.info("job_worker_started", extra={"worker_id": worker_id})
        while not self._stopped.is_set():
            try:
                tenant_id, job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._run_job(tenant_id, job_id, orchestrator)
        log.info("job_worker_stopped", extra={"worker_id": worker_id})

    async def _run_job(self, tenant_id: str, job_id: str, orchestrator) -> None:
        rec = self._records.get((tenant_id, job_id))
        if rec is None:
            log.warning("job_record_missing", extra={"tenant_id": tenant_id, "job_id": job_id})
            return
        started = time.time()
        rec = rec.model_copy(update={
            "status": JobStatus.RUNNING,
            "started_at": started,
        })
        self._records[(tenant_id, job_id)] = rec
        await persist_job(rec, self.root)

        try:
            resp = await orchestrator.predict_quantum_hardware(
                rec.request, tenant_id=tenant_id, request_id=job_id
            )
            completed = time.time()
            quantum_seconds = max(0.0, completed - started)
            cost = JobCost(
                quantum_seconds=quantum_seconds,
                shots=1024,  # default; replace with executor.get_execution_metadata().shots
                backend=str(resp.quantum_mode_used) if resp.quantum_mode_used else None,
                estimated_cost_usd=estimate_cost(quantum_seconds),
            )
            self.usage.record(tenant_id, cost)
            rec = rec.model_copy(update={
                "status": JobStatus.COMPLETED,
                "completed_at": completed,
                "response": resp,
                "cost": cost,
            })
        except Exception as e:
            log.exception("job_failed", extra={"tenant_id": tenant_id, "job_id": job_id})
            rec = rec.model_copy(update={
                "status": JobStatus.FAILED,
                "completed_at": time.time(),
                "error": ErrorResponse(
                    code=ErrorCode.INTERNAL,
                    message=f"{type(e).__name__}: {e}",
                    request_id=job_id,
                ),
            })
        self._records[(tenant_id, job_id)] = rec
        await persist_job(rec, self.root)
        # Best-effort webhook delivery; failures don't fail the job.
        if rec.callback_url:
            await self._deliver_webhook(rec)

    async def _deliver_webhook(self, rec: JobRecord) -> None:
        """Best-effort POST to the tenant's callback_url with the JobRecord.
        Failures are logged + recorded on the JobRecord but never raised."""
        if not rec.callback_url:
            return
        try:
            import httpx
        except ImportError:
            log.warning("httpx not available; skipping webhook delivery")
            return
        payload = rec.model_dump_json()
        status_str = "delivered"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    rec.callback_url,
                    content=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-Hetqml-Job-Id": rec.job_id,
                        "X-Hetqml-Tenant-Id": rec.tenant_id,
                    },
                )
                if r.status_code >= 400:
                    status_str = f"failed:{r.status_code}"
        except Exception as e:
            status_str = f"failed:{type(e).__name__}"
            log.warning("webhook_delivery_failed",
                        extra={"job_id": rec.job_id, "error": str(e)})
        # Record delivery status without failing the job.
        updated = rec.model_copy(update={"callback_status": status_str})
        self._records[(rec.tenant_id, rec.job_id)] = updated
        await persist_job(updated, self.root)
