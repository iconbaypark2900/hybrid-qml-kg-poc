"""Redis-backed job queue (env-gated).

Activation:
    HETQML_JOBS_BACKEND=redis
    HETQML_REDIS_URL=redis://localhost:6379/0
    HETQML_JOBS_QUEUE=hetqml-jobs

If either env is unset OR the redis client lib is missing, the in-memory
JobQueue from service.jobs is used instead.

Persistence layout in Redis:
    hetqml:job:{tenant_id}:{job_id} -> JobRecord JSON (HSET-like via SET)
    hetqml:queue:{queue_name}       -> RPUSH/LPOP queue of (tenant_id, job_id)

Tenant isolation: every key is namespaced by tenant_id, and `get(tenant, ...)`
refuses to return a record from another tenant.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Optional

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


def is_enabled() -> bool:
    return (
        os.environ.get("HETQML_JOBS_BACKEND", "").lower() == "redis"
        and bool(os.environ.get("HETQML_REDIS_URL"))
    )


def _new_job_id() -> str:
    return f"job_{uuid.uuid4().hex[:16]}"


class RedisJobQueue:
    """RQ-style queue without RQ — simple Redis list + per-job key.

    For full RQ semantics (retry policies, dashboards) swap this for `rq.Queue`.
    The shape mirrors `service.jobs.JobQueue` so call sites are interchangeable.
    """

    def __init__(self, root: Path, usage_store: Optional[UsageStore] = None):
        self.root = root
        self.usage = usage_store or UsageStore(root)
        self._workers: list[asyncio.Task] = []
        self._stopped = asyncio.Event()
        self._redis = None
        self._queue_name = os.environ.get("HETQML_JOBS_QUEUE", "hetqml-jobs")
        self._connect()

    def _connect(self) -> None:
        try:
            import redis  # type: ignore[import-not-found]
        except ImportError:
            log.warning(
                "HETQML_JOBS_BACKEND=redis but redis-py not installed; "
                "fallback to in-memory queue. Install: pip install redis>=5.0"
            )
            return
        try:
            url = os.environ["HETQML_REDIS_URL"]
            self._redis = redis.Redis.from_url(
                url, decode_responses=True, socket_timeout=5,
            )
            self._redis.ping()
            log.info("jobs queue: redis backend at %s, queue=%s",
                     url.split("@")[-1], self._queue_name)
        except Exception as e:
            log.warning("redis connect failed: %s; falling back to in-memory queue", e)
            self._redis = None

    @property
    def available(self) -> bool:
        return self._redis is not None

    def _job_key(self, tenant_id: str, job_id: str) -> str:
        return f"hetqml:job:{tenant_id}:{job_id}"

    def _queue_key(self) -> str:
        return f"hetqml:queue:{self._queue_name}"

    async def submit(
        self,
        tenant: Tenant,
        req: PredictRequest,
        callback_url: Optional[str] = None,
    ) -> JobRecord:
        if not tenant.quota.can_use_ibm_hardware:
            raise PermissionError("tenant lacks can_use_ibm_hardware quota")
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
        if self._redis is None:
            raise RuntimeError("redis not connected; should have used in-memory fallback")
        # Store + enqueue atomically via pipeline.
        pipe = self._redis.pipeline()
        pipe.set(self._job_key(record.tenant_id, record.job_id), record.model_dump_json())
        pipe.rpush(self._queue_key(), f"{record.tenant_id}|{record.job_id}")
        pipe.execute()
        return record

    async def get(self, tenant: Tenant, job_id: str) -> Optional[JobRecord]:
        if self._redis is None:
            return None
        raw = self._redis.get(self._job_key(tenant.tenant_id, job_id))
        if raw is None:
            return None
        rec = JobRecord.model_validate_json(raw)
        if rec.tenant_id != tenant.tenant_id:
            log.warning("tenant isolation violation in redis lookup; refusing")
            return None
        return rec

    async def start_workers(self, n: int, orchestrator) -> None:
        if self._redis is None:
            raise RuntimeError("redis not connected; cannot start workers")
        for i in range(n):
            t = asyncio.create_task(self._worker_loop(i, orchestrator))
            self._workers.append(t)

    async def stop(self, timeout: float = 5.0) -> None:
        self._stopped.set()
        for t in self._workers:
            t.cancel()
        for t in self._workers:
            try:
                await asyncio.wait_for(t, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._workers.clear()

    async def _worker_loop(self, worker_id: int, orchestrator) -> None:
        log.info("redis_worker_started", extra={"worker_id": worker_id})
        while not self._stopped.is_set():
            try:
                # Blocking pop with timeout so we can check _stopped.
                item = await asyncio.to_thread(
                    self._redis.blpop, self._queue_key(), timeout=1.0,
                )
            except Exception as e:
                log.warning("blpop failed: %s", e)
                await asyncio.sleep(1.0)
                continue
            if item is None:
                continue
            _key, payload = item
            try:
                tenant_id, job_id = payload.split("|", 1)
            except ValueError:
                log.warning("malformed queue item: %s", payload)
                continue
            await self._run_job(tenant_id, job_id, orchestrator)
        log.info("redis_worker_stopped", extra={"worker_id": worker_id})

    async def _run_job(self, tenant_id: str, job_id: str, orchestrator) -> None:
        raw = self._redis.get(self._job_key(tenant_id, job_id))
        if raw is None:
            log.warning("job missing in redis: %s/%s", tenant_id, job_id)
            return
        rec = JobRecord.model_validate_json(raw)
        started = time.time()
        rec = rec.model_copy(update={
            "status": JobStatus.RUNNING, "started_at": started,
        })
        self._redis.set(self._job_key(tenant_id, job_id), rec.model_dump_json())

        try:
            resp = await orchestrator.predict_quantum_hardware(
                rec.request, tenant_id=tenant_id, request_id=job_id,
            )
            completed = time.time()
            quantum_seconds = max(0.0, completed - started)
            cost = JobCost(
                quantum_seconds=quantum_seconds, shots=1024,
                backend=str(resp.quantum_mode_used) if resp.quantum_mode_used else None,
                estimated_cost_usd=estimate_cost(quantum_seconds),
            )
            self.usage.record(tenant_id, cost)
            rec = rec.model_copy(update={
                "status": JobStatus.COMPLETED, "completed_at": completed,
                "response": resp, "cost": cost,
            })
        except Exception as e:
            log.exception("job_failed", extra={"tenant_id": tenant_id, "job_id": job_id})
            rec = rec.model_copy(update={
                "status": JobStatus.FAILED, "completed_at": time.time(),
                "error": ErrorResponse(
                    code=ErrorCode.INTERNAL,
                    message=f"{type(e).__name__}: {e}",
                    request_id=job_id,
                ),
            })
        self._redis.set(self._job_key(tenant_id, job_id), rec.model_dump_json())

        if rec.callback_url:
            await self._deliver_webhook(rec)

    async def _deliver_webhook(self, rec: JobRecord) -> None:
        if not rec.callback_url:
            return
        try:
            import httpx
        except ImportError:
            return
        status_str = "delivered"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.post(
                    rec.callback_url, content=rec.model_dump_json(),
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
        updated = rec.model_copy(update={"callback_status": status_str})
        self._redis.set(self._job_key(rec.tenant_id, rec.job_id), updated.model_dump_json())


def maybe_build_queue(root: Path, usage_store: Optional[UsageStore] = None):
    """Returns a RedisJobQueue if env-enabled and connected, else None."""
    if not is_enabled():
        return None
    q = RedisJobQueue(root, usage_store=usage_store)
    if not q.available:
        return None
    return q
