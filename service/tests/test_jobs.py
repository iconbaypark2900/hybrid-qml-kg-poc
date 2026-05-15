"""Async job queue: submit, poll, tenant isolation, IBM hardware path."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from service.jobs import JobQueue
from service.schemas import (
    JobStatus,
    PredictMethod,
    PredictRequest,
    QuantumMode,
    Tenant,
    TenantQuota,
)
from service.tests.conftest import make_fake_orchestrator


def _tenant(tid: str, ibm: bool = True) -> Tenant:
    return Tenant(
        tenant_id=tid,
        name=tid,
        api_key_sha256="x" * 64,
        quota=TenantQuota(
            requests_per_minute=60,
            requests_per_day=1000,
            can_use_quantum_strict=True,
            can_use_ibm_hardware=ibm,
            max_batch_size=50,
        ),
        created_at=time.time(),
    )


@pytest.fixture
def queue(artifacts_dir: Path):
    return JobQueue(artifacts_dir)


@pytest.fixture
def orch_pair():
    orch, qe = make_fake_orchestrator(
        quantum_simulator=True, ibm_reachable=True, is_ibm_hardware_mode=True
    )
    return orch, qe


def test_submit_returns_queued_record(queue):
    t = _tenant("a")
    req = PredictRequest(drug_id="DB00001", disease_id="DOID:1234",
                         method=PredictMethod.QUANTUM_STRICT)
    rec = asyncio.run(queue.submit(t, req))
    assert rec.status == JobStatus.QUEUED
    assert rec.tenant_id == "a"
    assert rec.request.drug_id == "DB00001"
    assert rec.quantum_mode == QuantumMode.IBM_HERON


def test_submit_rejects_tenant_without_ibm_quota(queue):
    t = _tenant("a", ibm=False)
    req = PredictRequest(drug_id="DB00001", disease_id="DOID:1234")
    with pytest.raises(PermissionError):
        asyncio.run(queue.submit(t, req))


def test_get_returns_record_for_owning_tenant(queue):
    t = _tenant("a")
    req = PredictRequest(drug_id="DB00001", disease_id="DOID:1234")
    rec = asyncio.run(queue.submit(t, req))
    fetched = asyncio.run(queue.get(t, rec.job_id))
    assert fetched is not None
    assert fetched.job_id == rec.job_id


def test_get_returns_none_for_other_tenant(queue):
    t_a = _tenant("a")
    t_b = _tenant("b")
    req = PredictRequest(drug_id="DB00001", disease_id="DOID:1234")
    rec = asyncio.run(queue.submit(t_a, req))
    # Tenant B asking for tenant A's job → None (isolation, not 404 leakage)
    fetched = asyncio.run(queue.get(t_b, rec.job_id))
    assert fetched is None


def test_get_unknown_job_returns_none(queue):
    t = _tenant("a")
    fetched = asyncio.run(queue.get(t, "job_nonexistent"))
    assert fetched is None


def test_worker_executes_queued_job(queue, orch_pair):
    orch, _qe = orch_pair
    t = _tenant("a")
    req = PredictRequest(drug_id="DB00001", disease_id="DOID:1234")

    async def scenario():
        await queue.start_workers(1, orch)
        rec = await queue.submit(t, req)
        # Poll for completion (test should be fast — fake predict is instant)
        for _ in range(50):
            current = await queue.get(t, rec.job_id)
            if current and current.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                return current
            await asyncio.sleep(0.05)
        return None

    final = asyncio.run(scenario())
    assert final is not None, "job didn't reach terminal state"
    assert final.status == JobStatus.COMPLETED
    assert final.response is not None
    assert final.response.method_used == PredictMethod.QUANTUM_STRICT
    assert final.response.quantum_mode_used == QuantumMode.IBM_HERON
    asyncio.run(queue.stop())
