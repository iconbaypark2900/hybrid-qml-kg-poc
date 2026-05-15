"""Shared fixtures for service/ tests.

Strategy: every test gets a temp artifacts dir and a tenant store with two
known API keys (tenant 'a' with quantum quota, tenant 'b' without).
The orchestrator is a fake that doesn't need the kept ML library — actual
ML wiring is exercised separately in legacy_adapter tests (when present).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from service.app import create_app
from service.observability import StartupTracker
from service.orchestration import (
    EntityResolver,
    ExecutionRouter,
    Orchestrator,
    PredictorBundle,
)
from service.schemas import (
    ManifestChain,
    QuantumMode,
    Tenant,
    TenantQuota,
)
from service.settings import Settings
from service.tenants import TenantStore


# --- Tenant fixtures ------------------------------------------------------


TENANT_A_KEY = "tenant-a-key-test-only"
TENANT_B_KEY = "tenant-b-key-test-only"
TENANT_BAD_KEY = "this-key-is-not-registered"


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


@pytest.fixture
def tenant_a() -> Tenant:
    return Tenant(
        tenant_id="a",
        name="Tenant A (full quantum)",
        api_key_sha256=_sha(TENANT_A_KEY),
        quota=TenantQuota(
            requests_per_minute=120,
            requests_per_day=10000,
            can_use_quantum_strict=True,
            can_use_ibm_hardware=True,
            max_batch_size=50,
        ),
        created_at=time.time(),
    )


@pytest.fixture
def tenant_b() -> Tenant:
    return Tenant(
        tenant_id="b",
        name="Tenant B (classical only)",
        api_key_sha256=_sha(TENANT_B_KEY),
        quota=TenantQuota(
            requests_per_minute=60,
            requests_per_day=5000,
            can_use_quantum_strict=False,
            can_use_ibm_hardware=False,
            max_batch_size=10,
        ),
        created_at=time.time(),
    )


@pytest.fixture
def tenants_yaml(tmp_path: Path, tenant_a: Tenant, tenant_b: Tenant) -> Path:
    payload = {
        "tenants": [
            {"tenant_id": "a", "name": tenant_a.name,
             "api_key_sha256": tenant_a.api_key_sha256,
             "created_at": tenant_a.created_at,
             "quota": tenant_a.quota.model_dump()},
            {"tenant_id": "b", "name": tenant_b.name,
             "api_key_sha256": tenant_b.api_key_sha256,
             "created_at": tenant_b.created_at,
             "quota": tenant_b.quota.model_dump()},
        ]
    }
    p = tmp_path / "tenants.yaml"
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return p


# --- Settings + artifacts fixtures ---------------------------------------


@pytest.fixture
def artifacts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    return d


@pytest.fixture
def settings(tmp_path: Path, tenants_yaml: Path, artifacts_dir: Path) -> Settings:
    return Settings(
        version="0.1.0-test",
        git_sha="testsha",
        config_hash="testhash",
        repo_root=tmp_path,
        artifacts_dir=artifacts_dir,
        legacy_models_dir=tmp_path / "models",
        legacy_results_dir=tmp_path / "results",
        config_dir=tmp_path / "config",
        tenants_path=tenants_yaml,
        tenants_example_path=tmp_path / "nonexistent.yaml",
        cors_origins=["http://localhost:3000"],
        thread_pool_workers=2,
        process_pool_workers=1,
        job_workers=1,
        log_level="WARNING",
        log_format="text",
    )


# --- Fake orchestrator ---------------------------------------------------


class _FakeQuantumExecutor:
    def __init__(self, simulator: bool = True, ibm: bool = False, gpu: bool = False):
        self._ibm_initialized = True
        self._ibm_reachable = ibm
        self._simulator_available = simulator
        self._gpu_simulator_available = gpu
        self._last_calibration_ts = None
        self._is_ibm_hardware = ibm

    def current_mode(self) -> QuantumMode:
        if self._is_ibm_hardware and self._ibm_reachable:
            return QuantumMode.IBM_HERON
        if self._simulator_available:
            return QuantumMode.SIMULATOR
        return QuantumMode.UNAVAILABLE

    def is_ibm_hardware_mode(self) -> bool:
        return self._is_ibm_hardware


def _fake_classical_predict(X: np.ndarray) -> np.ndarray:
    # Stable deterministic "score" — use mean of features bounded to (0, 1).
    arr = np.atleast_2d(X)
    raw = float(np.tanh(np.mean(arr)))
    return np.array([(raw + 1) / 2])


def _fake_quantum_predict(X: np.ndarray) -> np.ndarray:
    arr = np.atleast_2d(X)
    raw = float(np.tanh(np.std(arr)))
    return np.array([(raw + 1) / 2 * 0.9])


def _fake_feature_prep(drug_id: str, disease_id: str) -> np.ndarray:
    seed = sum(ord(c) for c in drug_id + disease_id)
    rng = np.random.default_rng(seed)
    return rng.normal(size=(1, 8))


def make_fake_orchestrator(
    quantum_simulator: bool = True,
    ibm_reachable: bool = False,
    is_ibm_hardware_mode: bool = False,
    drug_ids: Optional[set[str]] = None,
    disease_ids: Optional[set[str]] = None,
    chain: Optional[ManifestChain] = None,
) -> tuple[Orchestrator, _FakeQuantumExecutor]:
    drugs = drug_ids if drug_ids is not None else {"DB00001", "DB00002"}
    diseases = disease_ids if disease_ids is not None else {"DOID:1234", "DOID:5678"}
    qe = _FakeQuantumExecutor(
        simulator=quantum_simulator, ibm=ibm_reachable, gpu=False
    )
    qe._is_ibm_hardware = is_ibm_hardware_mode
    bundle = PredictorBundle(
        classical_predict=_fake_classical_predict,
        quantum_predict=_fake_quantum_predict if quantum_simulator or ibm_reachable else None,
    )
    router = ExecutionRouter(bundle, qe)
    chain = chain or ManifestChain(
        embedding_id="emb-test", feature_pipeline_id="fp-test", model_id="mdl-test"
    )
    resolver = EntityResolver(drugs, diseases)
    orch = Orchestrator(
        resolver=resolver,
        router=router,
        feature_prep=_fake_feature_prep,
        classical_chain=chain,
        quantum_chain=None,
        ibm_predict_fn=_fake_quantum_predict,
    )
    return orch, qe


# --- App fixtures --------------------------------------------------------


@pytest.fixture
def app_factory(settings: Settings):
    """Returns a function `make_app(state_builder=None)` for tests to customize."""

    def _make(state_builder=None):
        if state_builder is None:
            async def _default(app, settings_, tracker):
                orch, qe = make_fake_orchestrator()
                app.state.orchestrator = orch
                app.state.entity_resolver = orch.resolver
                app.state.quantum_executor = qe
                app.state.embedder = type("E", (), {
                    "num_entities": 4,
                    "entity_to_id": {"DB00001": 0, "DB00002": 1,
                                     "DOID:1234": 2, "DOID:5678": 3},
                })()
            state_builder = _default
        return create_app(settings=settings, state_builder=state_builder)

    return _make


@pytest.fixture
def app(app_factory) -> FastAPI:
    return app_factory()


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_a() -> dict:
    return {"Authorization": f"Bearer {TENANT_A_KEY}"}


@pytest.fixture
def auth_b() -> dict:
    return {"Authorization": f"Bearer {TENANT_B_KEY}"}
