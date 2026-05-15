"""All four PredictMethod paths: classical, quantum_strict, quantum_preferred, auto.

Combined with three executor states: quantum-simulator-available,
quantum-unavailable, ibm-hardware-mode.
"""
from __future__ import annotations

import pytest

from service.tests.conftest import make_fake_orchestrator


@pytest.fixture
def app_no_quantum(app_factory):
    async def builder(app, settings, tracker):
        orch, qe = make_fake_orchestrator(quantum_simulator=False, ibm_reachable=False)
        app.state.orchestrator = orch
        app.state.entity_resolver = orch.resolver
        app.state.quantum_executor = qe
        app.state.embedder = type("E", (), {"num_entities": 0, "entity_to_id": {}})()
    return app_factory(state_builder=builder)


@pytest.fixture
def client_no_quantum(app_no_quantum):
    from fastapi.testclient import TestClient
    with TestClient(app_no_quantum) as c:
        yield c


@pytest.fixture
def app_ibm_hw(app_factory):
    async def builder(app, settings, tracker):
        orch, qe = make_fake_orchestrator(
            quantum_simulator=True, ibm_reachable=True, is_ibm_hardware_mode=True
        )
        app.state.orchestrator = orch
        app.state.entity_resolver = orch.resolver
        app.state.quantum_executor = qe
        app.state.embedder = type("E", (), {"num_entities": 4, "entity_to_id": {}})()
    return app_factory(state_builder=builder)


@pytest.fixture
def client_ibm_hw(app_ibm_hw):
    from fastapi.testclient import TestClient
    with TestClient(app_ibm_hw) as c:
        yield c


def _body(method: str) -> dict:
    return {"drug_id": "DB00001", "disease_id": "DOID:1234", "method": method}


def test_classical_method_always_succeeds(client, auth_a):
    r = client.post("/predict", json=_body("classical"), headers=auth_a)
    assert r.status_code == 200
    assert r.json()["method_used"] == "classical"


def test_quantum_strict_with_quantum_available_succeeds(client, auth_a):
    r = client.post("/predict", json=_body("quantum_strict"), headers=auth_a)
    assert r.status_code == 200
    payload = r.json()
    assert payload["method_used"] == "quantum_strict"
    assert payload["fallback_reason"] is None
    assert payload["quantum_mode_used"] in ("simulator", "ibm_heron")


def test_quantum_strict_without_quantum_returns_503(client_no_quantum, auth_a):
    r = client_no_quantum.post("/predict", json=_body("quantum_strict"), headers=auth_a)
    assert r.status_code == 503
    err = r.json()["detail"] if "detail" in r.json() else r.json()
    assert err["code"] == "quantum_unavailable"


def test_quantum_strict_routed_to_ibm_returns_400_use_jobs(client_ibm_hw, auth_a):
    r = client_ibm_hw.post("/predict", json=_body("quantum_strict"), headers=auth_a)
    assert r.status_code == 400
    err = r.json()["detail"] if "detail" in r.json() else r.json()
    assert err["code"] == "use_jobs_endpoint"


def test_quantum_preferred_with_quantum_uses_quantum(client, auth_a):
    r = client.post("/predict", json=_body("quantum_preferred"), headers=auth_a)
    assert r.status_code == 200
    payload = r.json()
    assert payload["method_used"] == "quantum_preferred"
    assert payload["fallback_reason"] is None


def test_quantum_preferred_without_quantum_falls_back_classical(client_no_quantum, auth_a):
    r = client_no_quantum.post("/predict", json=_body("quantum_preferred"), headers=auth_a)
    assert r.status_code == 200
    payload = r.json()
    assert payload["method_used"] == "classical"
    assert "fell back" in (payload["fallback_reason"] or "").lower()


def test_auto_with_quantum_uses_quantum(client, auth_a):
    r = client.post("/predict", json=_body("auto"), headers=auth_a)
    assert r.status_code == 200
    payload = r.json()
    assert payload["method_used"] in ("quantum_preferred", "classical")


def test_auto_without_quantum_uses_classical(client_no_quantum, auth_a):
    r = client_no_quantum.post("/predict", json=_body("auto"), headers=auth_a)
    assert r.status_code == 200
    assert r.json()["method_used"] == "classical"


def test_status_quantum_modes_filtered_by_tenant(client, auth_b):
    """Tenant B has can_use_ibm_hardware=False; ibm_heron must not appear."""
    r = client.get("/status", headers=auth_b)
    assert r.status_code == 200
    modes = r.json()["quantum"]["available_modes"]
    assert "ibm_heron" not in modes
