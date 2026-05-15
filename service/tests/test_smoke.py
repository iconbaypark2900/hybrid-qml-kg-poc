"""Smoke tests: service starts, /healthz works, /status reports honestly,
/predict happy path with fake orchestrator."""
from __future__ import annotations


def test_healthz_no_auth(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_status_requires_auth(client):
    r = client.get("/status")
    assert r.status_code == 401
    body = r.json()
    assert body["code"] == "auth_missing"


def test_status_with_auth_returns_full_payload(client, auth_a):
    r = client.get("/status", headers=auth_a)
    assert r.status_code == 200
    body = r.json()
    # Top-level honesty fields
    assert body["service_version"] == "0.1.0-test"
    assert body["git_sha"] == "testsha"
    assert body["config_hash"] == "testhash"
    assert body["overall"] in ("ok", "degraded", "loading")
    assert "components" in body
    assert isinstance(body["components"], list)
    assert body["startup"]["state"] in ("starting", "ready", "failed")
    assert body["tenant"]["tenant_id"] == "a"
    # Tenant view never includes the api_key_sha256
    assert "api_key_sha256" not in json_str(body["tenant"])


def test_status_quantum_status_reflects_capabilities(client, auth_a):
    r = client.get("/status", headers=auth_a)
    body = r.json()
    q = body["quantum"]
    assert "available_modes" in q
    assert "default_mode" in q
    assert q["default_mode"] in ("simulator", "ibm_heron", "gpu_simulator", "unavailable")


def test_predict_happy_path(client, auth_a):
    body = {"drug_id": "DB00001", "disease_id": "DOID:1234", "method": "classical"}
    r = client.post("/predict", json=body, headers=auth_a)
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["drug_id"] == "DB00001"
    assert payload["disease_id"] == "DOID:1234"
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["method_requested"] == "classical"
    assert payload["method_used"] == "classical"
    assert payload["fallback_reason"] is None
    assert payload["tenant_id"] == "a"
    assert payload["request_id"]


def test_predict_correlation_id_round_trips(client, auth_a):
    body = {"drug_id": "DB00001", "disease_id": "DOID:1234", "method": "classical"}
    r = client.post("/predict", json=body, headers={**auth_a, "X-Request-Id": "rid-fixed"})
    assert r.status_code == 200
    assert r.headers["X-Request-Id"] == "rid-fixed"
    assert r.json()["request_id"] == "rid-fixed"


def test_predict_unknown_drug_returns_422(client, auth_a):
    body = {"drug_id": "DB99999999", "disease_id": "DOID:1234"}
    r = client.post("/predict", json=body, headers=auth_a)
    assert r.status_code == 422
    err = r.json()["detail"] if "detail" in r.json() else r.json()
    assert err["code"] == "unknown_entity"
    assert err["detail"]["kind"] == "drug"


def json_str(obj):
    import json
    return json.dumps(obj)
