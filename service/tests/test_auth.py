"""Auth middleware: missing token, malformed, unknown key, valid key."""
from __future__ import annotations


def test_missing_authorization_header_returns_401(client):
    r = client.get("/status")
    assert r.status_code == 401
    body = r.json()
    assert body["code"] == "auth_missing"


def test_malformed_authorization_header_returns_401(client):
    r = client.get("/status", headers={"Authorization": "NotBearer abc"})
    assert r.status_code == 401
    assert r.json()["code"] == "auth_missing"


def test_empty_bearer_token_returns_401(client):
    r = client.get("/status", headers={"Authorization": "Bearer "})
    assert r.status_code == 401


def test_unknown_api_key_returns_401(client):
    r = client.get("/status", headers={"Authorization": "Bearer this-key-is-not-registered"})
    assert r.status_code == 401
    assert r.json()["code"] == "auth_invalid"


def test_valid_key_returns_200(client, auth_a):
    r = client.get("/status", headers=auth_a)
    assert r.status_code == 200
    assert r.json()["tenant"]["tenant_id"] == "a"


def test_public_paths_skip_auth(client):
    for path in ("/healthz", "/openapi.json", "/docs"):
        r = client.get(path)
        assert r.status_code in (200, 307), f"{path}: {r.status_code}"


def test_system_legacy_tenant_cannot_authenticate(client):
    """Legacy/system tenants exist for migration data; no real key resolves to them."""
    # The legacy tenant's api_key_sha256 is "0"*64 — no plaintext hashes to that.
    # Any random string fails:
    r = client.get("/status", headers={"Authorization": "Bearer x"})
    assert r.status_code == 401


def test_unloaded_tenant_store_returns_503_not_401(app_factory, auth_a):
    """Regression: during the lifespan startup window the placeholder
    TenantStore is loaded=False. AuthMiddleware must return 503 (try again
    later) instead of 401 (your key is bad), so a valid key isn't rejected
    in the boot race."""
    from fastapi.testclient import TestClient
    from service.tenants import TenantStore

    async def builder(app, settings, tracker):
        # Deliberately leave tenant_store as the placeholder (loaded=False)
        # to simulate a request that lands during the boot window.
        from service.tests.conftest import make_fake_orchestrator
        orch, qe = make_fake_orchestrator()
        app.state.orchestrator = orch
        app.state.entity_resolver = orch.resolver
        app.state.quantum_executor = qe
        app.state.embedder = type("E", (), {
            "num_entities": 4, "entity_to_id": {},
        })()
        # Force unloaded state
        app.state.tenant_store = TenantStore(tenants=[], loaded=False)

    app = app_factory(state_builder=builder)
    with TestClient(app) as c:
        r = c.get("/status", headers=auth_a)
    assert r.status_code == 503
    assert r.json()["code"] == "service_not_ready"


def test_cors_preflight_options_bypasses_auth(client):
    """OPTIONS preflights from the browser arrive without Authorization headers
    on purpose — AuthMiddleware must let them through so CORSMiddleware can
    answer with the access-control-* headers. Regression: blocking these with
    401 produces a 'Failed to fetch' on the browser side because the actual
    request is never sent."""
    r = client.options(
        "/status",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,x-request-id",
        },
    )
    assert r.status_code == 200, r.text
    # CORS layer answered, not auth layer
    assert "access-control-allow-origin" in r.headers
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"
