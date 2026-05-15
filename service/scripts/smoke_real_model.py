"""End-to-end smoke test: boot the FastAPI app against the real model files
in models/ and data/, hit /healthz, /status, and /predict.

Verifies that _default_orchestrator_builder succeeds against:
  - models/classical_logisticregression.joblib
  - models/scaler.joblib
  - data/entity_embeddings.npy
  - data/entity_ids.json

Usage:
    python -m service.scripts.smoke_real_model
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from service.app import create_app  # noqa: E402
from service.settings import Settings  # noqa: E402
from service.tenants import generate_api_key, hash_api_key  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Persistent artifacts dir to use (default: tempdir, "
             "i.e. no manifest chain). Pass <repo>/artifacts to test against "
             "a real chain produced by synthesize_manifest_chain.",
    )
    args = p.parse_args(argv)
    return _run(args.artifacts_dir)


def _run(artifacts_dir_override: Optional[Path]) -> int:
    plaintext, sha = generate_api_key()
    print(f"[smoke] generated ephemeral API key (sha256={sha[:12]}...)")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        tenants_yaml = tmp / "tenants.yaml"
        tenants_yaml.write_text(
            yaml.safe_dump({
                "tenants": [{
                    "tenant_id": "smoke",
                    "name": "Smoke test tenant",
                    "api_key_sha256": sha,
                    "created_at": time.time(),
                    "quota": {
                        "requests_per_minute": 60,
                        "requests_per_day": 1000,
                        "can_use_quantum_strict": False,
                        "can_use_ibm_hardware": False,
                        "max_batch_size": 50,
                    },
                }]
            }),
            encoding="utf-8",
        )
        print(f"[smoke] tenants.yaml written to {tenants_yaml}")

        if artifacts_dir_override is not None:
            artifacts_dir = artifacts_dir_override.resolve()
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            print(f"[smoke] using persistent artifacts_dir: {artifacts_dir}")
        else:
            artifacts_dir = tmp / "artifacts"
            artifacts_dir.mkdir()
            print(f"[smoke] using ephemeral artifacts_dir: {artifacts_dir}")

        settings = Settings(
            version="0.1.0-smoke",
            git_sha="smoke",
            config_hash="smoke",
            repo_root=REPO_ROOT,
            artifacts_dir=artifacts_dir,
            legacy_models_dir=REPO_ROOT / "models",
            legacy_results_dir=REPO_ROOT / "results",
            config_dir=REPO_ROOT / "config",
            tenants_path=tenants_yaml,
            tenants_example_path=tmp / "nonexistent.yaml",
            cors_origins=["http://localhost:3000"],
            thread_pool_workers=2,
            process_pool_workers=1,
            job_workers=1,
            log_level="INFO",
            log_format="text",
        )

        app = create_app(settings=settings)  # state_builder=None → real loader
        print("[smoke] entering TestClient lifespan...")
        try:
            with TestClient(app) as client:
                _run_assertions(client, plaintext)
        except Exception as e:
            print(f"[smoke] FAIL: {type(e).__name__}: {e}")
            return 2
    print("[smoke] OK")
    return 0


def _run_assertions(client: TestClient, plaintext: str) -> None:
    auth = {"Authorization": f"Bearer {plaintext}"}

    # /healthz — no auth required
    r = client.get("/healthz")
    assert r.status_code == 200, f"/healthz: {r.status_code}"
    print("[smoke] /healthz: OK")

    # /status — auth required, must report ready
    r = client.get("/status", headers=auth)
    assert r.status_code == 200, f"/status: {r.status_code} {r.text}"
    body = r.json()
    print(f"[smoke] /status overall: {body['overall']}")
    print(f"[smoke] /status startup.state: {body['startup']['state']}")
    print(f"[smoke] /status components:")
    for c in body["components"]:
        print(f"          - {c['name']}: {c['state']}"
              + (f" ({c['detail']})" if c.get("detail") else ""))

    if body["overall"] == "degraded":
        # Inspect why
        unavailable = [c for c in body["components"] if c["state"] == "unavailable"]
        for c in unavailable:
            print(f"[smoke] degraded reason: {c['name']} -> {c['detail']}")

    if body["startup"]["state"] != "ready":
        raise RuntimeError(
            f"service did not reach ready state: {body['startup']}"
        )

    # /predict — pick the first known drug + disease from the active manifest's embedder
    # We learn valid IDs by observing the resolver in the orchestrator.
    orch = client.app.state.orchestrator  # type: ignore[attr-defined]
    drug_id = next(iter(sorted(orch.resolver.drug_ids)))
    disease_id = next(iter(sorted(orch.resolver.disease_ids)))
    print(f"[smoke] sampled IDs: drug={drug_id}, disease={disease_id}")

    r = client.post(
        "/predict",
        json={"drug_id": drug_id, "disease_id": disease_id, "method": "classical"},
        headers=auth,
    )
    assert r.status_code == 200, f"/predict: {r.status_code} {r.text}"
    payload = r.json()
    print(f"[smoke] /predict (classical) payload:")
    print(json.dumps(payload, indent=2))
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["method_used"] == "classical"
    assert payload["tenant_id"] == "smoke"

    # If a quantum chain is active, exercise quantum_strict to prove the path works.
    quantum_chain = getattr(client.app.state, "quantum_chain", None)  # type: ignore[attr-defined]
    if quantum_chain is not None:
        print(f"[smoke] quantum chain active: {quantum_chain.model_id}; "
              f"trying quantum_strict")
        # Tenant 'smoke' lacks can_use_quantum_strict by default; promote in-memory
        # for the smoke run so we exercise the quantum path.
        store = client.app.state.tenant_store  # type: ignore[attr-defined]
        existing = store.get("smoke")
        if existing is not None:
            store._by_id["smoke"] = existing.model_copy(  # type: ignore[attr-defined]
                update={"quota": existing.quota.model_copy(
                    update={"can_use_quantum_strict": True}
                )}
            )
            store._by_sha[existing.api_key_sha256] = store._by_id["smoke"]  # type: ignore[attr-defined]

        r = client.post(
            "/predict",
            json={"drug_id": drug_id, "disease_id": disease_id, "method": "quantum_strict"},
            headers=auth,
        )
        assert r.status_code == 200, f"/predict quantum_strict: {r.status_code} {r.text}"
        qpayload = r.json()
        print(f"[smoke] /predict (quantum_strict) payload:")
        print(json.dumps(qpayload, indent=2))
        assert qpayload["method_used"] == "quantum_strict"
        assert qpayload["quantum_mode_used"] in ("simulator", "ibm_heron")
        assert qpayload["manifest_chain"]["model_id"] == quantum_chain.model_id, (
            "quantum response should reference the quantum manifest chain, not classical"
        )


if __name__ == "__main__":
    raise SystemExit(main())
