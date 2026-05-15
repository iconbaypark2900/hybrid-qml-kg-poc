# Runbook: service won't start

**Symptom**: `python -m service` exits non-zero, OR uvicorn boots but `/healthz` returns connection refused, OR `/healthz` returns 200 but `/status` reports `startup.state == "failed"`.

## Triage (first 5 minutes)

1. **Check process exit**:
   ```
   journalctl -u hetqml-service --since "10 min ago"
   ```
   If exit code 1-2 and the last log line is a Python traceback, jump to **Crash on import** below.

2. **Check `/status` startup field**:
   ```
   curl -sH "Authorization: Bearer $ADMIN_KEY" $URL/status | jq .startup
   ```
   - `state: "failed"` → jump to **Lifespan failure** below
   - `state: "starting"` for >2 min → jump to **Stuck loading** below
   - `state: "ready"` but components unhealthy → see [`MANIFEST_CHAIN_BROKEN.md`](MANIFEST_CHAIN_BROKEN.md)

3. **Check disk + memory**:
   ```
   df -h .
   free -m
   ```
   - `<1GB` free → manifest writes will fail; clean `artifacts/jobs/` and old `runs/`
   - OOM → check `dmesg | tail`; the cloudpickled VQC predictor can be large

## Crash on import

Likely causes, ordered by frequency:
1. **Missing dependency** — pip install drift between dev and prod. Run:
   ```
   .venv/bin/pip check
   ```
2. **Pydantic v1/v2 mismatch** — the service requires Pydantic ≥2.6. `pip show pydantic`.
3. **Qiskit version drift** — the `_build_qiskit_quantum_predictor` cloudpickle is bound to the qiskit-machine-learning version that wrote it. If qiskit was upgraded, the cloudpickle may fail to load. Either rebuild the manifest with the new version or pin qiskit.

## Lifespan failure

`/status.startup.last_step` tells you which lifespan step crashed:

| `last_step` | Likely cause | Fix |
|---|---|---|
| `failed: orchestrator: ...` | classical/quantum manifest broken | [`MANIFEST_CHAIN_BROKEN.md`](MANIFEST_CHAIN_BROKEN.md) |
| `failed: tenant_store: ...` | malformed `secrets/tenants.yaml` | `python -c "import yaml; yaml.safe_load(open('secrets/tenants.yaml'))"` |
| `failed: async_pools: ...` | OS thread/process limit | `ulimit -u`, raise via systemd `LimitNPROC=` |
| `failed: ibm_runtime: ...` | not fatal — service should boot anyway. If it killed startup, check for an unhandled `RuntimeError` in `observability.probe_ibm_runtime` |

## Stuck loading

If `state: "starting"` for >2 minutes:
1. **Cloudpickle load** of a large quantum predictor can take minutes for >12-qubit VQCs. Check `artifacts/runs/<active_quantum_id>/quantum_predictor.cloudpickle` size; if >50MB, bump the lifespan timeout.
2. **PyKEEN loading** the embedding `.npy` from a slow disk. Check `iotop`.
3. **`tenants.yaml` over a network drive**. Move to local disk.

## After recovery

1. Confirm `/status.overall == "ok"` (or only "degraded" for known unconfigured items).
2. Run a smoke `/predict`:
   ```
   curl -sH "Authorization: Bearer $TENANT_KEY" -H "Content-Type: application/json" \
        -d '{"drug_id":"DB00178","disease_id":"DOID:10534","method":"classical"}' \
        $URL/predict | jq .probability
   ```
3. Open an incident retrospective if downtime exceeded the SLO ([`SLO_SLA.md`](../SLO_SLA.md)).

## Escalation

- After 30 min without resolution → escalate per [`ON_CALL_ROTATION.md`](../ON_CALL_ROTATION.md).
- For data-loss risks (artifacts/ corruption, secrets/ leak), page the security on-call immediately.
