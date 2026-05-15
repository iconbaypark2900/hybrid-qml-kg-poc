# Runbook: quantum_strict requests failing

**Symptom**: `POST /predict {"method":"quantum_strict"}` returns `503 quantum_unavailable` or `400 use_jobs_endpoint`, but `method=classical` works.

## Decision tree

```
                    quantum_strict failing?
                         │
                ┌────────┴────────┐
              503                400
                │                  │
        check /status       caller asked for IBM hardware via /predict
                │                       │
   +-quantum.available_modes:           │
   │                                     ↓
   ├─ [unavailable]?  → no quantum    use POST /jobs instead.
   │  manifest active or weights      Classical predictions still work
   │  missing → Section A             through /predict.
   │
   └─ [simulator] but request was
      quantum_strict, mode shows
      "simulator" → expected; partner
      treats this as success
```

## Section A: quantum unavailable

```
curl -sH "Authorization: Bearer $TOKEN" $URL/status | jq '{
  quantum_chain: .active_quantum_manifest_chain,
  quantum_comp: (.components | map(select(.name=="quantum_model")))[0]
}'
```

| Output | Cause | Fix |
|---|---|---|
| `quantum_chain: null` | No `LATEST_QUANTUM.txt` | Run `python -m service.scripts.synthesize_quantum_manifest --mode qiskit-mini-train` then `POST /admin/reload` |
| `quantum_chain: {model_id}` but `quantum_comp.state: "degraded"` with detail "weights missing" | `quantum_weights.npz` deleted from disk | Re-run `synthesize_quantum_manifest` to repopulate; never delete artifact files manually |
| `quantum_chain: {model_id}` but `quantum_comp.state: "ok"` and `quantum_strict` still 503 | Predictor cloudpickle failed to load | Check service logs for `qiskit predictor build failed`; likely qiskit version drift — see [`SERVICE_WONT_START.md#crash-on-import`](SERVICE_WONT_START.md) |

## Section B: 400 use_jobs_endpoint

This is **correct behavior**, not a bug. IBM hardware predictions take seconds to minutes and would block the sync `/predict` route past any reasonable HTTP timeout. The caller must use the async path:

```
# Submit
curl -sH "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"request":{"drug_id":"DB00178","disease_id":"DOID:10534","method":"quantum_strict"}}' \
     $URL/jobs | jq -r .job_id > /tmp/job_id

# Poll (or supply callback_url to receive a webhook)
JOB_ID=$(cat /tmp/job_id)
curl -sH "Authorization: Bearer $TOKEN" $URL/jobs/$JOB_ID | jq .status
```

Update partner-facing docs / quickstart if this trips them up frequently.

## Section C: tenant 403

If a specific tenant gets 403 on `quantum_strict`:
```
curl -sH "Authorization: Bearer $TENANT_KEY" $URL/status | jq .tenant.quota
```
Look for `can_use_quantum_strict: false`. Update `secrets/tenants.yaml` and restart.

## After recovery

1. Send 3 test predictions: classical, quantum_preferred, quantum_strict. All should succeed (the third may still 400 if the model is IBM-hardware-bound — that's expected).
2. Check the tenant's `/jobs` queue is being drained — `curl ... $URL/jobs/<recent_id>` should show `status: "completed"`.

## Escalation

If `quantum_strict` is failing for >30 minutes and partners are blocked, page the on-call. Pre-incident: confirm classical path works so demos can continue.
