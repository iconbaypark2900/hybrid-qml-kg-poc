# Runbook: IBM Runtime down

**Symptom**: `/status` reports `components.ibm_runtime.state == "unavailable"` or `"degraded"`. `/predict method=quantum_strict` returns 503. `/jobs` submissions for IBM hardware queue but never run.

## Confirm scope

```
curl -sH "Authorization: Bearer $TOKEN" $URL/status | jq '.components | map(select(.name=="ibm_runtime"))'
```

Then independently:
```
.venv/bin/python -c "
from qiskit_ibm_runtime import QiskitRuntimeService
s = QiskitRuntimeService(channel='ibm_quantum', token='$IBM_QUANTUM_TOKEN')
print(s.backends())
"
```

- Both return errors → **IBM-side outage**. Check https://quantum.ibm.com/services for status, https://twitter.com/IBMQuantum for incidents.
- Service errors but direct call works → **service-side credential or network issue** (firewall, expired token).
- Direct call errors but service shows ok → stale cached state in `quantum_executor._ibm_reachable`. Restart with `/admin/reload` (won't help — cached at startup) or full process restart.

## During an IBM outage

**Communicate first**:
1. Update [status page](https://status.example.com) within 5 min: "Quantum hardware predictions degraded; classical predictions unaffected."
2. Notify any tenants with active `/jobs` queues — their jobs are queued but not running.

**Operationally**:
- Classical predictions (`method=classical`) are unaffected — keep serving them.
- `method=quantum_preferred` automatically falls back to classical and tags the response with `fallback_reason: "quantum unavailable; fell back to classical"`. Partners reading `method_used` will see it.
- `method=quantum_strict` returns 503 with `code: quantum_unavailable`. This is correct (audit-mandated honesty).
- Existing `/jobs` queue keeps processing in-memory but won't run; on restart they'll re-queue.

## Restoring after IBM is back

1. Restart the service to clear the cached `_ibm_reachable=False`:
   ```
   systemctl restart hetqml-service
   ```
2. Confirm: `curl $URL/status | jq '.quantum.ibm_runtime_reachable'` → `true`.
3. Drain the queued `/jobs` — they'll start running automatically on the next worker tick.
4. Email any tenant whose webhook callbacks were `failed:*` during the outage so they can re-poll.

## Token rotation (planned, not incident)

1. Generate new token at https://quantum.ibm.com/account
2. Update the `IBM_QUANTUM_TOKEN` secret in your secrets manager
3. `systemctl restart hetqml-service`
4. Confirm `/status` reports `ibm_runtime.state == "ok"`
5. Revoke the old token in IBM console

**Never commit `IBM_QUANTUM_TOKEN` to git or `tenants.yaml`** — it's tenant-of-this-service-as-IBM-customer, not per-tenant of the service.

## Root-cause when this hits

The audit's "silent simulator fallback" finding means a misconfigured `quantum_config_hardware.yaml` can train a model that quietly used simulator. After this incident, audit `actual_execution_mode` in every quantum manifest's `quantum_config.json`:
```
for f in artifacts/runs/QMDL-*/quantum_config.json; do
  jq -r '"\(input_filename): requested=\(.requested_execution_mode) actual=\(.actual_execution_mode) fallback=\(.fallback_used_at_train)"' "$f"
done
```
Any manifest with `fallback_used_at_train: true` was trained on simulator while requesting hardware — re-train before serving from it.
