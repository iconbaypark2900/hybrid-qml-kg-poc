# Runbook: manifest chain broken

**Symptom**: `/status.components.manifest_store.state == "degraded"` or `"unavailable"`. `/status.active_manifest_chain == null`. `/predict` returns 503 `service_not_ready`.

## Diagnose

```
ls -l artifacts/LATEST.txt artifacts/LATEST_QUANTUM.txt
cat artifacts/LATEST.txt
ls artifacts/runs/$(cat artifacts/LATEST.txt)/
```

| What you see | Cause | Fix |
|---|---|---|
| `LATEST.txt: No such file` | First boot, never synthesized | Run `python -m service.scripts.synthesize_manifest_chain` |
| `LATEST.txt` exists but points at a missing dir | `runs/<id>` deleted manually | Re-synthesize OR edit `LATEST.txt` to a valid id (see "Roll back" below) |
| `runs/<id>/` exists but missing `manifest.json` | Synthesizer crashed mid-write | Re-synthesize with `--force` |
| `manifest.json` exists but parent links broken | `parent_embedding` or `parent_feature_pipeline` points at a missing manifest | Re-synthesize the entire chain |
| sha256 mismatch between manifest and on-disk file | Artifact corruption (rare; likely disk error) | Restore from backup OR re-train |

## Roll back to a previous manifest

The runs directory is content-addressed and immutable, so previous chains stay around.

```
ls artifacts/runs/MDL-* -t | head -5
echo "MDL-<previous_id>" > artifacts/LATEST.txt
curl -sX POST -H "Authorization: Bearer $ADMIN_KEY" $URL/admin/reload | jq
```

`/admin/reload` re-reads `LATEST.txt` and rebuilds the orchestrator without restart. Confirm `/status.components.classical_model.state == "ok"` afterward.

## Re-synthesize from scratch

```
python -m service.scripts.synthesize_manifest_chain --force
python -m service.scripts.synthesize_quantum_manifest --mode qiskit-mini-train --force
curl -sX POST -H "Authorization: Bearer $ADMIN_KEY" $URL/admin/reload
```

For real-data training (not the smoke set):
```
python -m service.scripts.train_pipeline --max-entities 0 --cv-folds 5
```

## Verify

After repair:
1. `/status.overall` → `ok` (or `degraded` only for known-unconfigured items like ibm_runtime)
2. `/manifest/active` → returns the chain
3. `/predict` against a known drug-disease pair returns a probability
4. `/evaluations?model_manifest_id=<active>` returns at least the seeded eval

## Prevention

- **Never delete from `artifacts/runs/` manually** — even "old" manifests may be referenced by historical `EvaluationRecord`s
- **Backup `artifacts/runs/`** before any cleanup; treat as immutable append-only
- **Don't edit `manifest.json` files by hand** — rerun the synthesizer instead
- **Tenant pins survive across rebuilds** — if a tenant has `pinned_classical_model_id` and you delete that manifest, the tenant's responses will reference a phantom id. Either keep pinned manifests forever or notify the tenant before removal.

## Escalation

If multiple manifest dirs show sha256 mismatches simultaneously → suspect filesystem corruption. Stop the service, fsck the volume, restore from backup before resuming.
