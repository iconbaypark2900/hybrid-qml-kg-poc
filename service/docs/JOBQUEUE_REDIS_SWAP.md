# Replacing the in-memory JobQueue with Redis/RQ

The current [`service/jobs.py`](../jobs.py) `JobQueue` is single-process,
in-memory + disk-persisted. It works for low concurrency and survives a
single-process restart, but it can't run with `uvicorn --workers > 1`
because each worker would have its own queue and submitted jobs would
only be visible to the worker that received the POST.

This is the largest remaining backend gap before scaling past a handful of
concurrent IBM hardware predictions per service instance.

## What changes

### Dependencies (add to `requirements.txt`)

```
redis>=5.0
rq>=1.16
```

Optional but useful:

```
rq-dashboard>=0.7      # web UI for queue inspection
```

### Settings (`service/settings.py`)

Add:

```python
class Settings(BaseModel):
    ...
    redis_url: str = "redis://localhost:6379/0"
    job_queue_name: str = "hetqml-jobs"
    job_default_timeout_seconds: int = 600
    rq_worker_concurrency: int = 2
```

Read from env: `HETQML_REDIS_URL`, `HETQML_JOB_QUEUE_NAME`.

### Replacement queue (`service/jobs_redis.py`)

New module that exposes the same `JobQueue` interface as the in-memory
version (`submit`, `get`, `start_workers`, `stop`) but backed by:

- `redis.Redis.from_url(settings.redis_url)` for the connection
- `rq.Queue(name=settings.job_queue_name, connection=...)` for enqueue
- `JobRecord` serialized into a Redis hash keyed by `(tenant_id, job_id)`
  to preserve tenant isolation (RQ's default queues are global)
- A worker entrypoint module (`service.scripts.run_job_worker`) that runs
  `rq.Worker([queue]).work()` in a separate process

### App wiring (`service/app.py`)

Replace:

```python
queue = JobQueue(settings.artifacts_dir)
```

with:

```python
from .jobs_redis import RedisJobQueue
queue = RedisJobQueue(settings)
```

The lifespan no longer calls `queue.start_workers(n, orchestrator)` —
workers run as separate processes via:

```bash
python -m service.scripts.run_job_worker --queue hetqml-jobs --workers 2
```

This decouples worker lifecycle from the API process and unlocks
`uvicorn --workers > 1`.

### Tenant isolation contract

The current in-memory `JobQueue.get(tenant, job_id)` checks
`rec.tenant_id != tenant.tenant_id` and returns `None` to prevent
cross-tenant leakage. The Redis-backed version must do the same: keys are
namespaced (`hetqml:job:{tenant_id}:{job_id}`) and the lookup refuses any
key from a different tenant.

A `tenant_id` mismatch in the queue payload is a security incident — log
+ alert, don't return the record.

### Worker → orchestrator path

The orchestrator can't be cloudpickled across processes (it holds the
embedder, the classical model, the loaded VQC). Workers re-build it from
the active manifest chain at startup, the same way the API process does
in lifespan. This means workers also need `IBM_QUANTUM_TOKEN`,
`HETQML_TENANTS_PATH`, etc.

### Disk-backed `JobRecord` history

Keep [`service/persistence.py::persist_job`](../persistence.py) as-is —
the JSONL on disk is the audit trail, the Redis hash is the live cache.
After a job reaches a terminal state, write to disk and optionally evict
from Redis after 24h via `EXPIRE`.

## Migration plan

1. Land Redis + RQ in `requirements.txt`, deployment can install but the
   service still uses in-memory queue (no functional change).
2. Implement `RedisJobQueue` next to `JobQueue` — duck-typed compatible.
3. Add `Settings.use_redis_jobqueue` flag (default `False`).
4. Test: stand up a local Redis, set the flag, run the existing
   `test_jobs.py` tests against the Redis-backed queue (parametrize the
   `queue` fixture).
5. Remove the flag and delete the in-memory `JobQueue` once green in
   staging for two weeks.

## What can be deferred

- `rq-dashboard` for ops visibility (nice to have, not blocking)
- Job priority queues (`high`/`default`/`low`) — single queue is fine for
  v1; partition by tenant_id later if a noisy tenant is starving others.
- Retry/back-off policies for failed quantum jobs — RQ has these built-in
  via `Retry(max=3)`; wire them up after the basic swap is stable.
- Sentinel-based Redis HA — single-instance Redis is fine until ops
  requires it.

## Why this is deferred from v1

Replacing the queue requires:

- A Redis instance in the deployment topology (operational lift)
- Multi-process worker management (deployment lift)
- Re-running the full `test_jobs.py` matrix against the new backend
  (test lift)

The current single-process queue is a known limitation, not a bug. It
serves the demo + early partner workloads correctly and the response
contract (`JobRecord` shape) is unchanged by the swap. When you actually
need multi-worker or true horizontal scaling, this doc becomes the
implementation guide.
