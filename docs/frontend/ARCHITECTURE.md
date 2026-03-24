# Frontend architecture (Next.js + FastAPI + pipelines)

This document complements [../ARCHITECTURE.md](../ARCHITECTURE.md) (KG, quantum, classical layers) with the **application** layer: how the new UI talks to Python and where long-running work runs.

## Target stack

| Layer | Role | Location / command |
|-------|------|--------------------|
| **UI** | Routes, layout, design system, client state | Planned: `frontend/` (Next.js App Router, TypeScript, Tailwind) |
| **API** | JSON endpoints, validation, orchestrator access | `middleware/api.py` — run with `uvicorn middleware.api:app` |
| **Orchestrator** | Link prediction, entity resolution, mechanism ranking | `middleware/orchestrator.py` — `LinkPredictionOrchestrator` |
| **Pipeline** | Full training/evaluation runs, embeddings, ensemble | `scripts/run_optimized_pipeline.py`, Optuna scripts, `kg_layer/`, `quantum_layer/` |
| **Artifacts** | Results consumed by UI and API | `results/` (`optimized_results_*.json`, `latest_run.csv`, `experiment_history.csv`, …) |

## Request flow

```text
Browser (Next.js)
    │  fetch / server actions
    ▼
FastAPI (CORS in dev; restrict origins in production)
    │  orchestrator.* / future job service
    ▼
Models + KG embedder (in-process)          Long jobs: subprocess or worker
    │                                              │
    ▼                                              ▼
Predictions, status, rankings              results/*.json, CSV, logs
```

## Environment variables (planned)

| Variable | Used by | Example |
|----------|---------|---------|
| `NEXT_PUBLIC_API_URL` | Next.js client / server fetch | `http://localhost:8000` |
| `IBM_Q_TOKEN` / `.env` | Quantum backends | (see quantum config docs) |

Backend continues to use existing config for `LinkPredictionOrchestrator` and quantum settings (`config/quantum_config*.yaml`).

## API surface (current)

Implemented in `middleware/api.py` (OpenAPI at `/docs` when the server is running):

- `GET /status` — health, orchestrator readiness, entity count, model flags
- `POST /predict-link`, `GET /predict-link` — drug–disease link probability
- `POST /batch-predict` — batch predictions
- `POST /ranked-mechanisms` — mechanism-informed candidate ranking (`hypothesis_id`, `disease_id`, `top_k`)

**Not yet exposed as REST:** listing pipeline runs, starting `run_optimized_pipeline.py` as a job, streaming logs. Those belong in a small **job service** or extra routes that wrap subprocess/async tasks and read `results/` (see [MOCKUP_MAP.md](MOCKUP_MAP.md) “Planned” rows).

## CORS and deployment

- Development: API already allows broad CORS; for production, set `allow_origins` to your Next.js origin only.
- Options: run Next and FastAPI on one host behind a reverse proxy (`/api` → uvicorn), or separate origins with strict CORS.

## Legacy UI

- **Streamlit:** `benchmarking/dashboard.py` — keep for internal use until Next.js parity; same `results/` and concepts apply.

## See also

- [MOCKUP_MAP.md](MOCKUP_MAP.md) — screen-by-screen mapping
- [PIPELINE_UI_FLOW.md](PIPELINE_UI_FLOW.md) — journeys through pipeline outputs
