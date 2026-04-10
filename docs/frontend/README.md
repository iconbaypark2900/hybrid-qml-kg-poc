# Frontend & UI/UX upgrade

This folder documents the **Next.js** UI (`frontend/`), **FastAPI** backend (`middleware/api.py`), **data pipelines** (`scripts/run_optimized_pipeline.py` and related), and **Stitch mockups**.

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | End-to-end stack: browser → API → orchestrator → pipeline artifacts |
| [MOCKUP_MAP.md](MOCKUP_MAP.md) | Each mockup HTML → proposed route → API → data source (implemented vs planned) |
| [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md) | Quantum Slate tokens, fonts, and where the canonical spec lives |
| [PIPELINE_UI_FLOW.md](PIPELINE_UI_FLOW.md) | User journeys: which screens consume which pipeline outputs |
| [../planning/FRONTEND_ROLLOUT_PLAN.md](../planning/FRONTEND_ROLLOUT_PLAN.md) | Phased rollout: workstreams, checkpoints, tests, definition of done |

**Related:** system-level design remains in [../ARCHITECTURE.md](../ARCHITECTURE.md). The Streamlit app at `benchmarking/dashboard.py` is kept for internal/legacy use.

## Quick start

1. **Install:** `cd frontend && pnpm install`
2. **One command (recommended):** from repo root, `./scripts/dev_stack.sh` — starts FastAPI on `http://127.0.0.1:8000` and Next.js with `NEXT_PUBLIC_API_URL` set to match (see script for `API_PORT` / `FRONTEND_PORT`).
3. **Manual:** **API:** from repo root, `uvicorn middleware.api:app --reload --host 127.0.0.1 --port 8000` (uvicorn default). **Next.js:** `cd frontend && pnpm dev` — set `NEXT_PUBLIC_API_URL` in `.env.local` to the same host/port as the API (`frontend/.env.example` uses `8000`).
4. **Production build:** `cd frontend && pnpm build && pnpm start`
5. **Pipeline (CLI):** `python scripts/run_optimized_pipeline.py` — see [.cursor/rules/pipeline-scripts.mdc](../../.cursor/rules/pipeline-scripts.mdc) and [../reference/COMMAND_REFERENCE.md](../reference/COMMAND_REFERENCE.md)

## Mockup bundle location

Static HTML references (Stitch / design handoff):

`stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/<screen>/code.html`
