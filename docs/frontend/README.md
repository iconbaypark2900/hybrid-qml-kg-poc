# Frontend & UI/UX upgrade

This folder documents how the **Next.js** UI (planned), **FastAPI** backend (`middleware/api.py`), **data pipelines** (`scripts/run_optimized_pipeline.py` and related), and **Stitch mockups** fit together.

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | End-to-end stack: browser → API → orchestrator → pipeline artifacts |
| [MOCKUP_MAP.md](MOCKUP_MAP.md) | Each mockup HTML → proposed route → API → data source (implemented vs planned) |
| [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md) | Quantum Slate tokens, fonts, and where the canonical spec lives |
| [PIPELINE_UI_FLOW.md](PIPELINE_UI_FLOW.md) | User journeys: which screens consume which pipeline outputs |
| [../planning/FRONTEND_ROLLOUT_PLAN.md](../planning/FRONTEND_ROLLOUT_PLAN.md) | Phased rollout: workstreams, checkpoints, tests, definition of done |

**Related:** system-level design remains in [../ARCHITECTURE.md](../ARCHITECTURE.md). The Streamlit app at `benchmarking/dashboard.py` is the current dashboard until the Next.js app reaches parity.

## Quick start (when `frontend/` exists)

1. **API:** from repo root, `uvicorn middleware.api:app --reload --host 0.0.0.0 --port 8000`
2. **Next.js:** `cd frontend && pnpm dev` (or `npm run dev`) with `NEXT_PUBLIC_API_URL=http://localhost:8000`
3. **Pipeline (CLI):** `python scripts/run_optimized_pipeline.py` — see [.cursor/rules/pipeline-scripts.mdc](../../.cursor/rules/pipeline-scripts.mdc) and [../reference/COMMAND_REFERENCE.md](../reference/COMMAND_REFERENCE.md)

## Mockup bundle location

Static HTML references (Stitch / design handoff):

`stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/<screen>/code.html`
