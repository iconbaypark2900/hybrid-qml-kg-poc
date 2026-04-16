# Frontend Completion — Next.js Parity with Streamlit

**Status:** ~40% complete  
**Primary UI:** Streamlit (`benchmarking/dashboard.py`) — still the daily driver  
**Target:** Next.js replaces Streamlit as the primary interface

The Next.js app (`frontend/`) has 13+ routes scaffolded. Several are fully
implemented. Several are stubs or missing critical functionality. This document
tracks what remains.

---

## Route Status

### Fully Implemented (no major gaps)

| Route | What it does |
|-------|-------------|
| `/predict` | Calls `POST /predict`, renders treatment probability score |
| `/experiments` | Reads `GET /runs/latest`, displays result table |
| `/knowledge-graph` | Calls `GET /kg/stats`, displays graph entity/edge counts |
| `/export` | Calls `GET /exports`, lists downloadable result files |
| `/simulation` | Calls `GET /jobs`, shows pipeline job queue |
| `/hypotheses/new` | Mechanism-informed H-001/H-002/H-003 hypothesis ranker |

---

### Incomplete — Needs Work

#### `/quantum`

**Current state:** Reads quantum config and has IBM token verifier, but sections
use the `PagePlaceholder` component — they render a "coming soon" card instead
of real content.

**Missing:**
- Circuit depth and feature map visualization (Qiskit circuit diagram as SVG/image)
- Live qubit count and backend status display
- Kernel computation time history chart
- Toggle between ideal / noisy / hardware execution modes with config preview
- Link to IBM Quantum dashboard for the active backend

**API dependency:** `GET /quantum/config` (exists), `POST /quantum/verify` (exists)

---

#### `/visualization`

**Current state:** 1,300+ lines of TSX — the largest page in the app. Includes
KG browser, entity search, and pair scoring. Uses form input placeholders
suggesting it is ready, but has not been tested against a live API returning
real data.

**Known risks:**
- Entity lookup may fail for names not in the embedding index (no error handling for missing embeddings)
- KG browser requires `GET /kg/graph` which may not be implemented in `middleware/api.py`
- Pair scoring in this view duplicates `/predict` — needs deduplication or separation of concerns

**What to do:**
1. Audit against `GET /kg/graph`, `GET /kg/entity/{name}` in `middleware/api.py`
2. Add embedding-not-found error state with helpful suggestions
3. Add entity autocomplete from known compound/disease lists

---

#### `/analysis/drug-delivery` and `/analysis/next-steps`

**Current state:** Files exist. Content unknown from inspection.

**What to do:** Audit both pages. If they are placeholders, either implement or
remove from the sidebar navigation. Placeholder routes visible in the sidebar
erode user trust.

---

#### `/molecular-design`

**Current state:** Wraps the same `PredictForm` component as `/predict`.
It is a duplicate route.

**Decision needed:** Either differentiate it (e.g. focus on structure-based
input vs name-based input) or remove it and redirect to `/predict`.

---

### Missing Routes — Not Yet Built

These features exist in the Streamlit dashboard but have no Next.js equivalent:

#### Pipeline Job Trigger UI

**What Streamlit has:** A form to start a new pipeline run with configurable
embedding method, dim, epochs, relation, and quantum settings.

**What Next.js has:** `/simulation` shows existing jobs but provides no way
to launch a new one.

**What to build:**
- Form at `/simulation/new` with all pipeline flags exposed as inputs
- `POST /jobs` endpoint in `middleware/api.py` (check if it exists; if not, it needs to be added)
- Job status polling with auto-refresh on `/simulation`
- Link from job list to result detail view

---

#### Per-Prediction MoA Explanation Panel

**What is missing:** When `/predict` or `/visualization` returns a score for
a (compound, disease) pair, there is no breakdown of *why* the model scored
it that way.

**What to build:** After scoring a pair, show:
- The 10 MoA feature values (binding targets, shared targets, pathway genes, etc.)
- Which features contributed most to the score (feature importance from RF/ET)
- A "mechanistic support" indicator: green if `shared_targets > 0`, amber if `similar_compounds_treat > 0`, red if all zero

**API dependency:** `middleware/api.py` needs a `GET /predict/explain` endpoint
that returns MoA feature values alongside the score.

---

#### Benchmark Registry / Run Comparison UI

**What is missing:** `results/benchmark_registry.jsonl` tracks all pipeline
runs, but there is no UI to browse or compare them.

**What to build at `/experiments/compare`:**
- Table of all registered runs with columns: timestamp, relation, embedding method/dim, QSVC config, PR-AUC, ROC-AUC
- Diff view: select two runs, highlight config differences and metric deltas
- Filter by relation, embedding method, execution mode (ideal/noisy/hardware)

**API dependency:** `GET /experiments` (may need extension to return full registry)

---

#### Experiment History Chart

**What is missing:** PR-AUC over time across all runs. The Streamlit dashboard
has a time-series chart. Next.js has no chart component at all.

**What to build:**
- Line chart at `/experiments` showing PR-AUC for best model per run over time
- Separate series for classical-only, QSVC-only, and ensemble
- Tooltip with config details on hover

**Library:** The project already has Tailwind CSS. Add `recharts` or `chart.js`
(or use `@/components/ui/chart` if shadcn chart is already present).

---

## API Gaps in `middleware/api.py`

These endpoints need to be added or verified before the frontend features above
can be built:

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `POST /jobs` | Trigger a new pipeline run | Unknown — check `middleware/api.py` |
| `GET /jobs/{id}` | Poll job status | Unknown |
| `GET /predict/explain` | Return MoA features for a predicted pair | Likely missing |
| `GET /experiments` | Return all registry entries | Likely returns only latest run |
| `GET /kg/graph` | Return graph structure for visualization | Unknown |
| `GET /kg/entity/{name}` | Entity metadata + embedding lookup | Unknown |

---

## Priority Order for Frontend Work

1. **MoA explanation panel** — highest research value, differentiates from any existing tool
2. **Pipeline job trigger** — required before Streamlit can be retired
3. **Benchmark registry UI** — enables experiment comparison without pandas
4. **`/quantum` placeholder sections** — complete or remove
5. **`/analysis/*` audit** — implement or remove from nav
6. **Experiment history chart** — polish, low-risk
7. **`/visualization` live API audit** — verify against real data
8. **`/molecular-design` deduplication** — cleanup

---

## Dev Stack Notes

- Default ports: **8780** (FastAPI) and **3780** (Next.js) — do not use 8000/3000
- Restart Next.js after editing `frontend/.env.local`
- Run both together: `./scripts/dev_stack.sh` from repo root
- Next.js resolves backend via `NEXT_PUBLIC_API_URL` — must match the FastAPI host:port
- Common failure: browser calls one port while uvicorn listens on another → `Failed to fetch`
