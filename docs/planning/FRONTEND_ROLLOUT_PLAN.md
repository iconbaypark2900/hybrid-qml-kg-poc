# Frontend rollout plan: workstreams, checkpoints, and tests

This document splits the **Next.js + FastAPI + pipeline** UI upgrade into phases with **exit criteria** and **verification** (automated where possible). It complements [../frontend/MOCKUP_MAP.md](../frontend/MOCKUP_MAP.md), [../frontend/PIPELINE_UI_FLOW.md](../frontend/PIPELINE_UI_FLOW.md), and [../frontend/ARCHITECTURE.md](../frontend/ARCHITECTURE.md).

## Principles

1. **Vertical slices** — Each milestone delivers a thin path: **UI + API + data**, not all mockups at once.
2. **Checkpoint = demo + checks** — Merge when automated tests (where they exist) pass and manual criteria are ticked.
3. **Contract first** — For new JSON endpoints, agree on fields and errors before the UI hard-depends on them.

## Workstreams

| Stream | Focus | Depends on |
|--------|--------|------------|
| **A. Design system** | Tailwind tokens, shell layout, fonts (Quantum Slate) | Mockup `code.html`, [../frontend/DESIGN_SYSTEM.md](../frontend/DESIGN_SYSTEM.md) |
| **B. API** | Extend `middleware/api.py`: runs, jobs, exports | Orchestrator, `results/`, subprocess policy |
| **C. Next.js** | Routes from MOCKUP_MAP, components | A + B per slice |
| **D. QA / DX** | Lint, CI, API tests, smoke e2e | C |

---

## Phase 0 — Baseline

**Goal:** Backend and existing tests are stable before the UI rewrite.

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 0.1 | API healthy | FastAPI starts | `uvicorn middleware.api:app` → `GET /status` returns 200 (or documented failure if orchestrator unavailable) |
| 0.2 | Regression suite | No regressions in quantum improvements tests | `python run_tests.py --mode terminal` from repo root |
| 0.3 | Prediction path | Link prediction works end-to-end | `POST /predict-link` or `GET /predict-link` with a known drug/disease pair → 200 and structured body |

---

## Phase 1 — Next.js skeleton + design shell

**Goal:** App shell and navigation match the mockup **look**; pages can be placeholders.

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 1.1 | Build | Next app compiles | `pnpm build` or `npm run build` in `frontend/` |
| 1.2 | Tokens | Colors/surfaces align with Quantum Slate | Manual checklist vs [../frontend/DESIGN_SYSTEM.md](../frontend/DESIGN_SYSTEM.md); optional visual regression later |
| 1.3 | Routing | Nav covers planned routes | Manual: each sidebar destination returns 200 (stub content OK) |

**Automated:** ESLint + TypeScript (`strict`); optional Playwright “home loads, sidebar visible.”

---

## Phase 2 — System status (first vertical slice)

**Goal:** `/system` displays live data from `GET /status` — proves **Next ↔ FastAPI ↔ orchestrator**.

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 2.1 | Data binding | Status fields shown in UI | Integration test with mocked fetch **or** Playwright against local API |
| 2.2 | Resilience | API unreachable does not white-screen | Error UI test |
| 2.3 | CORS (dev) | Browser can call API from Next origin | Manual smoke in dev |

**Regression:** Phase 0 checks.

---

## Phase 3 — Predict + rank

**Goal:** Wire **existing** endpoints to UI (see MOCKUP_MAP “Partial” rows).

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 3.1 | Predictions | `/molecular-design` (or equivalent) calls predict | Assert response matches API schema (`PredictionResponse`) |
| 3.2 | Ranking | Hypothesis flow calls `POST /ranked-mechanisms` | Test with valid `hypothesis_id` / `disease_id` / `top_k` |
| 3.3 | Errors | 400/500 paths show clear messages | API returns structured error; UI displays it |

**Regression:** Phase 0 + `run_tests.py --mode terminal`.

---

## Phase 4 — Experiment overview (read `results/`)

**Goal:** `/experiments` reflects latest pipeline output.

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 4.1 | Backend contract | `GET /runs/latest` (or BFF equivalent) defines “latest” | Unit tests with fixture JSON under `tests/fixtures/` |
| 4.2 | UI parity | Metrics match file contents | Field assertions or snapshot |
| 4.3 | Empty state | No results yet | API + UI handle gracefully |

---

## Phase 5 — Pipeline jobs (async)

**Goal:** Simulation screens trigger `scripts/run_optimized_pipeline.py` via a **job** API; UI polls completion.

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 5.1 | Job lifecycle | create → poll → success/fail | Integration test with **mocked** subprocess or minimal dry-run |
| 5.2 | Safety | No zombie processes; timeouts documented | Code review + test |
| 5.3 | Integration | Completed job updates experiment overview | Smoke after job (optional nightly if slow) |

**Optional smoke:** `python scripts/pipeline_smoke.py` when present — fast path &lt; ~3 min per project conventions.

---

## Phase 6 — Analysis + export

**Goal:** Analysis and export routes; safe access to `results/`.

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 6.1 | Export security | No path traversal | Tests for malicious paths |
| 6.2 | Analysis | Matches fixture outputs | Contract or snapshot tests |

---

## Phase 7 — Knowledge graph + quantum views (later)

**Goal:** `/knowledge-graph`, `/quantum` with new or cached endpoints.

| ID | Checkpoint | Done when | Verification |
|----|------------|-----------|--------------|
| 7.1 | API frozen | Graph/quantum JSON contract documented | Unit + API tests |
| 7.2 | Performance | Large payloads paginated or streamed | Manual or load test |

---

## Cross-cutting definition of done

**API changes**

- [ ] `/docs` (OpenAPI) matches behavior.
- [ ] [../frontend/MOCKUP_MAP.md](../frontend/MOCKUP_MAP.md) updated (status column).

**UI changes**

- [ ] Errors and loading states for async calls.
- [ ] Works against `NEXT_PUBLIC_API_URL` on a fresh clone (document env in [../frontend/README.md](../frontend/README.md)).

**Regression**

- [ ] Phase 0 terminal tests where applicable.
- [ ] New automated tests for the phase’s critical path.

---

## Cadence

| When | Activity |
|------|----------|
| Each merge | Phase-specific checkpoint + Phase 0 regression |
| Weekly | Demo of current vertical slice; update MOCKUP_MAP statuses |
| Optional nightly | Full pipeline e2e after Phase 5 lands |

---

## See also

- [../frontend/README.md](../frontend/README.md) — frontend doc index
- [../frontend/ROUTES.md](../frontend/ROUTES.md) — proposed Next.js routes
- [../frontend/CONTRACTS.md](../frontend/CONTRACTS.md) — API JSON contracts
- [COMMAND_REFERENCE_NEXT_TASKS.md](COMMAND_REFERENCE_NEXT_TASKS.md) — CLI cross-reference
- [../reference/TESTING_SUITE.md](../reference/TESTING_SUITE.md) — existing test entry points
