# API contracts (FastAPI ↔ Next.js)

Source of truth in code: `middleware/api.py` (Pydantic models). OpenAPI UI: `GET http://localhost:8000/docs` when the server is running.

## Conventions

- **Content-Type:** `application/json` for `POST` bodies unless noted.
- **Errors:** FastAPI returns `{"detail": "..."}` for HTTP 4xx/5xx unless overridden.
- **CORS:** Development allows broad origins; production should restrict to the Next.js origin.

---

## `GET /status`

**Response** (`StatusResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `status` | string | e.g. `"healthy"` / `"error"` |
| `orchestrator_ready` | boolean | |
| `classical_model_loaded` | boolean | |
| `quantum_model_loaded` | boolean | Often `false` when orchestrator uses `use_quantum=False` |
| `entity_count` | integer | From embedder when available |
| `supported_relations` | string[] | Default includes `"CtD"` |

**Example**

```bash
curl -s http://localhost:8000/status | jq .
```

---

## `POST /predict-link` · `GET /predict-link`

**Request body** (`PredictionRequest`, POST)

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `drug` | string | yes | Name or ID (e.g. `DB00945`) |
| `disease` | string | yes | Name or ID (e.g. `DOID_9352`) |
| `method` | string | no | `"auto"` (default), `"classical"`, `"quantum"` |

**GET query params:** same names as fields.

**Response** (`PredictionResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `drug` | string | Echo |
| `disease` | string | Echo |
| `drug_id` | string | Resolved ID |
| `disease_id` | string | Resolved ID |
| `link_probability` | float | |
| `model_used` | string | |
| `status` | string | e.g. `"success"` / `"error"` |
| `error_message` | string \| null | |

---

## `POST /batch-predict`

**Request:** JSON array of `PredictionRequest` objects.

**Response:** JSON array of `PredictionResponse` (per item; failures may embed error in item).

---

## `POST /ranked-mechanisms`

**Request** (`RankedMechanismsRequest`)

| Field | Type | Required | Notes |
|-------|------|----------|--------|
| `hypothesis_id` | string | yes | e.g. `H-001`, `H-002`, `H-003` |
| `disease_id` | string | yes | Name or Hetionet ID |
| `top_k` | integer | no | Default `50` |

**Response** (`RankedMechanismsResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `ranked_candidates` | array | See `RankedCandidate` |
| `model_used` | string | |
| `hypothesis_id` | string | Echo |
| `status` | string | optional |
| `error_message` | string \| null | optional |

**RankedCandidate**

| Field | Type |
|-------|------|
| `compound_id` | string |
| `compound_name` | string |
| `score` | float |
| `mechanism_summary` | string |

---

## Planned (not in `api.py` yet)

| Endpoint | Purpose |
|----------|---------|
| `GET /runs/latest` | Latest pipeline summary from `results/` |
| `GET /runs/{id}` | Specific run metadata |
| `POST /jobs/pipeline` | Start `run_optimized_pipeline.py` as async job |
| `GET /jobs/{id}` | Job status / logs pointer |
| `GET /exports` or signed URLs | Safe download of `results/` artifacts |

When implemented, extend this file and [MOCKUP_MAP.md](MOCKUP_MAP.md).

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — stack and deployment
- [MOCKUP_MAP.md](MOCKUP_MAP.md) — which screen uses which endpoint
