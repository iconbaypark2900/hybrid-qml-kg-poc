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

## `GET /runs/latest`

**Response** (`LatestRunResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `status` | string | `"ok"` or `"no_results"` |
| `results_dir` | string | Resolved results directory path |
| `latest_csv` | object \| null | Parsed row from `latest_run.csv` |
| `latest_json` | object \| null | Parsed newest `optimized_results_*.json` (includes `ranking`, config, etc.) |
| `message` | string \| null | Human-readable context |

---

## `POST /jobs/pipeline`

**Request** (`JobCreateRequest`)

| Field | Type | Default | Notes |
|-------|------|---------|--------|
| `relation` | string | `"CtD"` | |
| `embedding_method` | string | `"ComplEx"` | |
| `embedding_dim` | integer | `64` | |
| `embedding_epochs` | integer | `100` | |
| `qml_dim` | integer | `12` | |
| `qml_feature_map` | string | `"ZZ"` | |
| `qml_feature_map_reps` | integer | `2` | |
| `fast_mode` | boolean | `true` | |
| `skip_quantum` | boolean | `false` | |
| `run_ensemble` | boolean | `false` | |
| `ensemble_method` | string | `"weighted_average"` | |
| `tune_classical` | boolean | `false` | |
| `results_dir` | string | `"results"` | |
| `quantum_config_path` | string | `"config/quantum_config.yaml"` | |

**Response** (`JobResponse`) — same schema as `GET /jobs/{id}`.

---

## `GET /jobs/{id}`

**Response** (`JobResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `id` | string | 12-char hex |
| `status` | string | `queued`, `running`, `success`, `failed` |
| `created_at` | float | Unix epoch |
| `started_at` | float \| null | |
| `finished_at` | float \| null | |
| `exit_code` | integer \| null | |
| `error` | string \| null | |
| `log_tail` | string[] \| null | Last ~200 lines of stdout |

---

## `GET /jobs`

**Response:** JSON array of `JobResponse` (most recent first).

---

## `GET /analysis/summary`

**Response** (`AnalysisSummaryResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `status` | string | `"ok"` or `"no_results"` |
| `best_model` | string \| null | |
| `best_pr_auc` | float \| null | |
| `model_count` | integer | |
| `classical_count` | integer | |
| `quantum_count` | integer | |
| `ensemble_count` | integer | |
| `ranking` | array \| null | Same structure as `LatestRunResponse.latest_json.ranking` |
| `relation` | string \| null | |
| `run_timestamp` | string \| null | |
| `message` | string \| null | |

---

## `GET /exports`

**Response** (`ExportListResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `status` | string | |
| `files` | array | `ExportFileInfo[]` |

**ExportFileInfo**

| Field | Type |
|-------|------|
| `name` | string |
| `size_bytes` | integer |
| `modified` | float |

---

## `GET /exports/{filename}`

Binary download (path traversal protected, allowlisted extensions: `.json`, `.csv`, `.txt`, `.log`).

---

## `GET /kg/stats`

**Response** (`KGStatsResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `status` | string | `"ok"` or `"unavailable"` |
| `entity_count` | integer | |
| `edge_count` | integer | |
| `relation_types` | string[] | |
| `embedding_dim` | integer \| null | |
| `qml_dim` | integer \| null | |
| `sample_entities` | string[] | First 20 |
| `sample_edges` | object[] | First 10, keys: `source`, `metaedge`, `target` |

---

## `GET /quantum/config`

**Response** (`QuantumConfigResponse`)

| Field | Type | Notes |
|-------|------|--------|
| `status` | string | |
| `execution_mode` | string \| null | `simulator`, `heron`, etc. |
| `backend` | string \| null | e.g. `ibm_torino` |
| `shots` | integer \| null | |
| `quantum_model_loaded` | boolean | |
| `config` | object \| null | Raw YAML as JSON |

---

## Planned (not in `api.py` yet)

| Endpoint | Purpose |
|----------|---------|
| `GET /runs/{id}` | Specific run metadata |

## See also

- [ARCHITECTURE.md](ARCHITECTURE.md) — stack and deployment
- [MOCKUP_MAP.md](MOCKUP_MAP.md) — which screen uses which endpoint
