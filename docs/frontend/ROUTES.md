# Next.js route tree

Maps **mockup folders** to **URL paths** for the App Router (`frontend/app/`). Keep **one** sidebar model (see [MOCKUP_MAP.md](MOCKUP_MAP.md)).

## Route table

| Path | Nav label | Mockup folder(s) | Notes |
|------|-----------|------------------|-------|
| `/` | Home | `experiment_overview` | Stateful: branches hero/pills on `GET /runs/latest`. |
| `/predict` | Predict treatment | `molecular_design` | Canonical pairwise prediction. |
| `/molecular-design` | — | `molecular_design` | Redirects to `/predict` (bookmark preservation). |
| `/experiments` | Latest run & models | `experiment_overview` | Latest run summary + sortable model leaderboard. |
| `/analysis/drug-delivery` | Drug delivery analysis | `analysis_of_drug_delivery` | Aggregated perf from `GET /analysis/summary`. |
| `/analysis/next-steps` | Recommendations | `analysis_next_steps` | Action cards deep-link into `New run` presets. |
| `/hypotheses/new` | Ranked candidates | `new_hypothesis_entry`, `add_new_hypothesis_form` | Persisted hypothesis library + editor + ranking + timeline. |
| `/visualization` | Charts & exploration | — | Multi-tab: predictions, 3D, KG graph, embeddings, circuit, model comparison. Accepts `?tab=kggraph|circuit|…` deep-link param. |
| `/simulation/parameters` | New run | `simulation_parameters` | Preset-driven pipeline form with hypothesis linkage and experiment metadata. |
| `/simulation` | Pipeline jobs | `simulation_control_panel` | Job list with 5 s polling + hypothesis linkage. |
| `/system` | System status | `system_status_details` | `GET /status` readiness + entity count. |
| `/knowledge-graph` | Knowledge graph | `knowledge_graph_exploration` | Entity search + subgraph explorer. Deep-links to `/visualization?tab=kggraph`. |
| `/quantum` | Quantum config | `knowledge_quantum_logic` | Config + IBM runtime verify. Deep-links to `/visualization?tab=circuit`. |
| `/settings` | Settings | — | Tenant-scoped IBM Quantum credential storage + verification. |
| `/export` | Export | `experiment_data_export_options` | `GET /exports` file list + download. |

## Actual `app/` layout

```text
frontend/app/
  layout.tsx                          # Quantum Slate shell, fonts, nav
  page.tsx                            # Stateful Home (fetchLatestRun)
  predict/page.tsx                    # Canonical prediction (PredictForm)
  molecular-design/page.tsx           # redirect("/predict")
  experiments/page.tsx                # Latest run & model leaderboard
  analysis/drug-delivery/page.tsx     # Drug delivery analysis
  analysis/next-steps/page.tsx        # Recommendations
  hypotheses/new/page.tsx             # Ranked candidates (RankedForm)
  visualization/page.tsx              # Charts & exploration (?tab= deep-links)
  simulation/parameters/page.tsx      # Preset + hypothesis-linked run form
  simulation/page.tsx                 # Pipeline job list + hypothesis column
  system/page.tsx                     # System status
  knowledge-graph/page.tsx            # KG stats + subgraph explorer
  quantum/page.tsx                    # Quantum config + IBM verify
  settings/page.tsx                   # Tenant-scoped IBM Quantum settings
  export/page.tsx                     # Export file list
```

Shared UI lives under `frontend/components/` (not listed here).

## Shared continuity primitives

- `ResearchSessionStrip` is rendered globally in `AppShell` and shows latest run/job context.
- `ResearchNextActions` appears on key pages to keep the researcher loop explicit.
- `ApiRecoveryCard` and `NoPipelineResultsCta` standardize recovery actions (`System`, `New run`, `Pipeline jobs`).

## See also

- [MOCKUP_MAP.md](MOCKUP_MAP.md) — API and data sources per screen
- [PIPELINE_UI_FLOW.md](PIPELINE_UI_FLOW.md) — user journeys
