# Proposed Next.js route tree

Maps **mockup folders** to **URL paths** for the App Router (`frontend/app/`). Adjust names when scaffolding; keep **one** sidebar model (see [MOCKUP_MAP.md](MOCKUP_MAP.md)).

## Route table

| Path | Mockup folder(s) | Notes |
|------|------------------|--------|
| `/` | `experiment_overview` | Redirect to `/experiments` or render overview at root—pick one convention |
| `/experiments` | `experiment_overview` | Latest run / metrics |
| `/system` | `system_status_details` | `GET /status` |
| `/knowledge-graph` | `knowledge_graph_exploration` | Planned API |
| `/quantum` | `knowledge_quantum_logic` | Config + results |
| `/simulation` | `simulation_control_panel` | Job control |
| `/simulation/parameters` | `simulation_parameters` | Pipeline args form |
| `/molecular-design` | `molecular_design` | Predictions |
| `/hypotheses/new` | `new_hypothesis_entry`, `add_new_hypothesis_form` | Same flow; two mockups = one route + variants |
| `/analysis/drug-delivery` | `analysis_of_drug_delivery` | |
| `/analysis/next-steps` | `analysis_next_steps` | |
| `/export` | `experiment_data_export_options` | |

## Suggested `app/` layout

```text
frontend/app/
  layout.tsx              # Quantum Slate shell, fonts, nav
  page.tsx                # or redirect
  experiments/page.tsx
  system/page.tsx
  knowledge-graph/page.tsx
  quantum/page.tsx
  simulation/page.tsx
  simulation/parameters/page.tsx
  molecular-design/page.tsx
  hypotheses/new/page.tsx
  analysis/drug-delivery/page.tsx
  analysis/next-steps/page.tsx
  export/page.tsx
```

Shared UI lives under `frontend/components/` (not listed here).

## See also

- [MOCKUP_MAP.md](MOCKUP_MAP.md) — API and data sources per screen
- [PIPELINE_UI_FLOW.md](PIPELINE_UI_FLOW.md) — user journeys
