# Mockup → route → API → data

Each row links the **Stitch static HTML** (design reference) to a **proposed Next.js route**, the **FastAPI** (or pipeline) **data source**, and implementation status.

**Mockup root:** `stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/`

| Screen (folder) | Mockup file | Proposed route | API / backend | Data / pipeline source | Status |
|-----------------|-------------|----------------|---------------|------------------------|--------|
| `experiment_overview` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/experiment_overview/code.html) | `/experiments` | `GET /runs/latest` | `results/optimized_results_*.json`, `results/latest_run.csv`, `results/experiment_history.csv` | **Wired** — API + Next.js page |
| `system_status_details` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/system_status_details/code.html) | `/system` | `GET /status` | Orchestrator, embedder entity count | **Wired** — API + Next.js page |
| `knowledge_graph_exploration` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/knowledge_graph_exploration/code.html) | `/knowledge-graph` | `GET /kg/stats` | `kg_layer`, trained embeddings, Hetionet-derived structures | **Wired** — API + Next.js page |
| `knowledge_quantum_logic` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/knowledge_quantum_logic/code.html) | `/quantum` | `GET /quantum/config` | `config/quantum_config*.yaml`, quantum run outputs in `results/` | **Wired** — API + Next.js page |
| `simulation_control_panel` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/simulation_control_panel/code.html) | `/simulation` | `POST /jobs/pipeline`, `GET /jobs/{id}`, `GET /jobs` | `scripts/run_optimized_pipeline.py` (subprocess) | **Wired** — job API + Next.js page |
| `simulation_parameters` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/simulation_parameters/code.html) | `/simulation/parameters` | Same job API + config schema | CLI flags mirrored in API body | **Wired** — Next.js form + job create |
| `molecular_design` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/molecular_design/code.html) | `/molecular-design` | `POST /predict-link`, `POST /batch-predict` | Orchestrator predictions | **Wired** — API + Next.js page |
| `new_hypothesis_entry` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/new_hypothesis_entry/code.html) | `/hypotheses/new` | `POST /ranked-mechanisms`; persistence **planned** | Mechanism hypotheses H-001–H-003, disease scope | **Wired** — ranking API + Next.js page |
| `add_new_hypothesis_form` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/add_new_hypothesis_form/code.html) | `/hypotheses/new` (same flow, alternate layout) | Same as above | Same | **Planned** |
| `analysis_of_drug_delivery` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/analysis_of_drug_delivery/code.html) | `/analysis/drug-delivery` | `GET /analysis/summary` | Results JSON + report-style fields | **Wired** — API + Next.js page |
| `analysis_next_steps` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/analysis_next_steps/code.html) | `/analysis/next-steps` | `GET /analysis/summary` | Parsed `results/*.json`, client-side recommendations | **Wired** — API + Next.js page |
| `experiment_data_export_options` | [code.html](../../stitch_knowledge_quantum_logic_biomedical/stitch_knowledge_quantum_logic_biomedical/experiment_data_export_options/code.html) | `/export` | `GET /exports`, `GET /exports/{filename}` | `results/` files, CSV/JSON | **Wired** — API + Next.js page |

## Legend

- **Wired:** API endpoint exists and Next.js page is connected to it.
- **Partial:** at least one real endpoint or data path exists; UI still to be built in Next.js.
- **Planned:** design reference exists; API contract and implementation still required.

## Implementation notes

1. **Single navigation model:** Sidebar labels in the mockups should match **one** route tree (avoid duplicating “Dashboard” across unrelated URLs without a shared `layout.tsx`).
2. **Token parity:** When building components, copy Tailwind `theme.extend` colors from any mockup `code.html` into the Next `tailwind.config` (see [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md)).
3. **Orchestrator:** `use_quantum=False` is the default in code today; the quantum mockup screens should read actual flags from `GET /status` and config once quantum paths are enabled.

## Related code

- API routes: `middleware/api.py`
- Orchestrator: `middleware/orchestrator.py`
- Pipeline entry: `scripts/run_optimized_pipeline.py`
