# Upcoming execution documentation

This folder is the hub for **executing** the next phase of the hybrid QML-KG project: MultiModelFusion pipeline integration, extended Optuna search, full-scale data runs, and quick experiments (higher PCA, graph features in QML). Do not duplicate task lists here—use the canonical sources linked below.

## Document map

| Type | Path | Purpose |
|------|------|---------|
| **Canonical task list** | [planning/UPCOMING_TASKS.md](../planning/UPCOMING_TASKS.md) | What to run; do not copy—link |
| **Progress / run log** | [planning/UPCOMING_TASKS_PROGRESS.md](../planning/UPCOMING_TASKS_PROGRESS.md) | Completed experiments, results |
| **CLI reference** | [planning/COMMAND_REFERENCE_NEXT_TASKS.md](../planning/COMMAND_REFERENCE_NEXT_TASKS.md) | Commands and flags |
| **Broader ideas** | [planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md](../planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md) | Experiment ideas beyond this phase |
| **Completed phase** | [planning/NEXT_TASKS_IMPLEMENTATION/README.md](../planning/NEXT_TASKS_IMPLEMENTATION/README.md) | Prior NEXT_TASKS implementation hub |

**Optional files in this folder** (add when needed):

| File | When to add |
|------|-------------|
| `MULTI_MODEL_FUSION_INTEGRATION.md` | Design notes and checklist for wiring MultiModelFusion into `run_optimized_pipeline.py` |
| `EXPERIMENT_RUNBOOK.md` | Phase-specific command deltas, machine notes, output paths |
| `RESULTS_LOG.md` | Append-only run table (alternative to extending UPCOMING_TASKS_PROGRESS) |

---

**Recommendation:** Start with this README as the single entry point. Add the optional files above when you begin integration work or need a dedicated run log.
