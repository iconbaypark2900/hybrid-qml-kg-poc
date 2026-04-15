# Documentation index

Markdown for this project lives under `docs/` (except the root `README.md`). Use this page as a map.

## Overview

| Path | Purpose |
|------|---------|
| [overview/PROJECT_EXPLANATION.md](overview/PROJECT_EXPLANATION.md) | High-level project narrative |
| [overview/IMPLEMENTATION_RECAP.md](overview/IMPLEMENTATION_RECAP.md) | Pipeline flags, GPU/hardware, recent changes |
| [TECHNICAL_PAPER.md](TECHNICAL_PAPER.md) | Methods, experiments, discussion |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |

## Frontend & UI upgrade (Next.js)

| Path | Purpose |
|------|---------|
| [frontend/README.md](frontend/README.md) | Index: routes, API contracts, mockup map, architecture, design system, pipeline flows |
| [frontend/ROUTES.md](frontend/ROUTES.md) | Proposed Next.js App Router paths vs mockups |
| [frontend/CONTRACTS.md](frontend/CONTRACTS.md) | FastAPI JSON request/response shapes for the UI |

## Planning & tasks

| Path | Purpose |
|------|---------|
| [planning/NEXT_TASKS.md](planning/NEXT_TASKS.md) | Task list |
| [planning/UPCOMING_TASKS.md](planning/UPCOMING_TASKS.md) | Follow-on tasks |
| [planning/UPCOMING_TASKS_PROGRESS.md](planning/UPCOMING_TASKS_PROGRESS.md) | Progress on upcoming tasks |
| [planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md](planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md) | Experiment log and commands |
| [planning/COMMAND_REFERENCE_NEXT_TASKS.md](planning/COMMAND_REFERENCE_NEXT_TASKS.md) | CLI for next-task experiments |
| [planning/IMPLEMENTATION_STATUS_REPORT.md](planning/IMPLEMENTATION_STATUS_REPORT.md) | Status vs. `NEXT_TASKS` |
| [planning/NEXT_TASKS_IMPLEMENTATION/README.md](planning/NEXT_TASKS_IMPLEMENTATION/README.md) | Implementation hub |
| [planning/FRONTEND_ROLLOUT_PLAN.md](planning/FRONTEND_ROLLOUT_PLAN.md) | Next.js UI rollout: phases, checkpoints, tests |
| [upcoming-execution/README.md](upcoming-execution/README.md) | Upcoming execution hub (fusion, Optuna, full-scale) |

## Reference (commands & testing)

| Path | Purpose |
|------|---------|
| [reference/COMMAND_REFERENCE.md](reference/COMMAND_REFERENCE.md) | Command reference |
| [reference/QUICK_START_COMMANDS.md](reference/QUICK_START_COMMANDS.md) | Quick-start commands |
| [reference/TEST_COMMANDS.md](reference/TEST_COMMANDS.md) | Test-related commands |
| [reference/TESTING_SUITE.md](reference/TESTING_SUITE.md) | Test suite overview |
| [reference/USAGE_EXAMPLES.md](reference/USAGE_EXAMPLES.md) | Usage examples |
| [reference/EXPECTED_OUTPUTS.md](reference/EXPECTED_OUTPUTS.md) | Expected pipeline outputs |
| [reference/FEATURE_DIAGNOSTICS.md](reference/FEATURE_DIAGNOSTICS.md) | Feature diagnostics |

## Guides

| Path | Purpose |
|------|---------|
| [guides/COMMANDS.md](guides/COMMANDS.md) | Command cookbook |
| [guides/DIRECTORY_GUIDE.md](guides/DIRECTORY_GUIDE.md) | Repository map |
| [guides/env_setup.md](guides/env_setup.md) | Environment setup |
| [guides/cli_container.md](guides/cli_container.md) | CLI / container notes |
| [guides/FULL_GRAPH_EMBEDDINGS_GUIDE.md](guides/FULL_GRAPH_EMBEDDINGS_GUIDE.md) | Full-graph embeddings |
| [guides/QSVC_ONLY_GUIDE.md](guides/QSVC_ONLY_GUIDE.md) | QSVC-focused runs |

## Core technical docs (`docs/` root)

Optimization, quantum behavior, evaluation, dashboard:

- [OPTIMIZATION_QUICKSTART.md](OPTIMIZATION_QUICKSTART.md), [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)
- [WHY_QUANTUM_UNDERPERFORMS.md](WHY_QUANTUM_UNDERPERFORMS.md), [WHEN_TO_USE_QUANTUM.md](WHEN_TO_USE_QUANTUM.md)
- [LEAKAGE_PREVENTION_GUIDE.md](LEAKAGE_PREVENTION_GUIDE.md), [CV_EVALUATION_GUIDE.md](CV_EVALUATION_GUIDE.md)
- [TUTORIAL.md](TUTORIAL.md), [THEORY.md](THEORY.md)
- [DASHBOARD_PRESETS_CHEATSHEET.md](DASHBOARD_PRESETS_CHEATSHEET.md), [DASHBOARD_PRESENTATION_AND_GLOSSARY.md](DASHBOARD_PRESENTATION_AND_GLOSSARY.md)

## Reports & analyses (`docs/reports/`)

Experiment writeups, results summaries, and historical notes: see [reports/](reports/).

## Improvements logs (`docs/improvements/`)

Iteration summaries: [improvements/](improvements/).

## Deployment

| Path | Purpose |
|------|---------|
| [HF_VENTURE.md](HF_VENTURE.md) | Hugging Face **lite** initiative: audience, positioning vs full app, success criteria |
| [deployment/HF_LITE.md](deployment/HF_LITE.md) | Hugging Face **lite** technical blueprint: tiers, APIs, phases, architecture diagrams |
| [../hf_space/README.md](../hf_space/README.md) | HF Space README template + deploy notes (`app_file`, slim `requirements.txt`) |
| [deployment/DEPLOY_HUGGINGFACE.md](deployment/DEPLOY_HUGGINGFACE.md) | Hugging Face Spaces (Streamlit dashboard push workflow) |
| [deployment/FLY_IO.md](deployment/FLY_IO.md) | Fly.io: Next.js + FastAPI production deploy |
| [deployment/DOCKER_INSTALL.md](deployment/DOCKER_INSTALL.md) | Docker install notes |
| [deployment/README_FEATUREMAP_TESTING.md](deployment/README_FEATUREMAP_TESTING.md) | Feature-map Docker testing index |
| [deployment/FEATUREMAP_TESTING_GUIDE.md](deployment/FEATUREMAP_TESTING_GUIDE.md), [deployment/FEATUREMAP_DOCKERFILE_ANALYSIS.md](deployment/FEATUREMAP_DOCKERFILE_ANALYSIS.md), … | Feature-map build & test notes |

The **`deployment/`** directory at the **repo root** holds Dockerfiles, `docker-compose.yml`, and image-specific requirements. Prose for deployment lives under **`docs/deployment/`** (above).

## Layer-specific docs

| Path | Purpose |
|------|---------|
| [kg_layer/KG_EMBEDDING_DESCRIPTION.md](kg_layer/KG_EMBEDDING_DESCRIPTION.md) | KG embedding notes |
| [quantum/quantum_embedding_feature_maps.md](quantum/quantum_embedding_feature_maps.md) | Quantum feature maps |
| [quantum/quantum_classifiers_qsvc_vqc.md](quantum/quantum_classifiers_qsvc_vqc.md) | QSVC / VQC classifiers |

## Code layout (outside `docs/`)

| Location | Purpose |
|----------|---------|
| [../tests/](../tests/) | Quantum improvements test modules (`test_quantum_improvements_*.py`) |
| [../run_tests.py](../run_tests.py) | Entry point: `python run_tests.py --mode terminal\|dashboard\|both` |
| [../scripts/implementations/](../scripts/implementations/) | Standalone experiment drivers (`implement_*.py`) |
| [../scripts/shell/](../scripts/shell/) | Bash helpers (`run_quantum_fixed.sh`, phased improvement scripts, …) |
| [../scripts/demos/](../scripts/demos/) | Small demos for the test suite |

## Other

- [experiments/README.md](../experiments/README.md) — experiment scripts
