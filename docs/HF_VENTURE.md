# Hugging Face lite — venture overview

This document frames **why** we are building a Hugging Face Space alongside the main **Next.js + FastAPI** product, and **what success looks like**. Implementation detail lives in [`deployment/HF_LITE.md`](deployment/HF_LITE.md).

---

## Positioning

| Surface | Role |
|---------|------|
| **Main product** | Full UI (`frontend/`), complete API (`middleware/api.py`), pipeline jobs, exports, production deploy (e.g. Fly per [`deployment/FLY_IO.md`](deployment/FLY_IO.md)). |
| **HF lite** | A **narrow, researcher-facing demo** on [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces): cheap to host, easy to fork, same **scientific story** (KG → embeddings → classical/quantum link prediction, IBM Quantum connectivity) without shipping the whole product. |

The lite version is **not** feature parity. It is **credible science + clear boundaries** so researchers can explore results and quantum/runtime flows without a full local or cloud production setup.

---

## Audience

- Researchers and reviewers who want a **quick, read-only** look at outputs and quantum configuration.
- Contributors evaluating **IBM Quantum / Runtime** integration (BYOK-style checks) in a constrained environment.
- Educators demonstrating **hybrid QML on a knowledge graph** without installing the full pipeline stack.

---

## What will be built (summary)

1. **Gradio** (or equivalent lightweight UI) as the Space front end—not the full Next.js app. **Scaffold:** [`hf_space/app.py`](../hf_space/app.py) + [`scripts/run_hf_lite.sh`](../scripts/run_hf_lite.sh).
2. A **minimal Python path**: the lite UI uses **the same** `middleware.api:app` in-process (`TestClient`), not a forked API—plus **curated** `data/`, `models/`, and `results/` on the Hub for reliable cold start.
3. **Tiered scope**: **Tier A + B (JSON)** are implemented in the scaffold (status, latest run, analysis, quantum config/verify, three `/viz/*` reads). Optional **single prediction** (`/predict-link`) can be added as Tier C.
4. **Explicit non-goals** for v1: long-running **pipeline jobs**, full **Optuna** / training loops, full **export** parity—those remain on the main product or **Hugging Face Jobs** if needed later.

---

## Success criteria

- Space **starts reliably** on free-tier constraints (CPU, image size, timeouts).
- **No secrets in the repo**; IBM tokens via Space **Secrets** or ephemeral user input where appropriate.
- README on the Hub explains **differences vs full app** and links back to this repository.
- Optional: one **frozen** experiment is visible end-to-end (latest run + at least one viz tab).

---

## Related documents

| Doc | Purpose |
|-----|---------|
| [`deployment/HF_LITE.md`](deployment/HF_LITE.md) | Technical blueprint: APIs, tiers, phases, diagrams. |
| [`deployment/FLY_IO.md`](deployment/FLY_IO.md) | Full-stack production (Next + FastAPI); data/model bundling patterns apply by analogy. |
| [`../AGENTS.md`](../AGENTS.md) | Local dev defaults (`dev_stack.sh`, ports). |

---

## Venture parking lot

- **HF Jobs** for “run pipeline” or heavy training off the interactive Space.
- Public **dataset or artifact** on the Hub to download at startup if the Docker image must stay small.
- Second Space variant: **Gradio-only quantum verify** (smallest possible image) vs **viz + results** (richer).
