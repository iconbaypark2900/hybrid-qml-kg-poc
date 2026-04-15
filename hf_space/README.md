---
title: Hybrid QML-KG Lite
emoji: 🔬
colorFrom: blue
colorTo: teal
sdk: gradio
sdk_version: 5.44.0
app_file: hf_space/app.py
pinned: false
license: mit
---

# Hybrid QML-KG — Hugging Face lite

Gradio UI that calls the same **FastAPI** application as the main product (`middleware/api.py`) via an in-process `TestClient` — no duplicate business logic.

## What’s included

- **Tier A:** `/status`, `/runs/latest`, `/analysis/summary`, `/quantum/config`, `/quantum/runtime/verify`
- **Tier B (JSON):** `/viz/run-predictions`, `/viz/model-metrics`, `/viz/circuit-params`

## Deploy this Space

1. Create a **Gradio** Space on Hugging Face.
2. Push **this entire repository** (or mirror `hf_space/` + parent packages — the app `chdir`s to repo root and imports `middleware.api`).
3. Point the Space at **`hf_space/app.py`** (this README’s YAML uses `app_file: hf_space/app.py`). If your Space repo only contains the `hf_space/` folder at root, copy `app.py` to the repo root and set `app_file: app.py`.
4. **Dependencies:** Hugging Face installs **`requirements.txt` from the repository root** by default. The root file in this monorepo is very large. For the Space, either:
   - maintain a **branch** where root `requirements.txt` is replaced by the contents of `hf_space/requirements.txt`, or  
   - use the **Docker** SDK and `pip install -r hf_space/requirements.txt` in the Dockerfile.
5. Optional **Secrets:** `IBM_Q_TOKEN`, `IBM_QUANTUM_INSTANCE` for IBM Runtime verify without pasting tokens in the UI.

## Local run

From the **repository root**:

```bash
./scripts/run_hf_lite.sh
```

Or:

```bash
export PYTHONPATH=.
python hf_space/app.py
```

## Not included (vs full app)

- Next.js UI (`frontend/`)
- Pipeline job APIs as a product surface (no long runs on free tier by default)

See [`docs/deployment/HF_LITE.md`](../docs/deployment/HF_LITE.md) and [`docs/HF_VENTURE.md`](../docs/HF_VENTURE.md).
