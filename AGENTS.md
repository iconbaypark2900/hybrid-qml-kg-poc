## Learned User Preferences

- When implementing an attached Cursor plan, do not modify the plan file; use existing todos and mark them in progress instead of recreating them.
- Prefer merging or consolidating work from extra git worktrees into the primary local clone and continuing on the main working tree.
- Do not publish `.cursor/` to Git remotes; if it was ever committed, stop tracking it with `git rm -r --cached .cursor`—`.gitignore` alone does not untrack files already in the index.

## Learned Workspace Facts

- The day-to-day UI is still Streamlit (`benchmarking/dashboard.py` / `scripts/launch_dashboard.sh`); the Next.js app under `frontend/` is the in-progress migration aligned with Stitch mockups and `docs/frontend/` until parity is reached.
- For API and Next.js work, the preferred slice order is: read-only latest run from `results/` (e.g. `GET /runs/latest`) plus `/experiments` in Next before pipeline job APIs and simulation polling.
- The Next.js app resolves the backend via `NEXT_PUBLIC_API_URL` when the dev server starts; that URL must match the FastAPI host and port. This repo’s default dev ports are **8780** (API) and **3780** (Next) so they do not collide with other projects on 8000/3000. Restart Next after editing `frontend/.env.local`.
- Running `./scripts/dev_stack.sh` from the repo root aligns FastAPI and Next with the same API base URL; a common failure mode is the browser calling one port while uvicorn listens on another, which shows up as `Failed to fetch`.
- **Fly.io combined image** (root `Dockerfile`): Next.js listens on **0.0.0.0:$PORT** (default **8080**, aligned with `fly.toml` `internal_port`); **FastAPI** stays on **127.0.0.1:8000** and is reached via Next rewrites—not the same as local dev’s **8780/3780** split. That image installs Python deps from **`requirements-huggingface.txt`**; keep **Qiskit 1.x** there—**Qiskit 2** removes `Sampler` from `qiskit.primitives` and breaks imports in `quantum_layer/qml_model.py`.
- A **Gradio “HF lite”** shell (`hf_space/app.py`) reuses the same FastAPI app as the main stack; run it locally with `./scripts/run_hf_lite.sh` (default **7860**) when you want a lightweight researcher-facing UI instead of Next.js or Streamlit.
- **Full-graph embedding on CUDA / DGX:** from the repo root run `./scripts/run_full_embedding_dgx.sh` (wraps `run_optimized_pipeline.py` with RotatE full-graph settings and `config/quantum_config_dgx.yaml`; logs under `results/`). Install CUDA-enabled PyTorch so PyKEEN uses the GPU. Details: `docs/deployment/DGX_SPARK.md`.
