# Deploying hybrid-qml-kg-poc on Fly.io

This guide covers hosting the **FastAPI** backend (`middleware/api.py`) and the **Next.js** app (`frontend/`) on [Fly.io](https://fly.io/). The Streamlit dashboard (`benchmarking/dashboard.py`) is optional and can run as a **second Fly app** if you need it in production.

## What you are deploying

| Component | Role | Default local dev |
|-----------|------|-------------------|
| FastAPI | REST API, orchestrator, predictions | `8780` (see `scripts/dev_stack.sh`) |
| Next.js | Web UI | `3780` |
| Streamlit | Legacy/benchmark UI | `8501` |

In production on Fly, you typically expose **one public HTTP port** (Fly sets `PORT`, often `8080`). The recommended pattern here matches the **root `Dockerfile`**: run **uvicorn** on `127.0.0.1:8000` and **Next.js** on `0.0.0.0:$PORT` so the browser talks only to Next; Next proxies API routes to FastAPI (see `frontend/next.config.mjs` rewrites).

---

## Prerequisites

1. [Fly.io account](https://fly.io/app/sign-up) and [flyctl](https://fly.io/docs/hands-on/install-flyctl/) installed.
2. Docker available locally (Fly builds use your `Dockerfile` unless you use a remote builder).
3. From this repository root, you can build the image you plan to ship (see **Data and models** below).

```bash
cd /path/to/hybrid-qml-kg-poc
fly version
```

---

## Critical: data, `models/`, and `results/`

The repo’s **`.dockerignore`** excludes `data/`, `models/`, and `results/` so local Docker builds stay small. The API and orchestrator **expect** those paths at runtime (embeddings, joblib models, Hetionet metadata, etc.).

Before deploying to Fly, choose **one** approach:

1. **Bake artifacts into the image (simplest for a demo)**  
   - Temporarily comment out or narrow the `data/`, `models/`, `results/` lines in `.dockerignore` for the production build only, **or** use a dedicated `Dockerfile` copy step that adds only the files you need.  
   - Rebuild and deploy. Keep image size and secrets in mind (do not bake private tokens).

2. **Fly Volumes (durable disk)**  
   - Create a [Fly volume](https://fly.io/docs/reference/volumes/) mounted at e.g. `/app/data`, `/app/models`, `/app/results`.  
   - SSH into the machine once and copy or download the required files into the volume, **or** use a release task / one-off machine to seed data.

3. **Object storage**  
   - Store large artifacts in S3/R2/etc. and download at boot (add a small init script). Not implemented in-repo; document your bucket and IAM separately.

If you deploy without fixing this, the container may start but `/status` or predictions can fail until paths exist.

---

## Option A — Single Fly app (Next.js + FastAPI, recommended)

The repository root **`Dockerfile`** already builds the Next.js app and runs both processes. Fly must bind the **public** process to **`0.0.0.0` and `$PORT`**.

### 1. Adjust the container start command for Fly

The stock `CMD` in the root `Dockerfile` uses a fixed port (`7860`) for Next.js. On Fly, use the **`PORT`** environment variable (injected by the platform).

Replace the final `CMD` with something equivalent to:

```dockerfile
ENV PORT=8080
CMD ["sh", "-c", "uvicorn middleware.api:app --host 127.0.0.1 --port 8000 & cd frontend && ./node_modules/.bin/next start -H 0.0.0.0 -p ${PORT}"]
```

Notes:

- **FastAPI** stays on **127.0.0.1:8000** (not publicly exposed); only Next listens on `$PORT`.
- Build the frontend with **same-origin API** so the browser does not hard-code a dev URL, e.g. build with `NEXT_PUBLIC_API_URL=` (empty) as in the root `Dockerfile`, so client calls use relative URLs and the Next rewrites/proxy can reach the API.

### 2. Create the Fly app

From the repo root:

```bash
fly launch --no-deploy
```

- Choose a **region** close to users.
- When prompted for a Dockerfile, select the **root** `Dockerfile` (or set `dockerfile` in `fly.toml` as below).
- Do **not** deploy until `fly.toml` matches your **internal port** (see next section).

### 3. Example `fly.toml` (HTTP service)

Adapt names and sizing to your org. This assumes the **container listens on `PORT`** (e.g. `8080`).

```toml
app = "your-app-name"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0
  processes = ["app"]

[[http_service.checks]]
  grace_period = "30s"
  interval = "15s"
  method = "GET"
  path = "/status"
  timeout = "10s"

[vm]
  cpu_kind = "shared"
  cpus = 1
  memory = "2048mb"
```

- **`internal_port`** must match whatever port **Next.js** uses (`${PORT}` in `CMD`).
- **`/status`** is served by FastAPI; with Next rewrites/fallback, it is often reachable at the same host (verify after deploy). If your health check fails, point `path` to a route you know returns `200` from Next, or split apps (Option B).

### 4. Secrets

Set tokens at runtime (not in the image):

```bash
fly secrets set IBM_QUANTUM_TOKEN="your-token"   # only if you use IBM Quantum features
```

Match any variable names your `config/*.yaml` and `middleware/api.py` expect.

### 5. Deploy

```bash
fly deploy
fly status
fly logs
```

Open `https://your-app-name.fly.dev` (or your custom domain).

### 6. Resource hints

- **CPU/RAM**: Quantum simulation and embedding work can be heavy; start at **2 GB** RAM and increase if the process OOMs.
- **Scale**: `fly scale count` / `fly scale vm` per [Fly scaling docs](https://fly.io/docs/launch/scale-count/).

---

## Option B — Two Fly apps (API + Next)

Use this if you want **separate scaling**, caching, or teams for API vs frontend.

1. **API app**  
   - Build from `deployment/Dockerfile.api`.  
   - Expose **8000** (or set `CMD` to listen on `$PORT` and set `internal_port` accordingly).  
   - Attach volumes if needed for `data/`, `models/`, `results/`.

2. **Next app**  
   - Build the `frontend` with Docker (multi-stage Node image) or deploy a static export if you later add `output: "export"` (not configured in-repo today).  
   - Set **`NEXT_PUBLIC_API_URL`** to the **public HTTPS URL** of the API app (e.g. `https://your-api.fly.dev`) at **build time** for client-side fetches, or use Next rewrites to proxy to that origin server-side.

This mirrors local `dev_stack` (two processes) but across two Fly apps.

---

## Option C — Streamlit dashboard only

To run the **Streamlit** benchmarking UI (`deployment/Dockerfile.dashboard`):

```bash
# In a new directory or separate fly.toml
fly launch --dockerfile deployment/Dockerfile.dashboard
```

- Expose **8501** (`internal_port = 8501`) or change Streamlit to `$PORT` the same way as Next.  
- Point the dashboard at API URLs or mounted `results/` as appropriate.  
- Compose in development uses **depends_on** the API; on Fly, use the API’s public URL or private networking ([Fly private networking](https://fly.io/docs/reference/private-networking/)) if both apps are on Fly.

---

## Troubleshooting

| Symptom | Things to check |
|--------|-------------------|
| `502` / connection refused | Next not listening on `0.0.0.0:$PORT`; `internal_port` in `fly.toml` mismatch. |
| `Failed to fetch` / wrong API host | `NEXT_PUBLIC_API_URL` at build time vs actual API URL; CORS on FastAPI if browser calls API directly. |
| Orchestrator / prediction errors | Missing `data/`, `models/`, or `results/` in the image or volume. |
| Health check fails | Path wrong; temporarily use `/` or disable checks, then fix. |
| Slow cold starts | `min_machines_running = 1` or accept cold start with `auto_stop_machines`. |

Useful commands:

```bash
fly logs -a your-app-name
fly ssh console -a your-app-name
```

---

## Checklist before production

- [ ] Data and model files present in image or on a Fly volume.  
- [ ] No secrets in the image; use `fly secrets`.  
- [ ] HTTPS-only traffic (`force_https = true`).  
- [ ] Health checks aligned with a real `200` route.  
- [ ] IBM Quantum or other cloud tokens rotated and scoped.  
- [ ] Review FastAPI CORS settings in `middleware/api.py` if the browser calls the API on a **different origin** than the Next app.

---

## References

- Fly.io: [Launch a new app](https://fly.io/docs/launch/), [Volumes](https://fly.io/docs/reference/volumes/), [Secrets](https://fly.io/docs/reference/secrets/).  
- Repo: root `Dockerfile`, `deployment/Dockerfile.api`, `deployment/Dockerfile.dashboard`, `deployment/docker-compose.yml`.  
- Local dev: `README.md`, `scripts/dev_stack.sh`, `AGENTS.md`.
