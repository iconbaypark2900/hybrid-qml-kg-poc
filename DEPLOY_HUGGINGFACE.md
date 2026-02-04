# Deploy the dashboard to Hugging Face Spaces

Follow these steps to run the **Hybrid QML-KG** Streamlit dashboard on [Hugging Face Spaces](https://huggingface.co/spaces).

## 1. Create a Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces).
2. Click **Create new Space**.
3. Fill in:
   - **Space name**: e.g. `hybrid-qml-kg-dashboard`
   - **License**: your choice (e.g. MIT).
   - **SDK**: choose **Streamlit**.
   - **Space hardware**: Free (CPU) is enough for the dashboard.
4. Click **Create Space**.

## 2. Push this repo to the Space

You can either connect GitHub or push via Git.

### Option A: Connect GitHub (recommended)

1. In your new Space, open **Settings** → **Repository**.
2. Under **Repository**, use **Clone this Space** and push from your machine:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   cd YOUR_SPACE_NAME
   git remote add origin-main https://github.com/YOUR_ORG/hybrid-qml-kg-poc.git
   git pull origin-main main --allow-unrelated-histories
   git push origin main
   ```
   Or: copy the contents of this repo into the cloned Space folder, then:
   ```bash
   git add .
   git commit -m "Add Hybrid QML-KG dashboard"
   git push origin main
   ```

### Option B: Push from this repo

From the **hybrid-qml-kg-poc** directory:

```bash
# Add HF as a remote (replace USERNAME and SPACE_NAME)
git remote add hf https://huggingface.co/spaces/USERNAME/SPACE_NAME

# Push (HF Spaces use 'main' by default)
git push hf main
```

If your default branch is `master`, use `git push hf master:main` or switch to `main` first. If your branch has another name (e.g. `roc/dashboard-polish`), push it as `main` on the Space: `git push hf roc/dashboard-polish:main`.

**Authentication:** The first push will prompt for your Hugging Face token. Either:

- **Login once** (run in your terminal; the CLI is `hf`, not `huggingface-cli`):
  ```bash
  pip install -U huggingface_hub
  hf auth login
  ```
  If `hf` isn’t on your PATH, use `.venv/bin/hf auth login`. Then run `git push hf …` as above.

- **Or push once with the token in the URL** (don’t commit the token): create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (Write access), then:
  ```bash
  git push https://YOUR_USERNAME:YOUR_TOKEN@huggingface.co/spaces/USERNAME/SPACE_NAME YOUR_BRANCH:main
  ```

## 3. (Optional) Use slim requirements for faster build

The root `requirements.txt` includes full dev dependencies (Jupyter, PyTorch, etc.) and can make the Space build slow or run out of memory.

For a **lighter, faster build**, use the dashboard-only dependencies:

- In the Space repo, **replace** the contents of `requirements.txt` with the contents of **`requirements-huggingface.txt`**,  
  **or**
- Rename `requirements-huggingface.txt` to `requirements.txt` before pushing.

Then commit and push so the Space rebuilds with the slim set.

## 4. App entry point

The Space is configured via the YAML block at the top of **README.md**:

- **`app_file: benchmarking/dashboard.py`** — Hugging Face runs:
  `streamlit run benchmarking/dashboard.py`

No need to add an `app.py` at the root; the dashboard stays in `benchmarking/dashboard.py`.

## 5. After deployment

- **URL**: `https://huggingface.co/spaces/USERNAME/SPACE_NAME`
- The dashboard will show:
  - **Overview**, **Results**, **Live prediction**, **Experiments**, **Comparison**, **KG inventory**, **Findings**, **Run benchmarks**, **Hardware readiness**.
- **Data**: Hetionet edges are downloaded automatically on first use of the KG / data-dependent pages. Embeddings and results appear only if you add `data/`, `results/`, or `models/` (e.g. from a run or by uploading to the Space).
- **Run benchmarks on the Space**: The "Run benchmarks" page runs the pipeline with Python (no bash).  
  - **Slim build** (default): the pipeline may fail on missing deps (e.g. `pykeen`). Use **Generate demo results** or upload results from a local run.  
  - **Full pipeline on the Space**: replace `requirements.txt` in the Space repo with the contents of **`requirements-huggingface-full.txt`** (adds `torch`, `pykeen`, `qiskit-algorithms`). Rebuild will take ~15–25 min. If the build runs out of memory, pick a larger CPU in Space settings (e.g. "CPU up to 2").

## Troubleshooting

| Issue | What to do |
|-------|------------|
| Build fails (OOM / timeout) | Use **requirements-huggingface.txt** as in step 3. |
| “App file not found” | Ensure README.md has `app_file: benchmarking/dashboard.py` and that `benchmarking/dashboard.py` exists in the repo. |
| KG / data pages empty | Normal if you haven’t run the pipeline or uploaded `data/` or `results/`. Hetionet edges will still download when the KG is used. |
| “No module named 'kg_layer'” | Ensure the full repo (including `kg_layer/`, `config/`, etc.) is in the Space, not only `benchmarking/`. |

---

**Summary**: Create a Streamlit Space → push this repo (or copy its contents) → optionally switch to `requirements-huggingface.txt` → your app will be at `https://huggingface.co/spaces/USERNAME/SPACE_NAME`.
