# Running full-graph embeddings on NVIDIA DGX Spark

This repo trains KG embeddings with **PyKEEN**; training uses **GPU automatically** when `torch.cuda.is_available()` (see `kg_layer/advanced_embeddings.py`). On a CPU-only PyTorch install, embedding training stays on CPU and is very slow.

## 1. One-shot script (recommended)

From the repository root (after dependencies are installed):

```bash
chmod +x scripts/run_full_embedding_dgx.sh
./scripts/run_full_embedding_dgx.sh
```

- Writes a timestamped log under `results/full_embedding_dgx_<stamp>.log` (override with `LOG_PATH=...`).
- Uses **full-graph RotatE**, **128** dims, **200** epochs, **hard** negatives, **`--classical_only`** (skips quantum training after embeddings).
- Points quantum config at `config/quantum_config_dgx.yaml` for consistency on GPU systems (classical-only runs still load config paths used elsewhere).

Environment overrides (optional):

```bash
EMBEDDING_EPOCHS=50 EMBEDDING_DIM=128 ./scripts/run_full_embedding_dgx.sh
LOG_PATH=/tmp/embed.log ./scripts/run_full_embedding_dgx.sh
```

## 2. Install PyTorch with CUDA on the DGX

Install a **CUDA build** of PyTorch that matches the **NVIDIA driver / CUDA** on the machine. Use the official matrix: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example (your CUDA version may differ — pick the matching `cu12x` wheel from the site):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Then verify:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

You should see `True` and a GPU name. Pipeline logs should include `PyKEEN training device: cuda`.

## 3. Data and outputs

- **Input:** Hetionet edges under `data/` (downloaded by the loader if missing).
- **Artifacts:** Embeddings and checkpoints under `data/` (per `AdvancedKGEmbedder` `work_dir`); run metrics and JSON under `results/` (see `results_dir`).

## 4. Long runs

Use `tmux`, `screen`, or a job scheduler so SSH disconnects do not kill training. The script itself does not daemonize.
