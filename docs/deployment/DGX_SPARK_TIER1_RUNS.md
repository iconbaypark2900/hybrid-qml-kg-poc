# Running Tier 1 Experiments on NVIDIA DGX Spark

**Purpose:** The MoA benchmark, CpD relation run, and multi-seed evaluation all
died on CPU after 7+ hours of embedding training. This guide gets them done on
DGX Spark in a fraction of the time using GPU-accelerated PyKEEN and Qiskit Aer.

**Expected wall time on DGX Spark (vs CPU):**

| Run | CPU time | DGX time |
|-----|----------|----------|
| RotatE embedding (200 epochs) | ~15 hours | ~45 min |
| QSVC full kernel (CtD) | ~43 min | ~2–4 min |
| QSVC Nyström m=200 | ~2 min | ~20 sec |
| All Tier 1 runs combined | days | ~4 hours |

---

## Step 1 — Connect to the DGX and Clone

```bash
ssh <your-dgx-hostname>
# or from WSL:
# ssh roc@<dgx-ip>

cd ~
git clone https://github.com/iconbaypark2900/hybrid-qml-kg-poc.git
cd hybrid-qml-kg-poc
```

If the repo is already cloned, pull the latest:

```bash
cd ~/hybrid-qml-kg-poc
git pull origin main
```

---

## Step 2 — Set Up the Python Environment

> **WARNING — read before running pip install.**
> Installing PyTorch downloads a ~2.5 GB wheel and expands it in memory.
> On a shared DGX this can freeze or reboot the machine if done naively.
> Follow the steps below exactly to stay within safe memory limits.

### 2a — Check available RAM first

```bash
free -h
```

You need at least **8 GB free** before running pip. If free memory is low,
check who else is using the machine:

```bash
who         # other logged-in users
nvidia-smi  # GPU processes
top -b -n1 | head -20   # top RAM consumers
```

If memory is tight, coordinate with other users or wait for their jobs to finish
before installing.

### 2b — Download the wheel separately before installing

This splits the download and install into two steps so pip never holds
the full wheel + extracted files in RAM at the same time.

```bash
# Check your CUDA version
nvidia-smi | grep "CUDA Version"

# Create a download directory on local disk (not RAM)
mkdir -p ~/torch_wheels && cd ~/torch_wheels

# Download the wheel without installing (pick cu124 or cu121 to match your driver)
pip download torch \
  --index-url https://download.pytorch.org/whl/cu124 \
  --no-deps \
  -d ~/torch_wheels

# Go back to the project
cd ~/hybrid-qml-kg-poc
```

### 2c — Create the venv and install from the local wheel

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install torch from the local wheel file (no network, no extra decompression RAM spike)
pip install ~/torch_wheels/torch-*.whl

# Install remaining dependencies one chunk at a time to avoid RAM spikes
pip install pykeen networkx scikit-learn numpy pandas
pip install qiskit qiskit-aer
pip install -r requirements.txt
```

If `pip install -r requirements.txt` still causes a freeze, install in smaller
batches — open `requirements.txt` and install 5–10 packages at a time.

Verify GPU is visible before running anything:

```bash
python3 -c "
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
else:
    print('ERROR: CUDA not available — stop here and fix the PyTorch install')
"
```

You must see `CUDA available: True` before proceeding.

Also verify Qiskit Aer GPU support:

```bash
python3 -c "
from qiskit_aer import AerSimulator
gpu_sim = AerSimulator(method='statevector', device='GPU')
print('Aer GPU simulator OK:', gpu_sim)
"
```

If this errors with `device GPU is not supported`, install the GPU Aer build:

```bash
pip install qiskit-aer-gpu
```

---

## Step 3 — Open a tmux Session

All runs take 30–90 minutes each. Use tmux so SSH disconnects do not kill them.

```bash
tmux new -s tier1
# To reattach later: tmux attach -t tier1
# To detach without killing: Ctrl-B then D
```

Inside tmux, activate the venv:

```bash
source .venv/bin/activate
export HYBRID_QML_SYSTEM=dgx   # tells the pipeline to use config/quantum_config_dgx.yaml
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
```

Confirm the GPU env var is set — every command below assumes it:

```bash
echo $HYBRID_QML_SYSTEM   # should print: dgx
```

---

## Step 4 — Run the MoA Benchmark

> **Why the previous runs failed:** The pipeline was re-training RotatE from
> scratch every run (15+ hours on CPU). Pre-trained 128d embeddings already
> exist at `data/rotate_128d_entity_embeddings.npy` from earlier successful runs.
> The flags `--use_cached_embeddings --allow_cached_embeddings_with_holdout`
> tell the pipeline to load them directly, skipping embedding training entirely.
> On DGX the only remaining compute is the QSVC kernel (~2–4 min on GPU).

```bash
mkdir -p results/moa_benchmark

nohup python3 scripts/run_optimized_pipeline.py \
  --relation CtD \
  --full_graph_embeddings \
  --embedding_method RotatE \
  --embedding_dim 128 \
  --embedding_epochs 200 \
  --negative_sampling hard \
  --qml_dim 16 \
  --qml_feature_map Pauli \
  --qml_feature_map_reps 2 \
  --qsvc_C 0.1 \
  --qml_pre_pca_dim 24 \
  --run_ensemble \
  --ensemble_method stacking \
  --tune_classical \
  --use_moa_features \
  --use_cached_embeddings \
  --allow_cached_embeddings_with_holdout \
  --results_dir results/moa_benchmark \
  > results/moa_benchmark/run_dgx.log 2>&1 &

echo "MoA PID: $!"
```

Monitor progress:

```bash
tail -f results/moa_benchmark/run_dgx.log
```

Look for these lines in order — each confirms a stage completed:

```
Using cached embeddings          ← skipped retraining, loaded from data/
QSVC kernel computation complete ← quantum kernel done (~2-4 min on GPU)
Ensemble PR-AUC: 0.XXXX          ← final result
```

If you see `Training epochs on cpu` instead, the cache was not found or CUDA
is not active — see Troubleshooting below.

---

## Step 5 — Run the CpD Relation

Start this in a **second tmux pane** (`Ctrl-B then %` to split, or `Ctrl-B then C`
for a new window) while MoA is running. CpD uses the same cached embeddings so
it also skips retraining.

```bash
mkdir -p results/cpd_run

nohup python3 scripts/run_optimized_pipeline.py \
  --relation CpD \
  --full_graph_embeddings \
  --embedding_method RotatE \
  --embedding_dim 128 \
  --embedding_epochs 200 \
  --negative_sampling hard \
  --qml_dim 16 \
  --qml_feature_map Pauli \
  --qml_feature_map_reps 2 \
  --qsvc_C 0.1 \
  --qml_pre_pca_dim 24 \
  --run_ensemble \
  --ensemble_method stacking \
  --tune_classical \
  --use_cached_embeddings \
  --allow_cached_embeddings_with_holdout \
  --results_dir results/cpd_run \
  > results/cpd_run/run_dgx.log 2>&1 &

echo "CpD PID: $!"
```

Monitor:

```bash
tail -f results/cpd_run/run_dgx.log
```

---

## Step 6 — Run the Multi-Seed Evaluation

Seed 42 is already done (`results/multiseed/seed_42/`). Run the remaining 4 seeds.
Nyström (`--qsvc_nystrom_m 200`) keeps each quantum kernel to ~20 sec on GPU.
Cached embeddings skip retraining for all 4 seeds.

```bash
for seed in 7 13 99 2026; do
  echo "=== Starting seed $seed ==="
  mkdir -p results/multiseed/seed_$seed
  python3 scripts/run_optimized_pipeline.py \
    --relation CtD \
    --full_graph_embeddings \
    --embedding_method RotatE \
    --embedding_dim 128 \
    --embedding_epochs 200 \
    --negative_sampling hard \
    --qml_dim 16 \
    --qml_feature_map Pauli \
    --qml_feature_map_reps 2 \
    --qsvc_C 0.1 \
    --qml_pre_pca_dim 24 \
    --run_ensemble \
    --ensemble_method stacking \
    --tune_classical \
    --qsvc_nystrom_m 200 \
    --use_cached_embeddings \
    --allow_cached_embeddings_with_holdout \
    --random_state $seed \
    --results_dir results/multiseed/seed_$seed \
    2>&1 | tee results/multiseed/seed_$seed.log
  echo "=== Seed $seed done ==="
done
```

After all 5 seeds exist, aggregate the results:

```bash
python3 scripts/aggregate_multiseed.py \
  --results-dir results/multiseed \
  --seeds 42 7 13 99 2026 \
  --out results/multiseed/summary.json
```

This writes mean ± std PR-AUC for RF, ET, QSVC, and Ensemble across all 5 seeds.

---

## Step 7 — Verify Results

Check that each run produced a JSON result (not just a log):

```bash
# MoA benchmark
ls results/moa_benchmark/*.json 2>/dev/null && echo "MoA: OK" || echo "MoA: MISSING"

# CpD run
ls results/cpd_run/*.json 2>/dev/null && echo "CpD: OK" || echo "CpD: MISSING"

# Multi-seed (should have 5)
ls results/multiseed/seed_*/optimized_results_*.json 2>/dev/null | wc -l
# Expected: 5

# Quick metrics summary
for f in results/moa_benchmark/optimized_results_*.json; do
  python3 -c "
import json, sys
d = json.load(open('$f'))
print('MoA ensemble PR-AUC:', d.get('ensemble_pr_auc', d.get('pr_auc', 'key not found')))
"
done

for f in results/cpd_run/optimized_results_*.json; do
  python3 -c "
import json
d = json.load(open('$f'))
print('CpD ensemble PR-AUC:', d.get('ensemble_pr_auc', d.get('pr_auc', 'key not found')))
"
done
```

---

## Step 8 — Push Results Back

Once all runs are complete, commit and push results to `origin` (iconbaypark2900):

```bash
cd ~/hybrid-qml-kg-poc

git add \
  results/moa_benchmark/ \
  results/cpd_run/ \
  results/multiseed/

git commit -m "Add MoA, CpD, and multi-seed benchmark results from DGX Spark

- MoA benchmark: CtD + --use_moa_features, RotatE 128d 200 epochs, Pauli QSVC
- CpD run: CpD relation, same config as headline CtD run
- Multi-seed: seeds 42 7 13 99 2026, Nystrom m=200 for quantum kernel
- All runs: HYBRID_QML_SYSTEM=dgx, GPU statevector Aer

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push origin main
```

Do NOT push to the `qgg` remote — that is the Quantum-Global-Group fork.

---

## If the Machine Crashed During pip install

The DGX likely OOM-killed itself because pip downloaded and extracted a large
wheel while other processes were also using RAM. Recovery steps:

```bash
# 1. SSH back in after the machine reboots
ssh <your-dgx-hostname>
cd ~/hybrid-qml-kg-poc

# 2. Remove any partial venv from the failed attempt
rm -rf .venv ~/torch_wheels

# 3. Check how much RAM is free NOW before trying again
free -h
# You need at least 8 GB free

# 4. Check who else is on the machine and what GPU jobs are running
who && nvidia-smi

# 5. Follow the RAM-safe install in Step 2 above (download wheel first, then install)
```

If the DGX is a shared machine administered by someone else (NVIDIA, MDC, etc.),
ask them whether a Python environment with PyTorch + CUDA is already installed
system-wide or available as a module:

```bash
module avail 2>/dev/null | grep -i torch   # if using environment modules
conda env list 2>/dev/null                 # if conda is available
```

Using an existing system environment avoids the pip install entirely.

---

## Troubleshooting

### Still seeing "Training epochs on cpu" despite --use_cached_embeddings

The cache flags were set but the pipeline is still retraining. Two causes:

**1. The cache file is missing on DGX.** The pre-trained embeddings live in `data/`
and are not committed to git (too large). Copy them from your local machine:

```bash
# From your local WSL machine — copy data/ to DGX
scp -r /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc/data/rotate_128d_* \
  <dgx-user>@<dgx-hostname>:~/hybrid-qml-kg-poc/data/

# Verify they arrived
ls ~/hybrid-qml-kg-poc/data/rotate_128d_*
# Should see: rotate_128d_entity_embeddings.npy, rotate_128d_entity_ids.json, rotate_128d_relation_embeddings.npy
```

**2. The leakage guard blocked the cache.** If you see the warning
`Leakage guard: hold-out set detected; disabling cached embeddings`, make sure
you included BOTH flags:

```bash
--use_cached_embeddings --allow_cached_embeddings_with_holdout
```

Both are required. `--use_cached_embeddings` alone is blocked by the leakage guard.

---

### CUDA not available after installing CUDA PyTorch

Check that the installed wheel matches the system CUDA version:

```bash
nvidia-smi | grep "CUDA Version"        # system driver max CUDA
python3 -c "import torch; print(torch.version.cuda)"  # torch built-for CUDA
```

They do not need to match exactly — the driver version must be >= the torch CUDA
version. If the driver is older than the wheel, install an older torch wheel:

```bash
# For driver CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Out of GPU memory during QSVC kernel

The 16-qubit statevector requires ~512 MB of GPU RAM. If you see an OOM error,
either reduce qubits or force CPU Aer by removing `HYBRID_QML_SYSTEM=dgx` and
using `--fast_mode` to reduce the dataset size for a smoke check.

### Qiskit Aer does not find GPU

```bash
python3 -c "from qiskit_aer import AerSimulator; print(AerSimulator().available_devices())"
```

If `GPU` is not listed, install the GPU Aer build:

```bash
pip uninstall qiskit-aer -y
pip install qiskit-aer-gpu
```

### PyKEEN still training on CPU despite CUDA being available

Check that the CUDA torch is installed **inside the active venv**, not system-wide:

```bash
which python3         # should be inside .venv/
python3 -c "import torch; print(torch.cuda.is_available())"
```

If `False`, reinstall torch inside the venv explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

### Embedding already cached from a previous run

If a `data/rotate_128d_*` cache file exists from a previous failed CPU run, the
pipeline may skip re-training and use the partial cache. To force a fresh embedding:

```bash
rm -rf data/rotate_128d_*
```

Then re-run.

---

## What to Do After the Runs Complete

1. Push results back (Step 8 above).
2. Open `docs/paper.tex` and update Table 3:
   - Add MoA PR-AUC row (compare vs baseline 0.7987)
   - Add CpD PR-AUC row in the multi-relational section
   - Replace single-seed ensemble value with `mean ± std` from the 5-seed summary
   - Add degree-heuristic row (already computed: PR-AUC = 0.7169, see
     `results/degree_heuristic_baseline.json`)
   - Add random baseline row: PR-AUC = 0.50
3. Fix the `[?]` citation keys in `docs/paper.tex` (see `docs/MASTER_WORKPLAN.md` §1.4).
4. Compile the PDF: `cd docs && pdflatex paper.tex && pdflatex paper.tex`

The full Tier 1 checklist and all subsequent tiers are documented in
[docs/MASTER_WORKPLAN.md](../MASTER_WORKPLAN.md).
