# Running the H1/H1b bootstrap CI on NVIDIA DGX Spark

This document covers the end-to-end workflow for computing the **paired-bootstrap confidence intervals** that operationalize hypotheses **H1** (QSVC alone vs each baseline) and **H1b** (stacking ensemble vs each baseline) from `preregistration/osf_preregistration_v1.md` §8.1, on real Hetionet, with QSVC kernel evaluation accelerated via `qiskit-aer-gpu` (cuStateVec).

For the orthogonal workflow of **training** the RotatE 128D embeddings on the DGX, see [DGX_SPARK.md](DGX_SPARK.md). That step writes the `data/rotate_128d_*.npy` files this workflow consumes.

## 1. One-shot script (recommended)

From the repository root, after the prerequisites in §2 are satisfied:

```bash
chmod +x scripts/run_bootstrap_ci_dgx.sh
./scripts/run_bootstrap_ci_dgx.sh
```

What it does:

1. Verifies `qiskit-aer-gpu` / cuStateVec is wired up (`scripts/verify_qiskit_gpu.py`); aborts on failure.
2. Runs `scripts/run_bootstrap_ci.py --gpu` with the locked headline configuration (PauliFeatureMap reps=2, RotatE 128D pair features, hard negatives, 5-fold stratified CV).
3. Tees stdout+stderr to `results/bootstrap_ci_dgx_<timestamp>.log`.
4. Persists per-fold OOF predictions and the manuscript-ready report (see §6).

Optional environment overrides:

```bash
LOG_PATH=/tmp/bootstrap.log ./scripts/run_bootstrap_ci_dgx.sh
RESUME_FROM_CACHE=1 ./scripts/run_bootstrap_ci_dgx.sh   # skip CV training, re-emit report
N_RESAMPLES=2000 ./scripts/run_bootstrap_ci_dgx.sh      # quicker bootstrap (default 10000)
SKIP_ENSEMBLE=1 ./scripts/run_bootstrap_ci_dgx.sh       # debug: H1 only, no H1b
SKIP_QSVC=1 ./scripts/run_bootstrap_ci_dgx.sh           # debug: classical baselines only (no GPU needed)
```

The wrapper picks `$PYTHON` if set, else `.venv/bin/python` if present, else `python3`.

## 2. Prerequisites

### 2.1 CUDA-enabled PyTorch (already covered)

The DGX setup for embedding training in [DGX_SPARK.md](DGX_SPARK.md) §2 also satisfies the PyTorch-CUDA requirement here. If you've already run `./scripts/run_full_embedding_dgx.sh` successfully on this machine, you can skip ahead.

### 2.2 qiskit-aer-gpu (the new piece)

`qiskit-aer-gpu` is **not** in `requirements-full.txt` — install it explicitly, matched to your CUDA version:

```bash
source .venv/bin/activate

# Pick the wheel that matches your CUDA major version. Example for CUDA 12.x:
pip install qiskit-aer-gpu

# The wheel name on PyPI may be qiskit-aer-gpu, qiskit-aer-gpu-cu12, or
# similar depending on the qiskit-aer release. Consult:
#   https://qiskit.org/ecosystem/aer/howtos/running_gpu.html
```

Verify the install succeeded **before** launching a long run:

```bash
python scripts/verify_qiskit_gpu.py
```

This script runs six checks:
1. `nvidia-smi` snapshot (informational)
2. `qiskit` / `qiskit-aer` / `qiskit-machine-learning` versions
3. `AerSimulator.available_devices()` — must include `'GPU'`
4. Aer backends listing
5. A 1-shot statevector simulation actually executing on the GPU device
6. The project's `QuantumExecutor.gpu_available()` (strict — `AerSimulator.available_devices()` membership check)

Exit code 0 means everything is wired up. Non-zero exits print specific remediation per failed check.

## 3. Data the driver needs

| File | Size | Auto-fetched? |
|---|---|---|
| `data/hetionet-v1.0-edges.sif` | ~89 MB | Yes — `kg_layer/kg_loader.load_hetionet_edges()` downloads on first call |
| `data/hetionet-v1.0-nodes.tsv` | ~2.4 MB | Yes |
| `data/rotate_128d_entity_embeddings.npy` | ~24 MB | **No** — must already exist (see below) |
| `data/rotate_128d_relation_embeddings.npy` | ~10 KB | **No** |
| `data/rotate_128d_entity_ids.json` | ~1 MB | **No** |

The cached RotatE 128D embeddings are required and not auto-fetched. Two ways to get them onto the DGX:

**Option A — rsync from another machine that has them** (fastest):

```bash
rsync -avz user@source:/path/to/hybrid-qml-kg-poc/data/rotate_128d_* \
  ./data/
```

**Option B — regenerate via the existing embedding workflow** (clean but ~hour of GPU time):

```bash
./scripts/run_full_embedding_dgx.sh
```

After regeneration, the cached embeddings live under `data/` and the bootstrap CI workflow can use them.

The expected SHA-256 hashes of the Hetionet snapshot are recorded in [`docs/reproducibility/hetionet_snapshot.md`](../reproducibility/hetionet_snapshot.md). To re-verify after sync:

```bash
python scripts/record_hetionet_hash.py
git diff docs/reproducibility/hetionet_snapshot.md   # should be empty
```

## 4. Verify the GPU stack

Already covered in §2.2. Run this whenever you change CUDA driver, qiskit-aer-gpu version, or move to a new machine:

```bash
python scripts/verify_qiskit_gpu.py
```

What success looks like (last lines):

```
=== Summary ===
OK — qiskit-aer-gpu is wired up; you can launch:
  python scripts/run_bootstrap_ci.py --gpu
```

What failure looks like (example — qiskit-aer-gpu not installed):

```
=== Summary ===
FAIL — issues to resolve:
  - AerSimulator.available_devices() does not include 'GPU' — qiskit-aer-gpu either not installed or not seeing CUDA. Install with `pip install qiskit-aer-gpu` against your CUDA version.
  - GPU statevector simulation failed: Simulation device "GPU" is not supported on this system. ...
```

## 5. Launch

```bash
./scripts/run_bootstrap_ci_dgx.sh
```

Or, equivalently, the raw form:

```bash
python scripts/run_bootstrap_ci.py --gpu
```

**Expected wall time:** tens of minutes on cuStateVec with a modern GPU (H100/A100/RTX 6000 Ada-class). On older GPUs it can stretch to ~hour. The dominant cost is QSVC fidelity-kernel matrix evaluation across 5 folds; classical baselines (LR + tuned RF + tuned ET via GridSearchCV) take the same ~5 minutes regardless of backend.

**What success looks like** (last lines of stdout):

```
[ensemble] training OOF meta-learner...

Wrote /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc/docs/results/bootstrap_ci_analysis.md
```

The exit code is 0; the report path is the manuscript-ready artifact.

## 6. Outputs and where they land

| Artifact | Path |
|---|---|
| Per-fold OOF predictions per model | `results/cv_predictions/fold_{0..4}.npz` |
| Manuscript-ready bootstrap CI report | `docs/results/bootstrap_ci_analysis.md` |
| Wrapper's tee'd stdout+stderr log | `results/bootstrap_ci_dgx_<timestamp>.log` |

The report includes:

- Run date, git commit hash, Hetionet snapshot SHA-256
- Quantum backend metadata (`config/quantum_config_gpu.yaml` for the GPU run)
- Bootstrap parameters (10,000 resamples, seed `BOOTSTRAP_SEED = 20260504`, 95% confidence)
- Per-model OOF point estimates (PR-AUC)
- **H1 table:** QSVC alone vs each classical baseline (point estimate, 95% CI, supports H1)
- **H1b table:** stacking ensemble vs each classical baseline (same shape)
- Conjunction-across-baselines decision and overall H1 / H1b verdict
- Notes flagging that R-GCN and TransE are not yet implemented (preregistration §6.2 deferred)

## 7. Long runs and interruption handling

The driver is calendar-time bounded by QSVC kernel evaluation, which can stretch beyond a single SSH session. Defensive habits:

- Use `tmux` or `screen` (or `nohup ... &` + `tail -f`) so SSH drops do not kill the run.
- The driver caches each fold to `results/cv_predictions/fold_{i}.npz` as it completes. A mid-run SIGINT costs at most one fold of work.
- To resume without re-training the folds already cached, use `RESUME_FROM_CACHE=1`:

  ```bash
  RESUME_FROM_CACHE=1 ./scripts/run_bootstrap_ci_dgx.sh
  ```

  This skips the CV training entirely and re-emits the report from cached folds. Useful after a seed change, threshold tweak, or simply to re-render the markdown.

## 8. Troubleshooting

| Symptom | Likely cause | Remediation |
|---|---|---|
| `verify_qiskit_gpu.py`: "AerSimulator.available_devices() does not include 'GPU'" | `qiskit-aer-gpu` not installed or wrong CUDA version | Reinstall `qiskit-aer-gpu` against the CUDA major version `nvidia-smi` reports |
| `verify_qiskit_gpu.py`: "GPU statevector simulation failed: ... CUDA driver" | Driver / wheel version mismatch | Match wheel CUDA major version to driver; sometimes requires `pip install --upgrade qiskit-aer-gpu` |
| Driver aborts immediately with `[gpu] ABORT: --gpu was requested but...` | Same as the verify failure (the `--gpu` gate uses the same check) | Run `verify_qiskit_gpu.py`, fix the issue, re-launch |
| `kg_layer/kg_loader.py` complains it can't find `data/rotate_128d_*.npy` | RotatE embeddings not present | Run `./scripts/run_full_embedding_dgx.sh` first, or rsync the files from another machine — see §3 |
| Driver crashes with CUDA OOM during a QSVC fold | GPU memory exhausted by 16-qubit Pauli kernel matrix | Drop `QSVC_QML_DIM` in `utils/preregistered_constants.py` (deviation requires a §12 amendment), or run with `--subsample <N>` for a debug pass (NOT a headline result) |
| Hetionet hash diff after rsync | Files corrupted in transit | Re-rsync; verify the source file's hash matches `docs/reproducibility/hetionet_snapshot.md` |
| Run completes but H1 booleans look wrong | Sanity check the OOF point estimates row in the report — if QSVC PR-AUC is below 0.5 something is fundamentally broken | Inspect `results/cv_predictions/fold_*.npz` keys and shapes; rerun with `SKIP_ENSEMBLE=1` to bisect |

## 9. Committing the result back to the branch

After a successful run, two artifacts are worth committing:

```bash
git add docs/results/bootstrap_ci_analysis.md
git add results/cv_predictions/fold_0.npz \
        results/cv_predictions/fold_1.npz \
        results/cv_predictions/fold_2.npz \
        results/cv_predictions/fold_3.npz \
        results/cv_predictions/fold_4.npz

# Suggested commit message form
git commit -m "feat(eval): bootstrap CI results from DGX (cuStateVec, $(git rev-parse --short HEAD))"

git push origin roc/preregistration-followups
# Optionally also:
git push qgg roc/preregistration-followups
```

Do not commit the `results/bootstrap_ci_dgx_*.log` log files — they're machine-specific artifacts. Add to `.gitignore` if you want to be explicit, or just don't `git add` them.
