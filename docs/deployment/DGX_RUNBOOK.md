# DGX Spark Runbook

Operational guide for running `hybrid-qml-kg-poc` on NVIDIA DGX Spark (or any
CUDA-equipped Linux box). Complements [`DGX_SPARK.md`](DGX_SPARK.md) (hardware
overview) and [`DGX_BOOTSTRAP_CI.md`](DGX_BOOTSTRAP_CI.md) (paired-bootstrap CI).

---

## 0. One-time setup

```bash
# Clone the repo into a workspace with at least 60 GB free
git clone https://github.com/Quantum-Global-Group/hybrid-qml-kg-poc.git
cd hybrid-qml-kg-poc

# Create a dedicated virtualenv — never install into the system Python
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip

# Install everything (CPU + omics)
pip install -r requirements.txt
bash scripts/dgx/install_gpu_omics.sh           # CPU omics path

# Add RAPIDS for GPU-accelerated single-cell (DGX only):
bash scripts/dgx/install_gpu_omics.sh --gpu
# Override CUDA tag if the default cu122 mismatches the driver:
CUDA_TAG=cu121 bash scripts/dgx/install_gpu_omics.sh --gpu
```

Pick the CUDA tag that matches your `nvidia-smi` output — RAPIDS wheels are
not forward-compatible across major CUDA versions.

---

## 1. Verify the environment

```bash
bash scripts/dgx/check_environment.sh
```

Exits non-zero if any required dependency is missing. Optional items (RAPIDS,
qiskit-aer GPU, gseapy) print `[warn]` but do not fail the run.

Expected output skeleton on a complete DGX setup:

```
=== Environment Check: hybrid-qml-kg-poc ===
--- Python ---
  [OK]   python3 >= 3.9
  [OK]   pip available
--- Core scientific stack ---
  [OK]   numpy
  [OK]   pandas
  ...
--- GPU / RAPIDS (optional — DGX Spark only) ---
  [OK]   cupy
  [OK]   rapids-singlecell
  [OK]   CUDA device count > 0
...
=== Summary: 30 passed, 0 failed, 0 optional warnings ===
```

---

## 2. Smoke test (M7 CI gate)

```bash
bash scripts/dgx/run_smoke_test.sh
```

Runs 6 steps end-to-end against synthetic data:
1. Import all new layers
2. Load HetionetResolver (47k nodes)
3. Reversal score unit check (`compute_reversal_score` round-trip)
4. Evidence fusion + explanation
5. Validation (known indication + DrugBank seed)
6. Full pipeline orchestration (writes `artifacts/predictions/`)

If step 6 fails on a fresh DGX, check that `data/hetionet-v1.0-edges.sif` is
present — the embedder skips KG training on missing data and falls back to
the synthetic candidate set.

---

## 3. Headline experiment (PR-AUC 0.7987 reproduction)

```bash
python scripts/run_optimized_pipeline.py \
    --relation CtD \
    --embedding_method RotatE \
    --embedding_dim 32 \
    --negative_sampling hard_degree_corrupt \
    --ensemble_method stacking
```

All defaults are locked in `utils/preregistered_constants.py`. Override only
for sensitivity analyses (e.g. `--qsvc_feature_map_type ZZ` for the Pauli vs
ZZ comparison).

Expected wallclock on DGX Spark:
- KG embedding training: 2-5 min (RotatE, 32d, 200 epochs)
- Quantum kernel computation: 5-15 min
- Classical baselines + stacking: < 1 min
- Total: ~10-20 min

---

## 4. Full repurposing pipeline

```bash
# Baseline preserving (kg-only zero-fills omics features)
python scripts/run_full_repurposing_pipeline.py --mode kg-only

# Full stack with reversal + clinical validation
python scripts/run_full_repurposing_pipeline.py \
    --mode kg+omics --validate --top-n 50

# Single-disease deep-dive
python scripts/run_full_repurposing_pipeline.py \
    --mode kg+omics --validate \
    --disease 'Disease::DOID:9352' --top-n 20
```

Outputs land in `artifacts/predictions/`:
- `top_candidates.csv` / `top_candidates.json`
- `run_summary.json` (mode, top compound, tier distribution)
- `final_repurposing_report.md`

---

## 5. Dashboards & notebooks

```bash
# Streamlit dashboard (binds to 0.0.0.0 for remote access)
bash scripts/dgx/launch_dashboard.sh
# → http://<dgx-host>:8501

# Jupyter Lab for the playbooks
bash scripts/dgx/launch_jupyter.sh
# → http://<dgx-host>:8888
```

Recommended walkthrough order:
1. [`playbooks/00_environment_check.ipynb`](../../playbooks/00_environment_check.ipynb)
2. [`playbooks/01_single_cell_preprocessing.ipynb`](../../playbooks/01_single_cell_preprocessing.ipynb)
3. [`playbooks/02_disease_signature_generation.ipynb`](../../playbooks/02_disease_signature_generation.ipynb)
4. [`playbooks/03_drug_signature_reversal.ipynb`](../../playbooks/03_drug_signature_reversal.ipynb)
5. [`playbooks/04_kg_embedding_training.ipynb`](../../playbooks/04_kg_embedding_training.ipynb)
6. [`playbooks/05_hybrid_qml_prediction.ipynb`](../../playbooks/05_hybrid_qml_prediction.ipynb)
7. [`playbooks/06_candidate_ranking_and_validation.ipynb`](../../playbooks/06_candidate_ranking_and_validation.ipynb)
8. [`playbooks/07_dashboard_demo.ipynb`](../../playbooks/07_dashboard_demo.ipynb)

---

## 6. Single-cell pipeline (alone)

```bash
bash scripts/dgx/run_single_cell_pipeline.sh \
    --input data/single_cell/my_dataset.h5ad \
    --disease 'Disease::DOID:9352' \
    --backend gpu
```

Falls back to a synthetic demo if `--input` is missing. Writes:
- `artifacts/single_cell/qc/qc_report.md`
- `artifacts/signatures/disease_signature.json`

---

## 7. Collecting artifacts

```bash
bash scripts/dgx/collect_artifacts.sh
```

Bundles `artifacts/` into `artifacts_YYYYMMDD_HHMMSS.tar.gz` with a
SHA256 manifest at `artifacts/MANIFEST_SHA256.txt`. Use this for OSF
preprint uploads and reproducibility verification.

---

## 8. Troubleshooting

### `rapids-singlecell` imports but `cp.cuda.runtime.getDeviceCount() == 0`
Driver/CUDA mismatch. Run `nvidia-smi` to check the driver's reported CUDA
version, then re-install with the matching tag:
```bash
CUDA_TAG=cu121 bash scripts/dgx/install_gpu_omics.sh --gpu
```

### Smoke test step 6 fails with "No upstream KG+QML scores found"
This is normal on a fresh checkout — the pipeline correctly falls back to
the demo candidate set. The step still passes; the warning is informational.

### Pipeline hangs on quantum kernel computation
Quantum kernel computation is O(N²) in the number of test pairs. Reduce the
test set size or switch to the CPU statevector simulator:
```bash
QISKIT_BACKEND=statevector_cpu python scripts/run_optimized_pipeline.py ...
```

### ClinicalTrials.gov 5xx errors during `--validate`
The validation layer logs and continues on transient API errors. Re-run only
the validation phase via the dashboard's "Clinical validation" page once
ClinicalTrials.gov is back.

### Streamlit dashboard shows "No candidates found"
The dashboard reads `artifacts/predictions/top_candidates.csv`. Run the full
pipeline at least once:
```bash
python scripts/run_full_repurposing_pipeline.py --mode kg+omics
```

---

## 9. Performance budget (DGX Spark reference)

| Stage | Wallclock | Notes |
|-------|-----------|-------|
| `check_environment.sh` | 5-10 s | |
| `run_smoke_test.sh` | 30-60 s | Synthetic data |
| KG embedding training (full Hetionet) | 2-5 min | GPU |
| Quantum kernel computation | 5-15 min | Statevector + GPU |
| Single-cell QC + clustering (10k cells) | 30-60 s | RAPIDS |
| Single-cell QC + clustering (100k cells) | 5-10 min | RAPIDS |
| Disease signature build | 10-30 s | Per cell type |
| Reversal scoring (1k drugs × 1 disease) | 5 s | Pure Python sets |
| Evidence fusion | < 1 s | |
| Full pipeline (no quantum recompute) | 1-2 min | With cached embeddings |

If a stage exceeds 3× these budgets, check `nvidia-smi` (driver state),
shared filesystem latency, or the `OMP_NUM_THREADS` env var (pin to 1 inside
quantum kernels to avoid thread contention).
