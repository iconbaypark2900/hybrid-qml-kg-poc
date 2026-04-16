# Quantum Path — Scaling and Improvement

**Status:** QSVC works but is near practical limits; VQC near-random; scaling unsolved  
**Key constraint:** Full fidelity quantum kernel is O(n²) — infeasible beyond ~1,500 pairs

---

## Running on NVIDIA DGX Spark (GPU)

Run heavy experiments on a CUDA host (e.g. DGX Spark) so **PyKEEN embeddings** and **quantum simulation** can both use hardware acceleration.

| Layer | What uses the GPU | Setup |
|-------|-------------------|--------|
| KG embeddings | PyTorch / PyKEEN | Install **CUDA-enabled PyTorch**; training uses GPU when `torch.cuda.is_available()`. See [docs/deployment/DGX_SPARK.md](../deployment/DGX_SPARK.md). For full-graph RotatE, use `./scripts/run_full_embedding_dgx.sh` from the repo root. |
| Quantum kernels | Qiskit Aer **cuStateVec** | Set `HYBRID_QML_SYSTEM=dgx` so `scripts/run_optimized_pipeline.py` auto-selects `config/quantum_config_dgx.yaml`, **or** pass `--gpu` (selects `config/quantum_config_gpu.yaml`). Requires a GPU-capable Aer build (`qiskit-aer-gpu` or equivalent). |

**Copy-paste convention for commands below:** each `python scripts/run_optimized_pipeline.py` line is prefixed with `HYBRID_QML_SYSTEM=dgx` so quantum execution uses `config/quantum_config_dgx.yaml` (GPU simulator on DGX). **Do not combine with `--gpu` on the same command:** `--gpu` forces `config/quantum_config_gpu.yaml` and overrides any `--quantum_config_path`. For GPU sim without the `dgx` env var, use `--gpu` alone (equivalent backend class; different YAML).

If you pass **`--quantum_config_path`** explicitly to something other than the default `config/quantum_config.yaml`, **`HYBRID_QML_SYSTEM=dgx` is not applied** (the auto-select only runs when the path stayed at the default). In that case set the path to `config/quantum_config_dgx.yaml` yourself for DGX.

---

## 1. Nyström Approximation — Enable Larger Relations

### Problem

The full fidelity quantum kernel matrix requires `n_train² / 2` circuit
evaluations. For CtD (755 positives → 1,208 training pairs), this is ~729K
evaluations taking 43 minutes on CPU. For CbG (11,571 positives → ~18,500
training pairs), it would require ~171M evaluations — weeks of CPU time.

### Solution: Nyström approximation

Select `m` landmark points, compute `m × n_train` kernel evaluations instead
of `n_train²`. Accuracy degrades gracefully with `m`.

**The flag exists:** `--qsvc_nystrom_m 200`

**What is missing:** A systematic sweep of `m` values to find the accuracy/
speed tradeoff:

```bash
for m in 50 100 200 400 800; do
  HYBRID_QML_SYSTEM=dgx python scripts/run_optimized_pipeline.py --relation CtD \
    --full_graph_embeddings --embedding_method RotatE \
    --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
    --quantum_only --qml_dim 16 --qml_feature_map Pauli \
    --qml_feature_map_reps 2 --qsvc_C 0.1 \
    --qsvc_nystrom_m $m \
    --results_dir results/nystrom_sweep_m$m
done
```

**Expected output:** Table of (m, PR-AUC, kernel_time) showing the knee of
the accuracy/speed curve. This table should be added to the paper.

**Full kernel reference:** m=None, PR-AUC=0.6343, time=2,619s

### Application to CbG

Once the Nyström sweep determines an acceptable `m` (e.g. 400), the CbG
relation becomes feasible:

```bash
HYBRID_QML_SYSTEM=dgx python scripts/run_optimized_pipeline.py --relation CbG \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --quantum_only --qml_dim 16 --qml_feature_map Pauli \
  --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --qsvc_nystrom_m 400 \
  --results_dir results/cbg_nystrom
```

---

## 2. VQC Architecture Search

### Problem

VQC results are near-random (~0.54–0.55 PR-AUC). The current ablation used:
- 8 qubits
- 50 iterations (SPSA / COBYLA / NFT)
- RealAmplitudes / EfficientSU2 ansatze with reps=3–4

This is a minimal search. Three likely culprits:

1. **Iteration budget too low** — SPSA on QSVC-scale problems typically needs 200–500 iterations
2. **Ansatz too shallow** — reps=4 with 8 qubits may not have enough expressivity
3. **Barren plateau** — gradient vanishes at initialization; warm-start or layerwise training helps

### What to do

**Step 1: Increase iteration budget**

```bash
HYBRID_QML_SYSTEM=dgx python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --vqc_only --qml_dim 8 --vqc_optimizer SPSA \
  --vqc_max_iter 200 --vqc_ansatz RealAmplitudes --vqc_reps 6 \
  --results_dir results/vqc_200iter
```

**Step 2: Try layerwise training**

Train reps=2 first until convergence, then add more layers and continue.
This is the layerwise learning protocol (LLP) which avoids barren plateaus.

**Step 3: Warm-start from QSVC weights**

Use the QSVC's support vectors to initialize the VQC parameter values before
the variational optimization loop.

**Files to use:**
- `quantum_layer/quantum_variational_feature_selection.py`
- `quantum_layer/quantum_circuit_optimization.py`
- `quantum_layer/circuit_optimizer.py`

---

## 3. Variational Quantum Kernel Learning

### Problem

The current QSVC uses a *fixed* feature map (Pauli or ZZ). The encoding
circuit's geometry is not learned — the kernel is defined before seeing any data.

### What this is

Variational quantum kernel learning (VQKL) parameterizes the encoding circuit
`U(x; θ)` and jointly trains `θ` to maximize kernel-target alignment (KTA)
while simultaneously training the SVM. The result is a kernel tuned to the
specific label structure of the biomedical link prediction task.

### Why this is promising

The Pauli inversion effect (§6.3 of the paper) shows that kernel geometry
matters more than standalone QSVC accuracy for ensemble diversity. VQKL
optimizes geometry directly.

**File:** `quantum_layer/quantum_kernel_engineering.py` — likely has stubs
for this. Audit before implementing.

**Implementation sketch:**

```python
# Pseudocode
for epoch in range(n_epochs):
    K = compute_quantum_kernel(X_train, X_train, feature_map_params=theta)
    kta = kernel_target_alignment(K, y_train)
    grad_theta = estimate_gradient(kta, theta)  # parameter shift rule
    theta = theta + lr * grad_theta
    # After kernel training: fit QSVC with learned K
svm = SVC(kernel='precomputed').fit(K, y_train)
```

---

## 4. GPU-Accelerated Simulation

### Current state

All primary runs use CPU statevector simulation. `config/quantum_config_gpu.yaml`
and `config/quantum_config_dgx.yaml` exist; `--gpu` and `HYBRID_QML_SYSTEM=dgx`
select GPU-backed simulation via cuStateVec (NVIDIA).

### What is missing

No GPU benchmark result exists. The expected speedup for 16-qubit statevector
is approximately 10–50× depending on the GPU, which would reduce the 43-minute
kernel computation to ~1–4 minutes.

**What to run** (DGX: uses `quantum_config_dgx.yaml`; same numerics as `--gpu` + `quantum_config_gpu.yaml`):

```bash
HYBRID_QML_SYSTEM=dgx python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE \
  --embedding_dim 128 --embedding_epochs 200 --negative_sampling hard \
  --qml_dim 16 --qml_feature_map Pauli --qml_feature_map_reps 2 \
  --qsvc_C 0.1 --qml_pre_pca_dim 24 \
  --run_ensemble --ensemble_method stacking --tune_classical \
  --results_dir results/gpu_run
```

**Expected result:** Same PR-AUC as CPU run (numerically identical for
statevector), much shorter kernel computation time. Report both in the paper
as evidence that the result is reproducible at different speeds.

---

## 5. Quantum Error Mitigation Evaluation

### Current state

`quantum_layer/quantum_error_mitigation.py` and
`quantum_layer/advanced_error_mitigation.py` exist but are not evaluated
against real results.

### What to evaluate

| Technique | When to use | Expected benefit |
|-----------|-------------|-----------------|
| ZNE (zero-noise extrapolation) | Noisy simulator and hardware | +2–5 pp PR-AUC recovery |
| Readout mitigation | Hardware | Corrects measurement bit-flip errors |
| Pauli twirling | Hardware | Converts coherent errors to stochastic |

**Experiment design:**
1. Run noisy simulator without mitigation → baseline noisy PR-AUC
2. Apply ZNE → PR-AUC with ZNE
3. Apply readout mitigation → PR-AUC with readout mitigation
4. Apply both → PR-AUC with full mitigation

This provides the noise mitigation ablation table for paper §6.5.

---

## 6. Feature Map Expansion

### Problem

Only ZZ and Pauli feature maps have been evaluated. The Qiskit circuit library
has others. `quantum_layer/quantum_feature_maps.py` may have additional
implementations.

### Candidates to evaluate

| Feature Map | Description | Circuit depth |
|-------------|-------------|---------------|
| ZZ (reps=2) | Second-order Pauli-Z | Shallow |
| ZZ (reps=3) | As above, deeper | Medium |
| Pauli (reps=1) | Mixed Pauli, less repetition | Shallow |
| Pauli (reps=2) | Current best ensemble performer | Medium |
| Pauli (reps=3) | Deeper Pauli | Deep |
| IQP-like | Instantaneous Quantum Polynomial | Very shallow |

The key metric for feature map selection in this context is not standalone
QSVC PR-AUC but **classical-quantum prediction decorrelation** — measure
Pearson correlation between QSVC predictions and RF predictions on the test
set. Lower correlation = better ensemble candidate.

---

## Summary — effort vs impact

| Work item | Effort | Impact |
|-----------|--------|--------|
| Nyström sweep (m values) | Low — bash loop | Enables CbG, DaG at scale |
| GPU benchmark | Low — config flag | Faster iteration on all experiments |
| VQC iteration budget increase | Low | Likely lifts VQC from ~0.55 to 0.60+ |
| KTA-based feature map selection | Medium | Replaces 43-min sweeps with 2-min screens |
| Noisy simulator + ZNE ablation | Medium | Fills ideal→hardware gap |
| VQKL (variational kernel learning) | High | Potentially large ensemble gain |
