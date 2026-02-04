# Dashboard & Presets Cheat Sheet

This project includes a Streamlit dashboard (`benchmarking/dashboard.py`) that can run benchmark presets and visualize results.

## Run the dashboard

From the project root:

```bash
cd /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc
source .venv/bin/activate
streamlit run benchmarking/dashboard.py
```

If you prefer not to `activate`:

```bash
cd /home/roc/quantumGlobalGroup/hybrid-qml-kg-poc
PATH="/home/roc/quantumGlobalGroup/hybrid-qml-kg-poc/.venv/bin:$PATH" streamlit run benchmarking/dashboard.py
```

## Where results go

- Latest: `results/latest_run.csv`
- History: `results/experiment_history.csv`
- QSVC predictions: `results/predictions_QSVC_*.csv`

## Presets: how to fill the dashboard fields

Open the dashboard → **Run benchmarks**.

### Common fields (applies to all presets)

- **Relation**: `CtD`
- **Max entities**:
  - **Recommended**: `0` (no cap; use all available CtD)
  - If you want to limit runtime: `2000` (CtD typically saturates below this anyway)
- **Seeds (comma-separated)**:
  - **Quick**: `42,123,456`
  - **More stable**: `42,123,456,789,1011`
- **Run both ideal + noisy**:
  - **ON** only when comparing quantum execution modes
  - **OFF** when running classical-only presets

### Dropdown options (what they mean)

- **QSVC Nyström landmarks (m)**:
  - `m=32` is the best default for speed vs fidelity.
  - `off (full kernel)` is much slower for larger training sets.
- **Quick run: sample positive edges**:
  - `400` gives a good “fast but meaningful” run.
  - `(no sampling)` uses all positives and is the most expensive.
- **Negative sampling**:
  - `hard` is recommended for realistic, non-trivial negatives.
  - `diverse` mixes hard+random (useful for coverage).
- **neg_ratio**:
  - `2.0` means 2 negatives per positive (often a better stress test than 1:1).
- **Calibrate probabilities (classical)**:
  - Optional. Helps probability quality; may add small runtime.

## Preset-by-preset cheat sheet (recommended settings)

### 1) Classical ceiling (CV, bigger data, full-graph)

**Goal**: estimate the best achievable classical PR-AUC (variance-reduced).

- **Max entities**: `0`
- **Qubits to run**: `[12]` (ignored by classical-only, leave as-is)
- **Run both ideal + noisy**: **OFF**
- **Dropdowns**:
  - **Negative sampling**: `(preset default)` or `hard`
  - **neg_ratio**: `(preset default)` or `2.0`
  - **QSVC Nyström** / **pos-edge sampling**: irrelevant; leave defaults

Recommended first run:
- Keep defaults, click **Run preset** once.

### 2) Recommended sweep (qubits 6/8/12 × seeds, ideal+noisy)

**Goal**: primary quantum vs classical comparison sweep (variance-aware).

- **Max entities**: `0`
- **Qubits to run**: `[6,8,12]`
- **Run both ideal + noisy**: **ON**
- **Dropdowns**:
  - **QSVC Nyström**: `m=32`
  - **Negative sampling**: `(preset default)` or `hard`
  - **neg_ratio**: `(preset default)` or `2.0`
  - **Sample positives**:
    - For speed: `400`
    - For full-run: `(no sampling)`

### 3) Quantum vs classical (same data, repeated seeds)

**Goal**: compare QSVC vs classical on the same data across multiple seeds.

- **Max entities**: `0`
- **Qubits to run**: `[12]` (or `[6,8,12]` if you want)
- **Run both ideal + noisy**: **ON**
- **Dropdowns**:
  - **QSVC Nyström**: `m=32`
  - **Sample positives**: `400` (recommended) or `(no sampling)`
  - **Negative sampling**: `hard`
  - **neg_ratio**: `2.0`

### 4) Learning curve (sample positives × seeds, classical-only)

**Goal**: quickly estimate how PR-AUC moves as you add more positives.

- **Max entities**: `0`
- **Qubits to run**: `[12]` (ignored)
- **Run both ideal + noisy**: **OFF**
- **Dropdowns**:
  - Leave `Sample positives` as `(no sampling)` (the preset already sweeps internally)
  - **Negative sampling**: `hard`
  - **neg_ratio**: `2.0`

### 5) Cheapest sanity (quantum-only, fast, noisy sim)

**Goal**: verify quantum pipeline runs end-to-end cheaply (not a final metric).

- **Max entities**: `80`–`200`
- **Qubits to run**: `[6]`
- **Seeds**: `42`
- **Run both ideal + noisy**: **OFF**
- **Dropdowns**:
  - **QSVC Nyström**: `m=24` or `m=32`
  - **Sample positives**: `200`
  - **Negative sampling**: `random` (ok for sanity)

### 6) Ablation: Nyström vs full kernel (small m vs none)

**Goal**: compare kernel approximation behavior.

- **Max entities**: `0`
- **Qubits to run**: `[6]` (keeps cost manageable)
- **Seeds**: `42,123`
- **Run both ideal + noisy**: **OFF**
- **Dropdowns**:
  - **Sample positives**: `200` or `400` (recommended)
  - Leave Nyström dropdown as `(preset default)` (the preset toggles internally)

## Hardware (IBM Quantum) quick notes

If/when you run on hardware:

```bash
export IBM_Q_TOKEN="PASTE_TOKEN"
export IBM_Q_INSTANCE="ibm-q/open/main"
export IBM_BACKEND="ibm_brisbane"  # or ibm_fez
```

Then use the dashboard’s **Hardware readiness (backend status)** page to confirm access.

