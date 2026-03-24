# Directory Guide

An extensive, practical map of the repository so you can locate code quickly, understand data flow end-to-end, and know where to extend functionality.

- Project root: `/home/roc/quantumGlobalGroup/hybrid-qml-kg-poc` (adjust to your clone path)

## Table of Contents
- [High-Level Overview](#high-level-overview)
- [Repository Tree](#repository-tree)
- [Data Flow](#data-flow)
- [Directories](#directories)
  - [benchmarking](#benchmarking)
  - [classical_baseline](#classical_baseline)
  - [config](#config)
  - [data](#data)
  - [deployment](#deployment)
  - [docs](#docs)
  - [kg_layer](#kg_layer)
  - [middleware](#middleware)
  - [models](#models)
  - [notebooks](#notebooks)
  - [quantum_layer](#quantum_layer)
  - [results](#results)
  - [scripts](#scripts)
- [Key Root Files](#key-root-files)
- [Common Workflows](#common-workflows)
- [Extending the Project](#extending-the-project)
- [Conventions](#conventions)

## High-Level Overview
This repo implements a hybrid quantum–classical pipeline for biomedical link prediction on Hetionet (focus: Compound-treats-Disease).

- Ingestion & embeddings: `kg_layer`
- Classical models: `classical_baseline`
- Quantum models + execution: `quantum_layer`
- Orchestration & runners: `scripts`, `middleware`
- Visualization & evaluation: `benchmarking`
- Configuration & deployment: `config`, `deployment`
- Results & artifacts: `results`, `models`, `data`

## Repository Tree
```
hybrid-qml-kg-poc/
├── benchmarking/            # Streamlit dashboard and evaluation assets
├── classical_baseline/      # Classical ML training and evaluation
├── config/                  # Runtime configs (e.g., quantum backend)
├── data/                    # Input datasets and preprocessed artifacts (local)
├── deployment/              # Docker and compose services
├── docs/                    # Documentation (see docs/README.md)
├── kg_layer/                # KG loading, processing, embeddings
├── middleware/              # FastAPI service and orchestration helpers
├── models/                  # Serialized model artifacts (e.g., .joblib)
├── notebooks/               # Jupyter notebooks for exploration/experiments
├── quantum_layer/           # QML models, encoders, executors, trainers
├── results/                 # Metrics, predictions, experiment history
├── scripts/                 # Pipeline runners, implementations/, shell/, demos/
├── tests/                   # Quantum improvements test suite
├── README.md                # Project overview & quickstart
├── run_tests.py             # Test runner (terminal / Streamlit)
├── app.py                   # Optional HF Spaces entry (dashboard)
├── requirements.txt         # Python dependencies
└── requirements-full.txt    # Full pipeline dependencies
```

## Data Flow
```
[Hetionet/raw data]
        |
        v
kg_layer/kg_loader.py  --> structured KG dataframes
        |
        v
kg_layer/kg_embedder.py --> node embeddings & pairwise features
        |                         |
        |                         ├─> classical_baseline/train_baseline.py
        |                         └─> quantum_layer/qml_trainer.py (QSVC/VQC)
        v
   models/ (optionally)       results/ (metrics, predictions, history)
        |
        v
benchmarking/dashboard.py (visualize)   middleware/api.py (serve predictions)
```

## Directories

### benchmarking
- Purpose: Interactive evaluation and visualization (Streamlit).
- Key files:
  - `benchmarking/dashboard.py`: Reads from `results/` and visualizes metrics, comparisons, and history.
- Inputs: `results/*.json`, `results/*.csv`
- Outputs: UI at `http://localhost:8501`
- Typical usage:
  - `streamlit run benchmarking/dashboard.py`

### classical_baseline
- Purpose: Classical ML baselines (e.g., Logistic Regression, SVM) for comparison.
- Key files:
  - `train_baseline.py`: Trains classical models using features from `kg_layer`.
  - `evaluate_baseline.py`: Standalone evaluation if needed.
- Inputs: Features/embeddings from `kg_layer`
- Outputs: Metrics in `results/`, optional model dumps in `models/`

### config
- Purpose: Toggle and parameterize runtime behavior (especially quantum backends).
- Key files:
  - `quantum_config.yaml`: Controls `execution_mode` (`simulator`, `heron`), backend (e.g., `ibm_torino`), and related runtime options.
- Notes: Update this before running quantum jobs (simulator vs real hardware).

### data
- Purpose: Local data storage for source datasets and preprocessed artifacts.
- Typical contents:
  - Raw Hetionet extracts, cached dataframes, sampled subsets for PoC runs.
- Notes: Not versioned for large files; consider `.gitignore` for bulky assets.

### deployment
- Purpose: Containerized deployment for API, dashboard, notebooks.
- Typical files:
  - `docker-compose.yml`: Brings up API (`middleware`), dashboard (`benchmarking`), and Jupyter.
- Usage:
  - `cd deployment && docker compose up --build`
  - Services: API `http://localhost:8000`, Dashboard `http://localhost:8501`, Notebooks `http://localhost:8888`

### docs
- Purpose: Extended documentation, design notes, references.
- Contents: Architecture notes, research references, design decisions.

### kg_layer
- Purpose: Knowledge graph loading, utilities, and embedding generation.
- Key files:
  - `kg_loader.py`: Loads Hetionet into in-memory structures/dataframes.
  - `kg_embedder.py`: Creates node embeddings; builds pairwise features for model training.
  - `kg_utils.py`: Reusable helpers (e.g., sampling, ID mapping, graph utilities).
- Inputs: Raw KG data (`data/`)
- Outputs: Embeddings/features (in-memory to callers, optionally persisted)

### middleware
- Purpose: Programmatic access via FastAPI; orchestration helpers.
- Key files:
  - `api.py`: FastAPI app exposing endpoints (e.g., `/predict-link`).
  - `orchestrator.py`: Shared orchestration logic for loading models/configs/features.
- Inputs: Trained artifacts (`models/`) and configuration (`config/`)
- Outputs: JSON responses for predictions and health
- Usage:
  - `uvicorn middleware.api:app --reload` (OpenAPI at `http://localhost:8000/docs`)

### models
- Purpose: Storage for serialized trained models (e.g., `.joblib`, `.pkl`).
- Lifecycle:
  - Written by training scripts, read by API/middleware for serving.
- Notes: Consider versioned naming (timestamp/model type) and symlinks for "latest".

### notebooks
- Purpose: Exploratory analysis and experiments.
- Examples:
  - `01-kg-ingestion.ipynb` – Load/explore Hetionet
  - `02-classical-baseline.ipynb` – Classical models
  - `03-qml-training.ipynb` – QML training
- Usage:
  - `jupyter notebook` then open files under `notebooks/`

### quantum_layer
- Purpose: Quantum models, encoders, execution wrapper, and training CLI.
- Key files:
  - `qml_model.py`: QSVC/VQC model definitions and wiring.
  - `qml_trainer.py`: CLI to train/evaluate QML models (QSVC/VQC) with options like `--qml_dim`, `--feature_map`, `--ansatz`, `--optimizer`.
  - `qml_encoder.py`: Encodes classical features into quantum states (feature maps).
  - `quantum_executor.py`: Handles backend selection (simulator vs IBM Quantum), job submission, and results retrieval.
  - `train_on_heron.py`: Example script to run on specific IBM hardware.
- Inputs: Features from `kg_layer`, config from `config/quantum_config.yaml`
- Outputs: Metrics (`results/`), optionally model artifacts (`models/`)

### results
- Purpose: Centralized outputs for experiments and runs.
- Typical contents:
  - `quantum_metrics_*.json`: Metrics per QML run (QSVC/VQC).
  - `predictions_*.csv`: Predictions for evaluated pairs.
  - `experiment_history.csv`: Append-only log of runs with parameters and performance.
  - Classical outputs like `rbf_svc_*.json` from baseline runs.

### scripts
- Purpose: Orchestrators and utilities to run the full pipeline or specific benchmarks.
- Key files:
  - `run_pipeline.py`: End-to-end pipeline (Data → Embeddings → Classical → Quantum → Results).
  - `benchmark_all.sh`: Runs QSVC → RBF-SVC → VQC in sequence for comparison.
  - `rbf_svc_fixed.py`: Classical RBF-SVC baseline.
  - `train_on_heron.py`: Convenience launcher targeting IBM hardware.
  - `test/`: Connectivity and sanity tests for quantum stack.
    - `test/quantum.py`: Interactive quantum backend tester (simulator or hardware).
    - `test/working_quantum.py`: Known-good minimal example circuit/workflow.
    - `test/test_quantum.py`: Simple tests to validate runtime and versions.

## Key Root Files
- `README.md`: Conceptual overview, quickstart, project structure summary.
- `docs/guides/COMMANDS.md`: Command cookbook for common tasks, parameters, troubleshooting, and deployment.
- `run_tests.py`: Quantum improvements test runner (terminal and/or Streamlit dashboard).
- `tests/`: Modules for `test_quantum_improvements_*`; JSON output under `results/test_results/` (created at runtime).
- `scripts/implementations/`: Standalone `implement_*.py` experiment drivers.
- `scripts/shell/`: Bash wrappers for pipeline and quantum-fix runs.
- `requirements.txt`: Dependency pinning for compatible environment.

## Common Workflows

### Run complete pipeline (default VQC):
```bash
python scripts/run_pipeline.py
```

### Benchmark QSVC, RBF-SVC, VQC:
```bash
bash scripts/benchmark_all.sh
```

### Quantum trainer (QSVC):
```bash
python -m quantum_layer.qml_trainer \
  --model_type QSVC \
  --qml_dim 5 \
  --feature_map ZZ \
  --feature_map_reps 2 \
  --results_dir results
```

### Quantum trainer (VQC):
```bash
python -m quantum_layer.qml_trainer \
  --model_type VQC \
  --qml_dim 5 \
  --feature_map ZZ \
  --feature_map_reps 2 \
  --ansatz RealAmplitudes \
  --ansatz_reps 3 \
  --optimizer COBYLA \
  --max_iter 50 \
  --results_dir results
```

### Switch to simulator vs hardware:
Edit `config/quantum_config.yaml`:
- `execution_mode: simulator` (local, free)
- `execution_mode: heron` and `backend: ibm_torino` (or `ibm_brisbane`) for real hardware

### Launch dashboard:
```bash
streamlit run benchmarking/dashboard.py
```

### Start API:
```bash
uvicorn middleware.api:app --reload
```

## Extending the Project

### New classical model:
- Add training/eval code under `classical_baseline/` (e.g., `train_new_model.py`).
- Log outputs into `results/` and optionally save artifacts to `models/`.

### New quantum circuit/feature map:
- Implement in `quantum_layer/qml_encoder.py` and/or extend `qml_model.py`.
- Add CLI flags in `quantum_layer/qml_trainer.py` to expose configuration.

### New KG features:
- Add transformations to `kg_layer/kg_embedder.py` and helpers in `kg_layer/kg_utils.py`.

### Serving a new endpoint:
- Add route in `middleware/api.py`, reuse loaders in `orchestrator.py`.

## Conventions

### Paths
Absolute paths commonly used in docs and commands for clarity.

### Results naming
Include model type and timestamp (`quantum_metrics_QSVC_YYYYMMDD_HHMM.json`).

### Environment
Activate venv before running:
```bash
source .venv/bin/activate
export PYTHONPATH=/home/roc/quantumGlobalGroup/hybrid-qml-kg-poc:$PYTHONPATH
```

### Cleanup
- Clear results: `rm -rf results/*.json results/*.csv`
- Clear caches: remove `__pycache__` and `*.pyc`
- Clear models: `rm -rf models/*.joblib`