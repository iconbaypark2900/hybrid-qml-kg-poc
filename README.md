---
title: Hybrid QML-KG Dashboard
app_file: benchmarking/dashboard.py
sdk: streamlit
sdk_version: "1.32.0"
---

# Hybrid Quantum-Classical Knowledge Graph Link Prediction

A hybrid quantum-classical machine learning system for biomedical link prediction on the [Hetionet](https://het.io/) knowledge graph. The system predicts **Compound-treats-Disease (CtD)** relationships by combining classical ensemble methods with quantum kernel classifiers, achieving a best PR-AUC of **0.7987** via a stacking ensemble.

---

## Results

| Model | Test PR-AUC | Type |
|-------|-------------|------|
| Ensemble-QC-stacking (Pauli) | **0.7987** | Hybrid ensemble |
| RandomForest-Optimized | 0.7838 | Classical |
| ExtraTrees-Optimized | 0.7807 | Classical |
| Ensemble-QC-stacking (ZZ) | 0.7408 | Hybrid ensemble |
| QSVC-Optimized | 0.7216 | Quantum |

**Target PR-AUC > 0.70: Achieved.**

The best result uses full-graph RotatE embeddings (128D, 200 epochs), hard negative sampling, 16-qubit Pauli feature maps (reps=2), QSVC regularization (C=0.1), and a stacking ensemble with GridSearchCV-tuned classical models.

---

## Architecture

```
Hetionet (CtD)
    |
    v
Full-Graph Embeddings (RotatE / ComplEx, 128D, 200 epochs)
    |
    v
Pair Feature Construction (concat + diff + Hadamard)
    |                               |
    v                               v
Classical Path                 Quantum Path
 - LogisticRegression           - PCA reduction (pre-PCA 24 -> 16D)
 - RandomForest                 - Pauli / ZZ feature maps
 - ExtraTrees                   - QSVC (C=0.1, kernel alignment)
 - GridSearchCV tuning          - VQC (SPSA optimizer)
    |                               |
    +-------------------------------+
                    |
                    v
          Stacking Ensemble
                    |
                    v
          PR-AUC & Rankings
```

### Pipeline Components

- **Knowledge graph layer** (`kg_layer/`): Hetionet ingestion, full-graph embedding training (RotatE, ComplEx, DistMult via PyKEEN), enhanced feature engineering (graph topology features, hard negative sampling), and embedding diversity analysis.
- **Quantum layer** (`quantum_layer/`): QSVC with fidelity quantum kernels (ZZ, Pauli feature maps), VQC with configurable ansatzes (RealAmplitudes, EfficientSU2, TwoLocal) and SPSA optimizer, kernel-target alignment, and quantum-classical ensemble (stacking or weighted average).
- **Classical baselines** (`classical_baseline/`): Logistic regression, random forest, and extra trees with optional GridSearchCV hyperparameter tuning.
- **Execution backends** (`quantum_layer/quantum_executor.py`): Statevector simulator (default), noisy simulator (Aer with device noise models), GPU-accelerated simulator (cuStateVec via `--gpu`), and IBM Quantum hardware (Heron).
- **Dashboard** (`benchmarking/dashboard.py`): Streamlit application with a narrative-driven, six-page layout: **The Problem**, **Our Approach**, **Results**, **What We Learned**, **Try It**, and **Technical Reference**. Includes Mermaid diagrams (KG subgraph, pipeline architecture, improvement journey), styled HTML cards with accent borders, dark sidebar navigation, best-run defaults, and Generate-demo / Run pipeline / Upload flows. Deployed to [Hugging Face Spaces](https://huggingface.co/spaces/rocRevyAreGoals15/QGG-HYBRID-PROJECT).
- **API** (`middleware/api.py`): FastAPI service for programmatic link predictions.

---

## Quick Start

### Installation

```bash
git clone <repository-url>
cd hybrid-qml-kg-poc

python -m venv .venv
source .venv/bin/activate

# Dashboard and lightweight runs
pip install -r requirements.txt

# Full pipeline (PyTorch, PyKEEN, embedding training)
pip install -r requirements-full.txt
```

### Reproduce the Best Result

```bash
python scripts/run_optimized_pipeline.py --relation CtD \
  --full_graph_embeddings --embedding_method RotatE --embedding_dim 128 \
  --embedding_epochs 200 --negative_sampling hard --qml_dim 16 \
  --qml_feature_map Pauli --qml_feature_map_reps 2 --qsvc_C 0.1 \
  --optimize_feature_map_reps --run_ensemble --ensemble_method stacking \
  --tune_classical --qml_pre_pca_dim 24 --fast_mode
```

For robust evaluation (K-fold CV, no fast_mode), add `--use_cv_evaluation --cv_folds 5`. For paper-ready runs, add `--run_multimodel_fusion --fusion_method bayesian_averaging` and omit `--fast_mode`. See [docs/reference/TEST_COMMANDS.md](docs/reference/TEST_COMMANDS.md).

### Launch the Dashboard

```bash
# Default port (8501)
streamlit run benchmarking/dashboard.py

# Or use the launch script to pick the first available port (8501, 8502, ...)
./scripts/launch_dashboard.sh
```

The dashboard uses a **six-page narrative flow**: The Problem, Our Approach, Results, What We Learned, Try It, and Technical Reference. It includes **Generate demo results** (under Try It) to load best-run metrics without running the pipeline, Mermaid diagrams for the knowledge graph and pipeline, and a dark sidebar with pill-style navigation. A live instance is hosted on [Hugging Face Spaces](https://huggingface.co/spaces/rocRevyAreGoals15/QGG-HYBRID-PROJECT).

### Start the API Server

```bash
uvicorn middleware.api:app --reload
```

---

## Pipeline Configuration

### Key Flags

| Flag | Description | Best-run value |
|------|-------------|----------------|
| `--full_graph_embeddings` | Train embeddings on all Hetionet relations | Enabled |
| `--embedding_method` | KG embedding algorithm | `RotatE` |
| `--embedding_dim` | Embedding dimensionality | `128` |
| `--embedding_epochs` | Training epochs for embeddings | `200` |
| `--negative_sampling` | Negative sampling strategy | `hard` |
| `--qml_dim` | Number of qubits / quantum feature dimension | `16` |
| `--qml_feature_map` | Quantum feature map type | `Pauli` |
| `--qml_feature_map_reps` | Feature map repetitions | `2` |
| `--qsvc_C` | QSVC regularization parameter | `0.1` |
| `--run_ensemble` | Enable quantum-classical ensemble | Enabled |
| `--ensemble_method` | Ensemble strategy | `stacking` |
| `--tune_classical` | GridSearchCV for classical models | Enabled |
| `--qml_pre_pca_dim` | Pre-PCA dimensionality | `24` |
| `--optimize_feature_map_reps` | Auto-select reps via kernel alignment | Enabled |
| `--use_graph_features_in_qml` | Add graph topology features to quantum input | Optional |
| `--gpu` | Use GPU-accelerated quantum simulation | Optional |

### Execution Modes

```bash
# CPU simulator (default)
python scripts/run_optimized_pipeline.py --relation CtD ...

# GPU-accelerated simulation (requires qiskit-aer-gpu)
python scripts/run_optimized_pipeline.py --relation CtD --gpu ...

# DGX Spark (auto-detected)
HYBRID_QML_SYSTEM=dgx python scripts/run_optimized_pipeline.py --relation CtD ...

# Explicit config file
python scripts/run_optimized_pipeline.py --quantum_config_path config/quantum_config_gpu.yaml ...
```

All GPU paths fall back to CPU automatically if no GPU is detected.

### Hyperparameter Search (Optuna)

```bash
python scripts/optuna_pipeline_search.py --n_trials 30 --objective ensemble
python scripts/optuna_pipeline_search.py --n_trials 20 --objective qsvc
python scripts/optuna_pipeline_search.py --n_trials 20 --objective classical
```

Results are saved to `results/optuna/optuna_trials.csv` and `results/optuna/optuna_best.json`.

---

## Experiment Log

| Variant | RF | ET | QSVC | Ensemble | Notes |
|---------|------|------|------|----------|-------|
| Base (200 ep, stacking, tune, pre-PCA 24) | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Best classical |
| + Pauli feature map (reps=2) | 0.7838 | 0.7807 | 0.6343 | **0.7987** | Best ensemble |
| + diverse negatives (dw=0.5) | 0.7144 | 0.7298 | 0.6689 | 0.6919 | Diverse hurts here |
| + qsvc_C=0.05 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Same as C=0.1 |
| + ensemble_quantum_weight=0.4 | 0.7838 | 0.7807 | 0.7216 | 0.7408 | Stacking learns weights |

**Key findings:**

- The Pauli feature map substantially improves ensemble performance (0.7408 to 0.7987) by changing how quantum kernels encode pair features.
- Stacking ensemble learns optimal classical/quantum weights automatically; manually setting `ensemble_quantum_weight` has no additional effect.
- Hard negative sampling outperforms diverse negative sampling in this configuration.
- VQC remains near random (best: 0.5474 with RealAmplitudes reps=4, SPSA); QSVC is the effective quantum model.

---

## Project Structure

```
hybrid-qml-kg-poc/
|-- kg_layer/                    # Knowledge graph processing
|   |-- kg_loader.py             # Hetionet data ingestion
|   |-- kg_embedder.py           # Embedding training (PyKEEN)
|   |-- advanced_embeddings.py   # RotatE, ComplEx, DistMult
|   |-- enhanced_features.py     # Graph topology features
|   +-- ...
|-- quantum_layer/               # Quantum ML implementation
|   |-- qml_model.py             # QSVC and VQC models
|   |-- qml_trainer.py           # Quantum training pipeline
|   |-- quantum_executor.py      # Backend routing (sim/GPU/hardware)
|   |-- advanced_qml_features.py # Quantum feature preparation
|   |-- quantum_kernel_engineering.py
|   |-- quantum_classical_ensemble.py
|   +-- ...
|-- classical_baseline/          # Classical ML baselines
|   +-- train_baseline.py
|-- scripts/                     # Pipeline and experiment scripts
|   |-- run_optimized_pipeline.py    # Main pipeline entry point
|   |-- launch_dashboard.sh          # Find free port and run Streamlit
|   |-- optuna_pipeline_search.py    # Bayesian HPO
|   |-- implementations/             # Standalone drivers (ensembles, fusion, tuning)
|   |-- shell/                       # Bash wrappers (quantum fixes, phased runs)
|   +-- demos/                       # Small demos for the quantum test suite
|-- tests/                       # Quantum improvements tests (see run_tests.py)
|   +-- test_quantum_improvements_*.py
|-- deployment/                  # Dockerfiles, compose (not docs; see docs/deployment/)
|-- benchmarking/                # Performance evaluation
|   |-- dashboard.py             # Streamlit dashboard
|   +-- ...
|-- config/                      # Quantum backend configurations
|   |-- quantum_config_gpu.yaml  # GPU simulator
|   |-- quantum_config_dgx.yaml  # DGX Spark
|   |-- quantum_config_ideal.yaml
|   |-- quantum_config_noisy.yaml
|   +-- ...
|-- experiments/                 # Ablation and analysis scripts
|   |-- vqc_optimization_analysis.py
|   |-- quantum_ablation.py
|   +-- ...
|-- middleware/                   # FastAPI prediction service
|   +-- api.py
|-- utils/                       # Shared utilities
|   |-- evaluation.py            # CV, metrics, model comparison
|   |-- calibration.py           # Model calibration
|   +-- reproducibility.py       # Seed control
|-- notebooks/                   # Jupyter notebooks
|-- docs/                        # Documentation (see docs/README.md)
|   |-- planning/               # Task lists, experiment logs
|   |-- reports/                # Results analyses and writeups
|   |-- reference/              # Commands, testing, usage
|   |-- guides/                 # Cookbooks and directory guide
|   |-- overview/               # Project explanation, implementation recap
|   |-- improvements/           # Improvement iteration logs
|   |-- deployment/             # Hugging Face, Docker install notes
|   |-- kg_layer/               # KG embedding notes
|   +-- quantum/                # Quantum layer notes
|-- requirements.txt             # Dashboard dependencies
|-- requirements-full.txt        # Full pipeline dependencies
|-- run_tests.py                 # Quantum test suite (terminal / Streamlit)
|-- app.py                       # Optional HF Spaces entry (runs benchmarking/dashboard.py)
+-- README.md
```

---

## Requirements

- Python 3.9+
- 8 GB RAM minimum (16 GB recommended for 16-qubit runs)
- Optional: NVIDIA GPU with CUDA for accelerated quantum simulation
- Optional: IBM Quantum account for hardware execution

### Dependency Groups

| File | Scope |
|------|-------|
| `requirements.txt` | Dashboard and lightweight local runs (Streamlit, Qiskit, scikit-learn) |
| `requirements-full.txt` | Full pipeline including PyTorch, PyKEEN, and embedding training |

---

## Documentation

| Document | Description |
|----------|-------------|
| `docs/README.md` | Index of all documentation under `docs/` |
| `docs/TECHNICAL_PAPER.md` | Technical paper: hybrid QML-KG link prediction, methods, results, and discussion |
| `docs/planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md` | Full experiment log, recommended commands, and optimization roadmap |
| `docs/overview/IMPLEMENTATION_RECAP.md` | Summary of pipeline improvements and GPU/hardware readiness |
| `docs/deployment/DEPLOY_HUGGINGFACE.md` | Deploying the dashboard to Hugging Face Spaces and pushing updates |
| `docs/WHY_QUANTUM_UNDERPERFORMS.md` | Root cause analysis of quantum-classical performance gap |
| `docs/OPTIMIZATION_PLAN.md` | Detailed optimization roadmap |

---

## Recent work

- **Dashboard rewrite**: Replaced the former 10-page info-dump with a six-page narrative (The Problem, Our Approach, Results, What We Learned, Try It, Technical Reference). Added Mermaid diagrams for the Hetionet subgraph, pipeline architecture, and improvement journey; custom HTML cards with colored accent borders and stat highlights; dark gradient sidebar with pill-style navigation; and section labels for clearer hierarchy.
- **Best-run defaults**: Dashboard and README default to the configuration that achieves PR-AUC 0.7987 (Pauli feature map, stacking ensemble, RotatE 128D, hard negatives, `--tune_classical`, `--qml_pre_pca_dim 24`). Generate demo results and Reproduce-command fallback use this configuration.
- **Launch script**: `scripts/launch_dashboard.sh` finds the first available port (8501, 8502, …) and runs the Streamlit app, so multiple instances or a busy port do not block startup.
- **Hugging Face Space**: The app is deployed at [QGG-HYBRID-PROJECT](https://huggingface.co/spaces/rocRevyAreGoals15/QGG-HYBRID-PROJECT). Push to the Space with `git push hf <your-branch>:main` (see `docs/deployment/DEPLOY_HUGGINGFACE.md`).
- **Documentation**: README updated with current results, pipeline features, project structure, and dashboard description. `docs/planning/NEXT_STEPS_TO_IMPROVE_PERFORMANCE.md` and `docs/overview/IMPLEMENTATION_RECAP.md` document the experiment log, recommended commands, and GPU/configuration options.

---

## License

MIT

---

## Citation

```bibtex
@software{hybrid_qml_kg,
  title  = {Hybrid Quantum-Classical Knowledge Graph Link Prediction},
  year   = {2026},
  url    = {https://github.com/yourusername/hybrid-qml-kg-poc}
}
```

---

## Acknowledgments

- [Hetionet](https://het.io/) biomedical knowledge graph
- [IBM Quantum](https://quantum.ibm.com/) and the Qiskit community
- [PyKEEN](https://pykeen.github.io/) for knowledge graph embedding training
