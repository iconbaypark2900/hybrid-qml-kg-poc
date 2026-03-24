# Hybrid Quantum-Classical Knowledge Graph Link Prediction — Project Explanation

## What This Project Does

This is a **hybrid quantum-classical machine learning system** that predicts biomedical relationships in a knowledge graph. Specifically, it answers:

> *“Does compound X treat disease Y?”*

The system predicts **Compound-treats-Disease (CtD)** links on [Hetionet](https://het.io/), a biomedical knowledge graph that connects genes, diseases, compounds, pathways, and other entities. The goal is to find missing or novel treatment links that could support drug repurposing or discovery.

---

## Why Hybrid Quantum-Classical?

**Classical methods** (graph embeddings + tree-based classifiers) already perform well on this task. **Quantum machine learning** uses quantum kernels and circuits to encode data differently and may capture structure that classical models miss.

Rather than replacing classical models, this project **combines** them:

- **Classical path:** Random forest, extra trees, logistic regression trained on graph embeddings
- **Quantum path:** QSVC (Quantum Support Vector Classifier) with quantum feature maps
- **Ensemble:** A stacking meta-learner combines both to produce the final prediction

The best result (PR-AUC **0.7987**) comes from this hybrid ensemble, outperforming classical-only (0.7838) and quantum-only (0.7216) baselines.

---

## How It Works (Architecture)

```
Hetionet Knowledge Graph
         │
         ▼
┌─────────────────────────────────┐
│  Full-Graph Embeddings          │
│  (RotatE, ComplEx, DistMult)    │
│  128D, 200 epochs via PyKEEN    │
└─────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Pair Feature Construction      │
│  concat + diff + Hadamard       │
│  + optional graph topology      │
└─────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Classical   Quantum
 Path       Path
 (RF, ET,   (PCA → Pauli/ZZ
  LR)        feature map → QSVC)
    │         │
    └────┬────┘
         ▼
┌─────────────────────────────────┐
│  Stacking Ensemble              │
│  Meta-learner combines all      │
│  base model predictions         │
└─────────────────────────────────┘
         │
         ▼
  PR-AUC, Rankings, Predictions
```

1. **Knowledge graph:** Hetionet edges are loaded and split into train/test.
2. **Embeddings:** Entities are embedded using RotatE (or ComplEx, DistMult) on the full graph, not just CtD.
3. **Features:** For each (compound, disease) pair, features are built from embeddings (concatenation, difference, Hadamard) plus optional graph metrics.
4. **Classical models:** Random forest, extra trees, and logistic regression are trained; GridSearchCV can tune hyperparameters.
5. **Quantum model:** Features are reduced (e.g., PCA to 16 qubits), encoded via a quantum feature map (Pauli or ZZ), and fed to QSVC.
6. **Ensemble:** A stacking meta-learner combines classical and quantum predictions.
7. **Evaluation:** Test PR-AUC and rankings are reported.

---

## Key Components

| Layer | Purpose | Key Files |
|-------|---------|-----------|
| **KG layer** | Load Hetionet, train embeddings, build pair features | `kg_layer/kg_loader.py`, `kg_layer/kg_embedder.py`, `kg_layer/enhanced_features.py` |
| **Quantum layer** | QSVC, VQC, quantum kernels, execution backends | `quantum_layer/qml_model.py`, `quantum_layer/qml_trainer.py`, `quantum_layer/quantum_executor.py` |
| **Classical baselines** | Random forest, extra trees, logistic regression | `classical_baseline/train_baseline.py` |
| **Ensemble** | Stacking and weighted-average fusion | `quantum_layer/quantum_classical_ensemble.py` |
| **Pipeline** | End-to-end orchestration and Optuna tuning | `scripts/run_optimized_pipeline.py`, `scripts/optuna_pipeline_search.py` |
| **Dashboard** | Streamlit UI for exploration and demos | `benchmarking/dashboard.py` |
| **API** | FastAPI service for programmatic predictions | `middleware/api.py` |

---

## Quantum Execution Backends

The project supports several execution modes:

| Mode | Config | Use Case |
|------|--------|----------|
| CPU statevector simulator | `quantum_config_ideal.yaml` | Local development, small runs |
| GPU simulator | `quantum_config_gpu.yaml` | Faster simulation with cuStateVec |
| Noisy simulator | `quantum_config_noisy.yaml` | Realistic noise modeling |
| IBM Quantum hardware | `quantum_config.yaml` (Heron) | Real quantum device (requires IBM token) |

Behavior is controlled by `config/quantum_config*.yaml` and `--quantum_config_path`.

---

## Best Configuration

The configuration that achieves PR-AUC **0.7987**:

- **Embeddings:** RotatE, 128D, 200 epochs, full-graph training, hard negative sampling
- **Quantum:** Pauli feature map, 2 repetitions, 16 qubits, pre-PCA 24→16D, QSVC C=0.1
- **Ensemble:** Stacking with classical models tuned via GridSearchCV
- **Flag:** `--optimize_feature_map_reps` for kernel alignment

---

## Project Layout

```
hybrid-qml-kg-poc/
├── kg_layer/              # KG loading, embeddings, feature engineering
├── quantum_layer/         # QSVC, VQC, kernels, ensemble, executor
├── classical_baseline/    # Classical ML baselines
├── scripts/               # Main pipeline, Optuna, launch scripts
├── benchmarking/          # Streamlit dashboard
├── middleware/            # FastAPI prediction service
├── config/                # Quantum backend configs
├── experiments/           # Ablation and analysis scripts
├── docs/                  # Technical paper and guides
├── requirements.txt       # Lightweight (dashboard, local runs)
└── requirements-full.txt  # Full pipeline (PyTorch, PyKEEN)
```

---

## What Was Learned

- **Pauli vs ZZ feature map:** Pauli improves ensemble performance (0.7987 vs 0.7408) by encoding pair features differently.
- **VQC underperforms:** Variational quantum circuits are harder to train; QSVC is the effective quantum component.
- **Stacking learns weights:** Manually setting ensemble weights does not outperform automatic stacking.
- **Hard negatives help:** Hard negative sampling outperforms diverse negative sampling in this setup.
- **Full-graph embeddings matter:** Training embeddings on all Hetionet relations yields better entity representations than training only on CtD.

---

## Tech Stack

- **Python 3.9+**, `.venv` recommended
- **Qiskit** — Quantum circuits, kernels, QSVC, backends
- **PyKEEN** — KG embeddings (RotatE, ComplEx, DistMult)
- **scikit-learn** — Classical models, stacking, evaluation
- **Streamlit** — Dashboard
- **FastAPI** — Prediction API

---

## Further Reading

- `README.md` — Quick start, commands, results table
- `docs/README.md` — Index of documentation under `docs/`
- `docs/TECHNICAL_PAPER.md` — Methods, experiments, discussion
- `docs/OPTIMIZATION_PLAN.md` — Optimization roadmap
- `docs/WHY_QUANTUM_UNDERPERFORMS.md` — Quantum–classical gap analysis
