# Hybrid QML-KG Project Commands

Quick reference guide for running the Hybrid Quantum-Classical Knowledge Graph Link Prediction system.

---

## 🚀 Quick Start

### Setup Environment
```bash
# Navigate to project
cd /home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc

# Activate virtual environment
source .venv/bin/activate

# Set Python path (needed for all commands)
export PYTHONPATH=/home/roc/quantumGlobalGroup/semantics/hybrid-qml-kg-poc:$PYTHONPATH
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Running Algorithms

### 1. Run Complete Pipeline (VQC Default)
Runs: Data loading → Embeddings → Classical Baseline → Quantum VQC → Benchmarking
```bash
python scripts/run_pipeline.py
```

### 2. Run All Three Algorithms (Benchmark)
Runs: QSVC → Classical RBF-SVC → VQC
```bash
bash scripts/benchmark_all.sh
```
Or make it executable first:
```bash
chmod +x scripts/benchmark_all.sh
./scripts/benchmark_all.sh
```

### 3. Individual Algorithm Commands

#### QSVC (Quantum Support Vector Classifier)
```bash
python -m quantum_layer.qml_trainer \
  --model_type QSVC \
  --qml_dim 5 \
  --feature_map ZZ \
  --feature_map_reps 2 \
  --results_dir results
```

#### VQC (Variational Quantum Classifier)
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

#### Classical RBF-SVC Baseline
```bash
python scripts/rbf_svc_fixed.py
```

---

## ⚙️ Configuration

### Switch Execution Mode (Simulator vs Real Quantum Hardware)

**Use Simulator (Free, Local)**
```bash
# Edit config/quantum_config.yaml
# Set: execution_mode: simulator
```

**Use IBM Quantum Hardware (Brisbane/Torino)**
```bash
# Edit config/quantum_config.yaml
# Set: execution_mode: heron
# Set: backend: ibm_torino  (or ibm_brisbane)
```

### Save IBM Quantum Token
```bash
python save_token.py
```
Enter your token from: https://quantum.ibm.com/account

---

## 🧪 Testing Quantum Connection

### Quick Test (Simulator)
```bash
python scripts/test/quantum.py
# Select option 3 for local simulator
```

### Test Real Quantum Hardware
```bash
python scripts/test/quantum.py
# Select option 1 (Brisbane) or 2 (Torino)
```

### Simple Connection Test
```bash
python scripts/test/test_connection.py
```

---

## 📈 View Results

### Latest Results
```bash
# View latest quantum metrics
cat results/quantum_metrics_latest.json

# View latest predictions
cat results/predictions_latest.csv

# View experiment history
cat results/experiment_history.csv
```

### All Results
```bash
# List all result files
ls -lt results/

# View specific QSVC results
cat results/quantum_metrics_QSVC_*.json | tail -30

# View specific VQC results
cat results/quantum_metrics_VQC_*.json | tail -30

# View classical RBF-SVC results
cat results/rbf_svc_128d_fixed_*.json | tail -30
```

---

## 🎛️ Advanced Options

### QML Trainer Options
```bash
python -m quantum_layer.qml_trainer --help
```

Key parameters:
- `--model_type`: QSVC or VQC
- `--qml_dim`: Number of qubits (default: 5)
- `--max_entities`: Dataset size (default: 300)
- `--embedding_dim`: Embedding dimension (default: 32)
- `--feature_map`: ZZ or Z
- `--feature_map_reps`: Feature map repetitions (default: 2)
- `--ansatz`: RealAmplitudes or EfficientSU2 (VQC only)
- `--ansatz_reps`: Ansatz repetitions (default: 3)
- `--optimizer`: COBYLA or SPSA
- `--max_iter`: Max optimization iterations (default: 50)
- `--train_limit`: Subsample training set for faster testing

### Example: Quick Test Run (Subsampled Data)
```bash
python -m quantum_layer.qml_trainer \
  --model_type VQC \
  --qml_dim 5 \
  --max_iter 20 \
  --train_limit 100 \
  --results_dir results
```

---

## 📊 Dashboard & API

### Launch Streamlit Dashboard
```bash
streamlit run benchmarking/dashboard.py
```
Opens at: http://localhost:8501

### Launch FastAPI Server
```bash
uvicorn middleware.api:app --reload
```
API docs at: http://localhost:8000/docs

### Example API Request
```bash
curl -X POST http://localhost:8000/predict-link \
  -H "Content-Type: application/json" \
  -d '{"drug": "DB00945", "disease": "DOID_9352"}'
```

---

## 🐳 Docker Deployment

### Build and Run with Docker Compose
```bash
cd deployment
docker compose up --build
```

Services:
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Notebooks: http://localhost:8888

### Stop Services
```bash
docker compose down
```

---

## 🔧 Troubleshooting

### Fix Numpy/Pandas Compatibility
```bash
pip uninstall -y pandas numpy
pip install --no-cache-dir --force-reinstall numpy==1.26.4 pandas==2.1.4
```

### Fix Qiskit Version Conflicts
```bash
# Remove all qiskit packages
pip uninstall -y qiskit qiskit-terra qiskit-aer qiskit-machine-learning qiskit-ibm-runtime

# Reinstall compatible versions
pip install qiskit qiskit-aer qiskit-machine-learning qiskit-ibm-runtime
```

### Reset Virtual Environment
```bash
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📝 Jupyter Notebooks

### Launch Jupyter
```bash
jupyter notebook
```

Available notebooks:
- `notebooks/01-kg-ingestion.ipynb` - Data loading and exploration
- `notebooks/02-classical-baseline.ipynb` - Classical models
- `notebooks/03-qml-training.ipynb` - Quantum ML training

---

## 🧹 Cleanup

### Clean Results
```bash
rm -rf results/*.json results/*.csv
```

### Clean Cache
```bash
rm -rf __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### Clean Models
```bash
rm -rf models/*.joblib
```

---

## 📚 Key Files

- `scripts/run_pipeline.py` - Complete pipeline (VQC)
- `scripts/benchmark_all.sh` - Run all three algorithms
- `scripts/rbf_svc_fixed.py` - Classical RBF-SVC baseline
- `quantum_layer/qml_trainer.py` - QML training (CLI)
- `config/quantum_config.yaml` - Quantum execution config
- `middleware/api.py` - FastAPI server
- `benchmarking/dashboard.py` - Streamlit dashboard

---

## 💡 Tips

1. **Start with simulator mode** to avoid using quantum time
2. **Use --train_limit** for quick testing with smaller datasets
3. **Run QSVC first** (fastest) before VQC (slowest)
4. **Check results/** directory after each run
5. **Monitor queue times** before running on real quantum hardware

---

## 🆘 Getting Help

```bash
# General help
python scripts/run_pipeline.py --help
python -m quantum_layer.qml_trainer --help

# Check configuration
cat config/quantum_config.yaml

# View logs
tail -f results/experiment_history.csv
```

---

**Last Updated**: 2025-10-17
**Project**: Hybrid Quantum-Classical Knowledge Graph Link Prediction

