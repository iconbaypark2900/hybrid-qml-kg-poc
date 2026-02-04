---
title: Hybrid QML-KG Dashboard
app_file: benchmarking/dashboard.py
sdk: streamlit
sdk_version: "1.32.0"
---

# Hybrid Quantum–Classical Knowledge Graph Link Prediction

A proof-of-concept system that combines **classical machine learning** and **quantum computing** to predict drug–disease treatment relationships in the **Hetionet** biomedical knowledge graph.

---

## Overview

This project demonstrates how **quantum machine learning (QML)** can be applied to biomedical link prediction.  
It compares classical machine learning approaches with quantum algorithms to explore potential quantum advantages in **parameter efficiency** and **scalability**.

- **Knowledge Graph**: Hetionet  
  - Think of Hetionet as a giant **map of medical facts**.  
  - **Nodes (dots)** represent medical entities—drugs, diseases, genes, symptoms, etc.  
  - **Edges (lines)** represent known relationships, like *this drug treats that disease* or *this gene is linked to that condition*.
- **Focus**: We zoom in on one relationship type: **Compound treats Disease (CtD)**.  
- **Proof-of-Concept**: About **300 items** are sampled to quickly explore how well classical and quantum approaches can learn hidden links.

---

## How It Works

### 1. Build a Training Set
- **Positive examples**: known drug–disease treatments.  
- **Negative examples**: drug–disease pairs not known to be treatments.

### 2. Convert Nodes to Numbers
Each drug and disease is transformed into a unique numeric **fingerprint (embedding)** so that algorithms can process them.

### 3. Create Pairwise Features
For every drug–disease pair, combine the two fingerprints into a single row of numbers that describes their relationship.

### 4. Train Models to Spot Patterns
- **Classical model**: Logistic Regression (standard machine learning).
- **Quantum models**:
  - **QSVC (Quantum Support Vector Classifier)** — computes quantum similarity between pairs.
  - **VQC (Variational Quantum Classifier)** — a small quantum circuit trained with SPSA optimization.

### 5. Predict New Links
Once trained, the models can identify drug–disease pairs that **might represent new treatments**, helping with **drug repurposing**.

---

## Key Results

| Model | Train PR-AUC | Test PR-AUC | Key Takeaway |
|-------|-------------|------------|--------------|
| **Logistic Regression** | 0.86 | 0.60 | Strong on training data but dropped on new data (overfit). |
| **QSVC** | 0.75 | **0.65** | Held up slightly better on unseen data. |
| **VQC** | 0.55 | 0.49 | Performed about as well as random guessing; needs re-tuning. |

> **Precision** = when we say “treats,” how often we are correct.  
> **Recall** = of all real treatments, how many we actually find.  
> **PR-AUC** = balances precision and recall across thresholds (higher is better and key when positives are rare).

---

## Features

- **Biomedical Link Prediction** – Predicts drug–disease treatment relationships from Hetionet.
- **Quantum ML Implementation** – QSVC and VQC built with Qiskit and trained with SPSA.
- **Classical Baselines** – Logistic Regression and other standard machine learning models.
- **IBM Quantum Integration** – Ready to run on real IBM Quantum hardware (Heron, Brisbane, Torino) or simulators.
- **Interactive Dashboard** – Streamlit app to tell the end-to-end story (data → models → benchmarks) and run benchmarks / view evidence-backed predictions.
- **REST API** – FastAPI service for programmatic predictions.

---

## Installation

```bash
# Clone repository
git clone <repository-url>
cd hybrid-qml-kg

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your IBM Quantum token
```

---

## Quick Start

1. **Load Knowledge Graph Data**
   ```bash
   python kg_layer/kg_loader.py
   ```
2. **Train Classical Baseline**
   ```bash
   python classical_baseline/train_baseline.py
   ```
3. **Train Quantum Model (Simulator)**
   ```bash
   python quantum_layer/qml_trainer.py
   ```
4. **Launch Dashboard**
   ```bash
   streamlit run benchmarking/dashboard.py
   ```
5. **Start API Server**
   ```bash
   uvicorn middleware.api:app --reload
   ```

---

## Project Structure

```
hybrid-qml-kg/
├── kg_layer/               # Knowledge graph processing
│   ├── kg_loader.py        # Load Hetionet data
│   └── kg_embedder.py      # Generate embeddings
├── classical_baseline/     # Classical ML models
│   └── train_baseline.py   # Train classical models
├── quantum_layer/          # Quantum ML implementation
│   ├── qml_model.py        # QSVC and VQC models
│   └── qml_trainer.py      # Quantum training pipeline
├── benchmarking/           # Performance evaluation
│   └── dashboard.py        # Streamlit dashboard
├── middleware/             # FastAPI prediction service
│   └── api.py
└── notebooks/              # Jupyter notebooks for experiments
```

---

## Usage Example

### Make a Prediction via API
```python
import requests

response = requests.post("http://localhost:8000/predict-link", json={
    "drug": "DB00945",      # Aspirin
    "disease": "DOID_9352"  # Type 2 Diabetes
})
print(response.json())
# {"link_probability": 0.73, "model_used": "QSVC"}
```

### Run on IBM Quantum Hardware
```bash
python quantum_layer/train_on_heron.py
```

---

## Requirements

- Python 3.9+
- IBM Quantum account (free tier available)
- 8 GB RAM minimum
- Optional: Docker for containerized deployment

---

## Takeaways

- The **classical model** fit the training set well but overfit, dropping from 0.86 to 0.60 PR-AUC on unseen data.
- The **quantum kernel model (QSVC)** generalized slightly better, achieving 0.65 PR-AUC on the test set.
- The **VQC** configuration needs feature/circuit re-tuning; future work will increase feature dimension, add graph-based priors, and use better early stopping.

This demonstrates how quantum methods can **complement classical techniques**, and how combining them may uncover new treatment relationships hidden in biomedical data.

---

## License
MIT

---

## Citation
```
@software{hybrid_qml_kg,
  title = {Hybrid Quantum–Classical Knowledge Graph Link Prediction},
  year = {2025},
  url = {https://github.com/yourusername/hybrid-qml-kg}
}
```

---

## Acknowledgments
- Hetionet biomedical knowledge graph  
- IBM Quantum and Qiskit community  
- Complexity science concepts for understanding emergent behavior in large-scale biomedical networks
