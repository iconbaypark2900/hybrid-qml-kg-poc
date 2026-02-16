# Usage Examples for Hybrid Quantum-Classical Knowledge Graph Link Prediction

This document provides practical examples of how to use the enhanced hybrid quantum-classical system for knowledge graph link prediction.

## 1. Basic Setup and Data Loading

```python
import pandas as pd
from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges, prepare_link_prediction_dataset
from kg_layer.kg_embedder import HetionetEmbedder

# Load Hetionet data
df = load_hetionet_edges()
task_edges, _, _ = extract_task_edges(df, relation_type="CtD", max_entities=300)
train_df, test_df = prepare_link_prediction_dataset(task_edges)

# Initialize embedder and generate features
embedder = HetionetEmbedder(embedding_dim=32, qml_dim=5)
if not embedder.load_saved_embeddings():
    embedder.train_embeddings(task_edges)
    embedder.reduce_to_qml_dim()

# Prepare features for models
X_train_classical = embedder.prepare_link_features(train_df)
X_test_classical = embedder.prepare_link_features(test_df)
X_train_quantum = embedder.prepare_link_features_qml(train_df)
X_test_quantum = embedder.prepare_link_features_qml(test_df)
y_train = train_df["label"].values
y_test = test_df["label"].values
```

## 2. Training Individual Models

### Classical Model
```python
from classical_baseline.train_baseline import ClassicalLinkPredictor

# Train classical baseline
classical_model = ClassicalLinkPredictor(model_type="LogisticRegression")
classical_model.train(train_df, embedder, test_df)

# Get predictions
classical_predictions = classical_model.predict(X_test_classical)
```

### Quantum Model
```python
from quantum_layer.qml_model import QMLLinkPredictor

# Initialize and train quantum model
quantum_model = QMLLinkPredictor(
    model_type="QSVC",
    num_qubits=5,
    feature_map_type="ZZ",
    feature_map_reps=2
)
quantum_model.fit(X_train_quantum, y_train)

# Get predictions
quantum_predictions = quantum_model.predict(X_test_quantum)
```

## 3. Using the Enhanced Ensemble

### Creating and Training the Ensemble
```python
from quantum_layer.quantum_classical_ensemble import QuantumClassicalEnsemble

# Create ensemble with custom configuration
ensemble = QuantumClassicalEnsemble(
    quantum_model=quantum_model,
    classical_model=classical_model,
    ensemble_method="weighted_average",
    weights={"quantum": 0.5, "classical": 0.5}
)

# Fit the ensemble
ensemble.fit(
    X_train=X_train_classical,
    y_train=y_train,
    X_quantum=X_train_quantum,
    X_classical=X_train_classical
)

# Make ensemble predictions
ensemble_predictions = ensemble.predict(
    X=X_test_classical,
    X_quantum=X_test_quantum,
    X_classical=X_test_classical
)

# Get probability scores
ensemble_probabilities = ensemble.predict_proba(
    X=X_test_classical,
    X_quantum=X_test_quantum,
    X_classical=X_test_classical
)
```

### Using Different Ensemble Methods
```python
# Voting ensemble
voting_ensemble = QuantumClassicalEnsemble(
    quantum_model=quantum_model,
    classical_model=classical_model,
    ensemble_method="voting"
)

# Stacking ensemble
stacking_ensemble = QuantumClassicalEnsemble(
    quantum_model=quantum_model,
    classical_model=classical_model,
    ensemble_method="stacking",
    use_stacking=True
)
```

## 4. Evaluating Ensemble Performance

### Basic Evaluation
```python
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

# Evaluate individual models
classical_ap = average_precision_score(y_test, classical_predictions)
quantum_ap = average_precision_score(y_test, quantum_predictions)
ensemble_ap = average_precision_score(y_test, ensemble_probabilities)

print(f"Classical AP: {classical_ap:.4f}")
print(f"Quantum AP: {quantum_ap:.4f}")
print(f"Ensemble AP: {ensemble_ap:.4f}")
```

### Ensemble Diversity Analysis
```python
# Analyze how diverse and effective the ensemble is
diversity_metrics = ensemble.evaluate_ensemble_diversity(
    X_test_classical, y_test,
    X_quantum=X_test_quantum,
    X_classical=X_test_classical
)

print(f"Model correlation: {diversity_metrics['correlation']:.3f}")
print(f"Disagreement rate: {diversity_metrics['disagreement_rate']:.3f}")
print(f"Ensemble improvement: {diversity_metrics['ensemble_improvement']:.3f}")
```

## 5. Optimized Pipeline Usage

### Using the Factory Function
```python
from quantum_layer.quantum_classical_ensemble import create_optimized_quantum_classical_ensemble

# Create an optimized ensemble with default settings
optimized_ensemble = create_optimized_quantum_classical_ensemble(
    ensemble_method="weighted_average",
    random_state=42
)

# The factory function creates properly configured quantum and classical models
# You can then fit and use it as shown above
```

## 6. Advanced Configuration

### Custom Quantum Configuration
```python
from quantum_layer.qml_model import QMLLinkPredictor

# Create quantum model with custom configuration
quantum_config = {
    "model_type": "QSVC",
    "num_qubits": 8,
    "feature_map_type": "ZZ",
    "feature_map_reps": 3,
    "entanglement": "full",
    "random_state": 42
}

quantum_model = QMLLinkPredictor(**quantum_config)
```

### Custom Classical Configuration
```python
from classical_baseline.train_baseline import ClassicalLinkPredictor

# Create classical model with custom configuration
classical_model = ClassicalLinkPredictor(
    model_type="RandomForest",
    random_state=42
)
```

## 7. Working with Different Relations

```python
# Example with different relation types
relations = ["CtD", "DaG", "GiG"]  # Compound-treats-Disease, Disease-associated-with-Gene, Gene-interacts-with-Gene

for relation in relations:
    # Extract task-specific edges
    task_edges, _, _ = extract_task_edges(df, relation_type=relation, max_entities=300)
    train_df, test_df = prepare_link_prediction_dataset(task_edges)
    
    # Train and evaluate for this relation
    # (rest of the pipeline remains the same)
```

## 8. Performance Optimization Tips

### For Large Datasets
```python
# Enable Nyström approximation for large datasets
# This is automatically enabled in the enhanced system for datasets > 500 samples
# But you can configure it explicitly:

from quantum_layer.qml_trainer import qsvc_with_precomputed_kernel

# The enhanced system automatically enables Nyström for large datasets
# You can also manually configure it via args.nystrom_m parameter
```

### For Faster Execution
```python
# Use fast mode for quicker iterations during development
import argparse

# Create args object with fast settings
class Args:
    def __init__(self):
        self.qml_dim = 5
        self.feature_map = "ZZ"
        self.feature_map_reps = 1
        self.entanglement = "linear"
        self.quantum_config = "config/quantum_config.yaml"
        self.nystrom_m = 100  # Smaller landmark set for faster computation

args = Args()
```

## 9. Error Handling and Debugging

### Robust Model Training
```python
try:
    # Attempt to train ensemble
    ensemble.fit(X_train_classical, y_train, X_train_quantum, X_train_classical)
    print("Ensemble training successful!")
except Exception as e:
    print(f"Ensemble training failed: {e}")
    # Fallback to individual models
    print("Falling back to individual models...")
```

### Checking Model Availability
```python
# Check if quantum hardware is available
from quantum_layer.quantum_executor import QuantumExecutor

try:
    executor = QuantumExecutor("config/quantum_config.yaml")
    print(f"Execution mode: {executor.execution_mode}")
    if executor.execution_mode == "heron":
        print("Quantum hardware available!")
    else:
        print("Using simulator")
except Exception as e:
    print(f"Quantum execution issue: {e}")
```

## 10. Saving and Loading Models

```python
import joblib
import pickle

# Save ensemble (you'll need to save components individually)
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump({
        'quantum_model': quantum_model,
        'classical_model': classical_model,
        'ensemble_weights': ensemble.weights,
        'ensemble_method': ensemble.ensemble_method
    }, f)

# Load ensemble
with open('ensemble_model.pkl', 'rb') as f:
    data = pickle.load(f)
    loaded_quantum = data['quantum_model']
    loaded_classical = data['classical_model']
    
    loaded_ensemble = QuantumClassicalEnsemble(
        quantum_model=loaded_quantum,
        classical_model=loaded_classical,
        ensemble_method=data['ensemble_method'],
        weights=data['ensemble_weights']
    )
```

These examples demonstrate the key capabilities of the enhanced hybrid quantum-classical system, showing how to leverage the improvements made to the quantum kernel computation, error handling, embedding pipeline, and ensemble integration.