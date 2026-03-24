# ETL Pipeline Analysis: Feature Flow and Component Mapping

## Complete ETL Pipeline Overview

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ETL PIPELINE FLOW                            │
└─────────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   └─> kg_loader.py
       ├─> download_hetionet_if_missing()  [Downloads Hetionet TSV]
       ├─> load_hetionet_edges()           [Loads into DataFrame]
       └─> extract_task_edges()            [Filters by relation type]
           └─> Returns: task_edges DataFrame with source/metaedge/target

2. DATA PREPARATION
   └─> kg_loader.py
       └─> prepare_link_prediction_dataset()
           ├─> Splits positive edges into train/test
           ├─> get_negative_samples()      [Generates random negatives]
           └─> Returns: train_df, test_df with 'label' column

3. EMBEDDING GENERATION
   └─> kg_embedder.py
       └─> HetionetEmbedder.train_embeddings()
           ├─> _train_with_pykeen()        [TransE embeddings via PyKEEN]
           │   └─> OR fallback to:
           └─> _train_fallback()           [Deterministic random embeddings]
           └─> reduce_to_qml_dim()         [PCA reduction for QML]
               └─> Returns: entity_embeddings [N×embedding_dim]
                   └─> reduced_embeddings [N×qml_dim]

4. FEATURE ENGINEERING
   └─> kg_embedder.py
       ├─> prepare_link_features()         [Classical: 4×dim features]
       │   └─> Features: [h, t, |h-t|, h*t] concatenated
       │   └─> Dimension: 4 × embedding_dim (e.g., 4×32 = 128D)
       │
       └─> prepare_link_features_qml()     [Quantum: qml_dim features]
           ├─> mode="diff": |h - t|        [qml_dim features]
           ├─> mode="hadamard": h ⊙ t      [qml_dim features]
           └─> mode="both": PCA([|h-t|, h⊙t]) → qml_dim

5. MODEL TRAINING
   ├─> Classical: train_baseline.py
   │   └─> ClassicalLinkPredictor.train()
   │       ├─> StandardScaler()            [Feature scaling]
   │       └─> Model.fit()                 [LogisticRegression/SVM/RF]
   │
   └─> Quantum: qml_trainer.py
       └─> QMLTrainer.train_and_evaluate()
           ├─> QMLLinkPredictor()          [VQC or QSVC]
           ├─> quantum_executor.py         [Backend selection]
           └─> circuit_optimizer.py        [Optional optimization]

6. EVALUATION
   └─> evaluate_baseline.py + metrics_tracker.py
       ├─> Comprehensive metrics calculation
       ├─> Visualization (ROC, PR curves)
       └─> Results persistence
```

---

## Feature Engineering Deep Dive

### Classical Features (128D for embedding_dim=32)

**Source:** `kg_embedder.py::prepare_link_features()`

**Process:**
1. For each (head, tail) pair in link_df:
   - Retrieve head embedding: `hv = _get_vec(h, reduced=False)` → 32D
   - Retrieve tail embedding: `tv = _get_vec(t, reduced=False)` → 32D
   - Compute difference: `diff = |hv - tv|` → 32D
   - Compute Hadamard product: `had = hv * tv` → 32D
   - Concatenate: `[hv, tv, diff, had]` → 128D

**Usage:**
- Input to: `ClassicalLinkPredictor.train()`
- Scaled via: `StandardScaler()` before training
- Models: LogisticRegression, SVM, RandomForest

### Quantum Features (5D for qml_dim=5)

**Source:** `kg_embedder.py::prepare_link_features_qml()`

**Process:**
1. Reduce embeddings: `reduce_to_qml_dim()` via PCA → 5D per entity
2. For each (head, tail) pair:
   - Retrieve reduced embeddings: `hv, tv` → each 5D
   - Apply mode:
     - `"diff"`: `|hv - tv|` → 5D
     - `"hadamard"`: `hv ⊙ tv` → 5D
     - `"both"`: PCA([|hv-tv|, hv⊙tv]) → 5D

**Usage:**
- Input to: `QMLLinkPredictor.fit()`
- Encoded via: `ZZFeatureMap` or `ZFeatureMap` in quantum circuit
- Models: QSVC (kernel-based), VQC (variational)

---

## Component-by-Component Feature Flow

### 1. kg_loader.py

**Purpose:** Data ingestion and preprocessing

**Key Functions:**
- `load_hetionet_edges()`: Returns DataFrame with columns ['source', 'metaedge', 'target']
- `extract_task_edges()`: Filters by relation_type, optionally limits entities
- `prepare_link_prediction_dataset()`: Creates train/test splits with negative sampling
- `get_negative_samples()`: Generates random negative pairs

**Output:** 
- `task_edges`: DataFrame with positive edges only
- `train_df`, `test_df`: DataFrames with 'source_id', 'target_id', 'label' columns

**Missing Components:**
- ✅ Hard negative mining (placeholders exist but not implemented)
  - `get_hard_negatives_similarity()`: Placeholder
  - `get_hard_negatives_adversarial()`: Placeholder

### 2. kg_embedder.py

**Purpose:** Embedding generation and feature construction

**Key Functions:**
- `train_embeddings()`: Trains TransE (PyKEEN) or generates deterministic embeddings
- `reduce_to_qml_dim()`: PCA reduction from embedding_dim → qml_dim
- `prepare_link_features()`: Classical 4×dim features
- `prepare_link_features_qml()`: Quantum qml_dim features
- `_get_vec()`: Retrieves embedding for single entity (handles unseen entities)

**Feature Dimensions:**
- Classical: `4 × embedding_dim` (default: 4×32 = 128D)
- Quantum: `qml_dim` (default: 5D)

**Missing Components:**
- ✅ All core functions implemented
- ⚠️ Hard negative mining integration (depends on kg_loader)

### 3. feature_engineering.py

**Purpose:** Additional QML feature encoding strategies

**Key Functions:**
- `make_qml_features()`: Multiple encoding strategies (diff, hadamard, concat, diff_prod, poly)
- `polynomial_features()`: Polynomial feature generation

**Status:** Utility module, not directly used in main pipeline (alternative to kg_embedder methods)

### 4. train_baseline.py

**Purpose:** Classical model training

**Key Functions:**
- `ClassicalLinkPredictor.train()`: Main training function
  - Calls `prepare_features_and_labels()` → uses `embedder.prepare_link_features()`
  - Applies `StandardScaler()`
  - Trains model (LogisticRegression/SVM/RandomForest)
  - Evaluates on train and test sets
- `regularization_path()`: Analyzes regularization strength
- `train_rbf_svc_cv()`: RBF-SVC with cross-validation

**Feature Flow:**
```
train_df → embedder.prepare_link_features() → X_train (128D)
         → StandardScaler() → X_train_scaled
         → Model.fit(X_train_scaled, y_train)
```

**Missing Components:**
- ✅ Core training implemented
- ⚠️ Some duplication with rbf_svc_fixed.py

### 5. evaluate_baseline.py

**Purpose:** Model evaluation and visualization

**Key Functions:**
- `calculate_comprehensive_metrics()`: Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Matthews correlation
- `plot_confusion_matrix()`: Visualization
- `plot_roc_curve()`: ROC curve with AUC
- `plot_precision_recall_curve()`: PR curve with AP score
- `compare_models()`: Multi-model comparison
- `generate_evaluation_report()`: Text report generation

**Usage:** Called after training to generate evaluation reports

**Missing Components:**
- ✅ All evaluation functions implemented

### 6. qml_model.py

**Purpose:** Quantum model wrapper

**Key Functions:**
- `QMLLinkPredictor.__init__()`: Configures quantum model
- `QMLLinkPredictor.fit()`: Trains VQC or QSVC
- Model selection: QSVC (kernel-based) or VQC (variational)

**Feature Flow:**
```
X_train_qml (5D) → ZZFeatureMap/ZFeatureMap → Quantum Circuit
                → QSVC: Kernel matrix computation
                → VQC: Variational optimization
```

**Missing Components:**
- ✅ Core model implementation complete

### 7. qml_trainer.py

**Purpose:** Quantum training orchestration

**Key Functions:**
- `QMLTrainer.train_and_evaluate()`: Full training pipeline
  - Prepares classical and quantum features
  - Trains classical baseline
  - Trains quantum model
  - Computes metrics
  - Saves results
- `qsvc_with_precomputed_kernel()`: Optimized QSVC with precomputed kernel
- `_save_loss_history()`: VQC loss tracking

**Feature Flow:**
```
train_df → embedder.prepare_link_features() → X_train_classical (128D)
         → embedder.prepare_link_features_qml() → X_train_qml (5D)
         → Classical: ClassicalLinkPredictor.train()
         → Quantum: QMLLinkPredictor.fit(X_train_qml)
```

**Missing Components:**
- ✅ Core training implemented
- ⚠️ Some code duplication with CLI section

### 8. quantum_executor.py

**Purpose:** Quantum backend management

**Key Functions:**
- `QuantumExecutor.__init__()`: Loads config, initializes IBM Quantum service
- `get_sampler()`: Returns sampler (statevector or hardware)
- `optimize_circuit()`: Applies circuit optimizations
- `estimate_cost()`: Cost estimation for hardware runs

**Integration:** Used by `qml_model.py` to get sampler for kernel/model execution

**Missing Components:**
- ✅ Backend selection implemented
- ⚠️ Hardware training script (train_on_heron.py) is empty

### 9. circuit_optimizer.py

**Purpose:** Quantum circuit optimization

**Key Classes:**
- `LightConePruner`: Prunes gates outside measurement light cone
- `AdaptiveTrotterization`: Adaptive time-step Trotterization
- `DistanceBasedRescaling`: Rescales couplings based on graph distance
- `ProblemSpecificCompiler`: Combines optimizations

**Status:** Advanced optimization module, not directly integrated into main pipeline

**Missing Components:**
- ⚠️ Not integrated into main training pipeline (optional enhancement)

### 10. qml_encoder.py

**Purpose:** Alternative encoding strategies

**Key Classes:**
- `QMLEncoder`: Supports amplitude, basis, and feature map encoding

**Status:** Alternative implementation, not used in main pipeline (kg_embedder handles encoding)

**Missing Components:**
- ⚠️ Not integrated into main pipeline

---

## Script Execution Flow

### run_pipeline.py

**Execution Steps:**
1. Load Hetionet → `load_hetionet_edges()`
2. Extract task edges → `extract_task_edges(relation="CtD")`
3. Prepare dataset → `prepare_link_prediction_dataset()`
4. Generate embeddings → `HetionetEmbedder.train_embeddings()`
5. Reduce dimensions → `reduce_to_qml_dim()`
6. Train classical → `ClassicalLinkPredictor.train()`
7. Train quantum → `QMLTrainer.train_and_evaluate()`
8. Generate scaling plot → `run_scalability_simulation()`

**Features Used:**
- Classical: `embedder.prepare_link_features()` → 128D
- Quantum: `embedder.prepare_link_features_qml()` → 5D

### rbf_svc_fixed.py

**Execution Steps:**
1. Load Hetionet (supports all relations or specific)
2. Extract task edges
3. Prepare train/test splits
4. Build embeddings → `_make_embedding_getter()`
5. Build classical features → `_build_features_with_getter()` → 4×dim
6. Test multiple classical models (LogisticRegression, SVM, RandomForest, etc.)
7. Build QML features → `embedder.prepare_link_features_qml()`
8. Test quantum models (QSVC, VQC variants)
9. Generate comprehensive comparison report

**Features Used:**
- Classical: `_pair_features(u, v)` → [u, v, |u-v|, u*v] → 4×dim
- Quantum: `embedder.prepare_link_features_qml()` → qml_dim

**Special Features:**
- Robust column inference (`_find_cols()`)
- Flexible embedding access (`_make_embedding_getter()`)
- Parallel model testing in fast mode
- Comprehensive metrics comparison

---

## Missing Core ETL Components

### 1. Hard Negative Mining
**Status:** Placeholders exist but not implemented
**Files:** `kg_loader.py::get_hard_negatives_*()`
**Impact:** Currently uses random negatives only

### 2. Quantum Hardware Training
**Status:** Empty file
**Files:** `quantum_layer/train_on_heron.py`, `scripts/train_on_heron.py`
**Impact:** Cannot run on real IBM Quantum hardware

### 3. Circuit Optimization Integration
**Status:** Module exists but not integrated
**Files:** `quantum_layer/circuit_optimizer.py`
**Impact:** Missing potential performance improvements

### 4. Alternative Encoder Integration
**Status:** Module exists but not used
**Files:** `quantum_layer/qml_encoder.py`
**Impact:** Missing alternative encoding strategies

### 5. API Integration for Dashboard
**Status:** Dashboard uses dummy predictions
**Files:** `benchmarking/dashboard.py`
**Impact:** Cannot make real-time predictions

---

## Feature Dimension Summary

| Stage | Dimension | Description |
|-------|-----------|-------------|
| Raw Entities | N/A | Entity IDs (strings) |
| Entity Embeddings | embedding_dim (32) | TransE or deterministic embeddings |
| Reduced Embeddings | qml_dim (5) | PCA-reduced for quantum |
| Classical Features | 4 × embedding_dim (128) | [h, t, |h-t|, h*t] |
| Quantum Features | qml_dim (5) | |h-t| or h⊙t or PCA([|h-t|, h⊙t]) |

---

## Complete File Inventory

### Core ETL Files (Required)
✅ `kg_layer/kg_loader.py` - Data loading
✅ `kg_layer/kg_embedder.py` - Embedding generation
✅ `kg_layer/feature_engineering.py` - Feature utilities
✅ `kg_layer/kg_utils.py` - KG utilities
✅ `classical_baseline/train_baseline.py` - Classical training
✅ `classical_baseline/evaluate_baseline.py` - Evaluation
✅ `quantum_layer/qml_model.py` - Quantum models
✅ `quantum_layer/qml_trainer.py` - Quantum training
✅ `quantum_layer/quantum_executor.py` - Backend management

### Supporting Files (Optional/Advanced)
⚠️ `quantum_layer/circuit_optimizer.py` - Not integrated
⚠️ `quantum_layer/qml_encoder.py` - Not integrated
⚠️ `quantum_layer/iterative_learning.py` - Advanced feature
⚠️ `quantum_layer/advanced_error_mitigation.py` - Advanced feature

### Pipeline Scripts
✅ `scripts/run_pipeline.py` - Main pipeline
✅ `scripts/rbf_svc_fixed.py` - Comprehensive comparison
✅ `scripts/e2e_smoke.py` - Smoke test

### Missing/Incomplete
❌ `quantum_layer/train_on_heron.py` - Empty
❌ `scripts/train_on_heron.py` - Empty
❌ Hard negative mining - Placeholders only

---

## Conclusion

**Core ETL Pipeline:** ✅ Complete
- All essential components are implemented
- Feature flow is well-defined and consistent
- Data transformations are properly chained

**Enhancements Needed:**
1. Implement hard negative mining
2. Complete quantum hardware training scripts
3. Integrate circuit optimization (optional)
4. Connect dashboard to real API

**No Critical Missing Files:** The core ETL pipeline is functional and complete.

