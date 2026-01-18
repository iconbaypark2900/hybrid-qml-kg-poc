# Comprehensive Optimization Plan for Hetionet CtD Link Prediction

## Current Performance Baseline (CtD Dataset)
- **Best Classical**: RandomForest - PR-AUC: 0.6244
- **Best Quantum**: QSVC - PR-AUC: 0.5564
- **Gap**: -0.0680 (Classical wins)

## Target Goals
1. **Classical**: Improve from 0.6244 to 0.75+ PR-AUC
2. **Quantum**: Improve from 0.5564 to 0.70+ PR-AUC
3. **Speed**: Reduce training time while maintaining quality
4. **Generalization**: Ensure improvements work across different relations

---

## Part 1: Classical Optimizations

### 1.1 Advanced Knowledge Graph Embeddings ⭐⭐⭐
**Current**: TransE (basic) or random fallback embeddings (32D)
**Impact**: HIGH - Better embeddings = better features

**Improvements**:
- **ComplEx**: Complex-valued embeddings for asymmetric relations
  - Formula: Re(⟨h, r, conj(t)⟩)
  - Better captures compound-disease interactions

- **RotatE**: Rotation in complex space
  - Formula: h ∘ r ≈ t (element-wise rotation)
  - Excellent for hierarchical and symmetric patterns

- **DistMult**: Bilinear model
  - Formula: ⟨h, r, t⟩
  - Fast, good for symmetric relations

**Implementation**:
```python
# Use PyKEEN with multiple embedding methods
embedder = HetionetEmbedder(
    embedding_method='ComplEx',  # or 'RotatE', 'DistMult'
    embedding_dim=64,  # Increase from 32
    num_epochs=100,    # More training
    batch_size=512,
    learning_rate=0.001,
    regularization=0.01
)
```

**Expected Improvement**: +0.05 to +0.10 PR-AUC

### 1.2 Enhanced Feature Engineering ⭐⭐⭐
**Current**: [h, t, |h-t|, h*t] = 4*dim features (128D)
**Impact**: HIGH - Rich features improve all models

**New Features**:
1. **Embedding-based** (keep existing, add):
   - Cosine similarity: cos(h, t)
   - L2 distance: ||h - t||²
   - Dot product: h · t
   - Concatenation weighted: [α*h, β*t]

2. **Graph-based** (NEW):
   - Node degree (in/out for compounds and diseases)
   - Common neighbors count
   - Shortest path length (if exists)
   - Jaccard coefficient of neighbors
   - Adamic-Adar index
   - Preferential attachment score

3. **Domain-specific for CtD** (NEW):
   - Compound degree centrality
   - Disease degree centrality
   - Number of shared drug classes
   - Co-occurrence in known treatments

**Implementation**:
```python
def enhanced_features(h, t, graph, h_id, t_id):
    # Embedding features (existing + new)
    emb_feats = np.concatenate([
        h, t,                      # Original embeddings
        np.abs(h - t),             # Absolute difference
        h * t,                      # Hadamard product
        [np.dot(h, t)],            # Dot product
        [cosine_similarity(h, t)], # Cosine similarity
        [np.linalg.norm(h - t)]    # L2 distance
    ])

    # Graph features (NEW)
    graph_feats = np.array([
        graph.degree(h_id),                    # Compound degree
        graph.degree(t_id),                    # Disease degree
        len(nx.common_neighbors(graph, h_id, t_id)),  # Common neighbors
        jaccard_coefficient(graph, h_id, t_id),
        adamic_adar_index(graph, h_id, t_id),
    ])

    return np.concatenate([emb_feats, graph_feats])
```

**Expected Improvement**: +0.03 to +0.08 PR-AUC

### 1.3 Better Hyperparameter Optimization ⭐⭐
**Current**: Simple grid search with limited parameter space
**Impact**: MEDIUM - Fine-tuning can significantly improve results

**Improvements**:
- **Bayesian Optimization** (Optuna/HyperOpt)
- **Larger search space**:
  - RandomForest: n_estimators [100, 500], max_depth [5, 50], min_samples_split
  - SVM-RBF: C [0.01, 100], gamma [0.001, 1.0]
  - LogisticRegression: C [0.001, 10], penalty ['l1', 'l2', 'elasticnet']

- **Early stopping** for faster convergence
- **Cross-validation** improvements (StratifiedKFold, 10-fold)

**Expected Improvement**: +0.02 to +0.05 PR-AUC

### 1.4 Ensemble Methods ⭐⭐
**Current**: Individual models only
**Impact**: MEDIUM - Ensembles often outperform single models

**Improvements**:
- **Stacking**: Train meta-model on predictions from multiple base models
  - Base: RandomForest, SVM-RBF, GradientBoosting, XGBoost
  - Meta: LogisticRegression or LightGBM

- **Voting**: Soft voting with optimized weights
- **Boosting**: XGBoost, LightGBM with link prediction objective

**Expected Improvement**: +0.03 to +0.07 PR-AUC

### 1.5 Class Imbalance Handling ⭐
**Current**: class_weight='balanced', 1:1 ratio
**Impact**: LOW-MEDIUM - Already handled but can be improved

**Improvements**:
- **SMOTE**: Generate synthetic positive examples
- **Focal Loss**: Focus on hard examples
- **Adjusted ratios**: Try 1:2, 1:3 negative:positive
- **Cost-sensitive learning**: Different misclassification costs

**Expected Improvement**: +0.01 to +0.03 PR-AUC

---

## Part 2: Quantum Optimizations

### 2.1 Advanced Quantum Feature Engineering ⭐⭐⭐
**Current**: |h - t| in 5D space
**Impact**: HIGH - Better quantum features are crucial

**Improvements**:
1. **Multiple encoding strategies**:
   - Amplitude encoding: ψ = (h ⊗ t) / ||h ⊗ t||
   - Phase encoding: angles from features
   - Hybrid: Classical + quantum features

2. **Feature selection for quantum**:
   - Use mutual information to select best 5 features
   - PCA with more components (8-10) then select top 5
   - Kernel PCA for non-linear reduction

3. **Normalization strategies**:
   - Tanh squashing to [-1, 1]
   - Min-max scaling per qubit
   - Z-score normalization

**Implementation**:
```python
def optimized_qml_features(h, t, method='hybrid'):
    if method == 'amplitude':
        # Tensor product and normalize
        feat = np.kron(h, t)
        feat = feat[:5] / (np.linalg.norm(feat[:5]) + 1e-9)
    elif method == 'phase':
        # Phase encoding from angles
        feat = np.arctan2(h[:5], t[:5])
    elif method == 'hybrid':
        # Combine multiple strategies
        diff = np.abs(h[:5] - t[:5])
        prod = (h[:5] * t[:5])
        feat = (diff + prod) / 2

    # Normalize to quantum-friendly range
    feat = np.tanh(feat)
    return feat
```

**Expected Improvement**: +0.05 to +0.15 PR-AUC

### 2.2 Optimized Quantum Circuits ⭐⭐
**Current**: ZZFeatureMap + RealAmplitudes ansatz
**Impact**: MEDIUM-HIGH - Circuit design affects expressivity

**Improvements**:
1. **Feature Maps**:
   - **Pauli Feature Map**: More expressive rotations
   - **Custom feature maps**: Tailored to graph structure
   - **Data-reuploading**: Multiple encoding layers

2. **Ansatzes**:
   - **Hardware-efficient**: Optimized for real QPUs
   - **Problem-inspired**: Graph-aware entanglement
   - **Layered approach**: Alternate encoding + variational layers

3. **Circuit depth optimization**:
   - Reduce reps from 3 to 2 (faster, less noise)
   - Use SWAP networks for better connectivity
   - Prune low-gradient parameters

**Implementation**:
```python
# Custom feature map with data reuploading
def custom_feature_map(num_qubits, reps=2):
    fm = QuantumCircuit(num_qubits)
    for rep in range(reps):
        # Rotation layer
        for i in range(num_qubits):
            fm.ry(Parameter(f'x[{i}]_rep{rep}'), i)
        # Entanglement layer
        for i in range(num_qubits - 1):
            fm.cx(i, i+1)
        # Data re-upload
        for i in range(num_qubits):
            fm.rz(Parameter(f'x[{i}]_rep{rep}_z'), i)
    return fm
```

**Expected Improvement**: +0.03 to +0.08 PR-AUC

### 2.3 Better Optimizers and Training ⭐⭐
**Current**: COBYLA (gradient-free), SPSA
**Impact**: MEDIUM - Better optimization finds better parameters

**Improvements**:
1. **Advanced optimizers**:
   - **ADAM**: Adaptive learning rates
   - **L-BFGS-B**: Quasi-Newton method
   - **Nesterov momentum**: Accelerated convergence

2. **Training strategies**:
   - **Warm start**: Initialize from classical model
   - **Layer-wise training**: Train ansatz layers sequentially
   - **Learning rate scheduling**: Decay over epochs
   - **Early stopping**: Monitor validation loss

3. **Increase iterations**:
   - Current: 50 iterations
   - Proposed: 200-500 iterations with early stopping

**Expected Improvement**: +0.02 to +0.05 PR-AUC

### 2.4 Quantum Error Mitigation ⭐
**Current**: None (using simulator)
**Impact**: LOW (for simulator), HIGH (for real QPU)

**Improvements**:
- **Zero-noise extrapolation**: Extrapolate to zero noise
- **Probabilistic error cancellation**: Cancel known error channels
- **Measurement error mitigation**: Calibration matrices
- **Readout error mitigation**: Post-processing corrections

**Expected Improvement**: +0.00 to +0.02 PR-AUC (simulator), +0.05 to +0.10 (QPU)

### 2.5 Hybrid Quantum-Classical ⭐⭐
**Current**: Separate quantum and classical models
**Impact**: MEDIUM - Combine strengths of both

**Improvements**:
1. **Quantum-enhanced features**: Use quantum kernel as additional feature
2. **Ensemble**: Combine quantum + classical predictions
3. **Quantum feature selector**: Use QML to select classical features
4. **Multi-stage**: Quantum pre-filter → classical refinement

**Expected Improvement**: +0.03 to +0.10 PR-AUC

---

## Part 3: Training and Evaluation Improvements

### 3.1 Data Augmentation ⭐⭐
**Current**: Only real edges
**Impact**: MEDIUM - More data helps quantum models

**Improvements**:
- **Hard negative mining**: Sample challenging negatives
- **Negative sampling strategies**:
  - Random (current)
  - Type-aware (same entity types)
  - Degree-aware (similar connectivity)
- **Positive augmentation**:
  - Add transitive edges (if A→B and B→C, maybe A→C)
  - Use multi-hop paths as weak positives

**Expected Improvement**: +0.02 to +0.05 PR-AUC

### 3.2 Better Cross-Validation ⭐
**Current**: Simple train/test split
**Impact**: LOW-MEDIUM - More reliable estimates

**Improvements**:
- **Nested CV**: Inner loop for hyperparameters, outer for evaluation
- **Temporal split**: If data has time component
- **Entity-based split**: Ensure no entity leakage
- **Stratified by entity type**: Balance compound/disease types

**Expected Improvement**: More reliable estimates, +0.01 to +0.02 PR-AUC

### 3.3 Curriculum Learning ⭐
**Current**: Random sampling
**Impact**: LOW-MEDIUM - Especially helpful for quantum

**Improvements**:
- Start with easy examples (high confidence)
- Gradually add harder examples
- Use semi-supervised learning for unlabeled edges

**Expected Improvement**: +0.01 to +0.03 PR-AUC

---

## Implementation Priority

### Phase 1: High-Impact Classical (Week 1)
1. ✅ Advanced KG embeddings (ComplEx/RotatE) - `kg_layer/advanced_embeddings.py`
2. ✅ Enhanced feature engineering - `kg_layer/enhanced_features.py`
3. ✅ Bayesian hyperparameter optimization - `scripts/bayesian_optimization.py`

### Phase 2: High-Impact Quantum (Week 2)
1. ✅ Optimized quantum feature engineering - `quantum_layer/advanced_qml_features.py`
2. ✅ Custom feature maps and ansatzes - `quantum_layer/custom_circuits.py`
3. ✅ Better optimizers and training - `quantum_layer/advanced_training.py`

### Phase 3: Ensemble and Hybrid (Week 3)
1. ✅ Ensemble methods - `classical_baseline/ensemble_models.py`
2. ✅ Hybrid quantum-classical - `hybrid/quantum_classical_ensemble.py`
3. ✅ Data augmentation - `kg_layer/data_augmentation.py`

### Phase 4: Validation and Tuning (Week 4)
1. ✅ Run comprehensive experiments
2. ✅ Statistical significance testing
3. ✅ Documentation and reporting

---

## Expected Final Results

### Conservative Estimates
- **Classical**: 0.72 - 0.78 PR-AUC (+0.08 to +0.14)
- **Quantum**: 0.64 - 0.72 PR-AUC (+0.08 to +0.16)
- **Gap**: Quantum competitive or better!

### Optimistic Estimates
- **Classical**: 0.75 - 0.82 PR-AUC (+0.11 to +0.18)
- **Quantum**: 0.70 - 0.78 PR-AUC (+0.14 to +0.22)
- **Gap**: Quantum wins in specific scenarios!

---

## Next Steps

Run the following command to start Phase 1:

```bash
# Create optimized embeddings
python scripts/optimize_embeddings.py --relation CtD --method ComplEx --dim 64

# Train with enhanced features
python scripts/train_enhanced.py --relation CtD --fast_mode

# Compare results
python scripts/compare_optimizations.py
```

Each optimization is modular and can be tested independently!
