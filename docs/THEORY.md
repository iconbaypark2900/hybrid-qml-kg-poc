# Theoretical Foundations

## 1. Knowledge Graph Embeddings

### TransE: Translation-Based Embeddings

TransE models relationships as translations in the embedding space:

**h + r ≈ t**

Where:
- `h` is the head entity embedding (vector)
- `r` is the relation embedding (vector)
- `t` is the tail entity embedding (vector)

The training objective minimizes the distance between `h + r` and `t` for positive triples while maximizing it for negative triples.

**Why Translation-Based Models?**
- Simple and interpretable
- Efficient to train
- Good baseline for link prediction
- Captures relational patterns

**Alternatives:**
- **DistMult**: Diagonal matrix representation (better for symmetric relations)
- **ComplEx**: Complex embeddings (handles asymmetric relations)
- **RotatE**: Rotation-based embeddings (handles diverse relation patterns)
- **TuckER**: Tensor factorization (high expressiveness)

## 2. Link Prediction as Binary Classification

### Problem Formulation

Given a knowledge graph with entities and relations, predict whether a link (edge) exists between two entities.

**Positive Samples:** Known links from the knowledge graph
- Example: "Aspirin treats Headache" → label = 1

**Negative Samples:** Non-existing links
- Example: "Aspirin treats Cancer" → label = 0 (if not in KG)

### Feature Construction

For a pair of entities (h, t), we construct features:
- **Concatenation**: [h, t] (2×d dimensions)
- **Difference**: |h - t| (d dimensions)
- **Hadamard Product**: h ⊙ t (d dimensions)
- **Combined**: [h, t, |h-t|, h⊙t] (4×d dimensions)

### Evaluation Metrics

**PR-AUC (Precision-Recall Area Under Curve):**
- Preferred for imbalanced datasets (typical in link prediction)
- Focuses on positive class performance
- More informative than ROC-AUC when negatives >> positives

**Why PR-AUC over ROC-AUC?**
- Link prediction datasets are highly imbalanced
- PR-AUC penalizes false positives more heavily
- Better reflects practical performance

## 3. Quantum Feature Maps

### Encoding Classical Data → Quantum States

**ZZFeatureMap:**
```
|x⟩ → U_ZZ(x)|0⟩^⊗n
```

Where `U_ZZ(x)` applies entangling gates based on feature values.

**Circuit Structure:**
1. Feature encoding layer: Apply rotation gates parameterized by features
2. Entangling layer: Apply ZZ gates between qubits
3. Repeat for multiple repetitions

**Kernel Trick:**
The quantum feature map implicitly defines a kernel:
```
K(x, y) = |⟨φ(x)|φ(y)⟩|²
```

This kernel captures non-linear relationships that classical kernels might miss.

## 4. Variational Quantum Classifiers (VQC)

### Architecture

1. **Feature Map**: Encodes classical features into quantum states
2. **Ansatz**: Parameterized quantum circuit (trainable)
3. **Measurement**: Expectation value of an observable
4. **Classical Optimizer**: Updates ansatz parameters

### Ansatz Types

**RealAmplitudes:**
- Efficient parameterization
- Good for low-depth circuits
- Limited expressiveness

**EfficientSU2:**
- More expressive
- Better for complex patterns
- Deeper circuits

**TwoLocal:**
- Flexible architecture
- Customizable rotation and entanglement blocks
- Hardware-efficient designs

### Optimization Challenges

**Barren Plateaus:**
- Gradient vanishes exponentially with qubit count
- Mitigation: Shallow circuits, local cost functions

**Local Minima:**
- Classical optimizers (COBYLA, SPSA) can get stuck
- Mitigation: Multiple random initializations, better optimizers

## 5. Complexity Analysis

### Classical Scaling

**Training Complexity:** O(N²) where N is number of entities
- Need to compute pairwise features
- Quadratic in dataset size

**Storage:** O(N²) for pairwise feature matrix

### Quantum Scaling

**Theoretical:** O(log N) via amplitude amplification
- Quantum parallelism
- Logarithmic depth circuits

**Practical (NISQ):**
- Limited by noise and error rates
- Current hardware: O(N) to O(N log N) in practice
- Future fault-tolerant: Approaching O(log N)

### Crossover Point

The crossover point where quantum becomes faster depends on:
- Dataset size (N)
- Quantum hardware capabilities
- Error rates and mitigation strategies
- Classical algorithm optimizations

**Current Estimate:** N > 10,000 entities (future hardware)

## 6. Statistical Validation

### Cross-Validation

**Nested CV:**
- Outer loop: Unbiased performance estimation
- Inner loop: Hyperparameter tuning
- Prevents overfitting to test set

### Significance Testing

**Paired t-test:**
- Tests if quantum and classical differ significantly
- Assumes normal distribution of differences

**Wilcoxon signed-rank test:**
- Non-parametric alternative
- More robust to outliers

**Effect Size (Cohen's d):**
- Quantifies practical significance
- |d| < 0.2: negligible
- |d| < 0.5: small
- |d| < 0.8: medium
- |d| ≥ 0.8: large

## References

1. Bordes, A., et al. (2013). "Translating Embeddings for Modeling Multi-relational Data." NIPS.
2. McClean, J. R., et al. (2018). "Barren Plateaus in Quantum Neural Network Training Landscapes." Nature Communications.
3. Lloyd, S., et al. (2020). "Quantum embeddings for machine learning." arXiv:2001.03622.
4. Abbas, A., et al. (2021). "The power of quantum neural networks." Nature Computational Science.

