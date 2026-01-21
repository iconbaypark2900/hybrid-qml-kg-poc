# Quantum Embedding with Feature Maps

## Science

Quantum feature maps encode classical data vectors **x** ∈ ℝⁿ into quantum states |φ(**x**)⟩ through parameterized quantum circuits. The encoding process transforms classical features into quantum amplitudes and phases, enabling quantum algorithms to operate on classical data.

The fundamental operation maps each feature component xᵢ to rotation angles on qubits. For a feature map with `reps` repetition layers, the circuit applies:

1. **Encoding layer**: Rotation gates RY(xᵢ) on each qubit i, where RY(θ) = e^(-iθY/2) rotates the qubit state around the Y-axis
2. **Entangling layer**: Controlled-Z (CZ) gates create quantum correlations between qubits

The ZZ feature map, for example, implements the transformation:

|φ(**x**)⟩ = U_ZZ(**x**)|0⟩^⊗n

where U_ZZ(**x**) = ∏_{ℓ=1}^{reps} [∏_{i=1}^{n} RY(xᵢ) · ∏_{(i,j)∈E} CZ(i,j)]

Here, E represents the entanglement pattern (linear, full, or circular connectivity). The quantum state |φ(**x**)⟩ lives in a 2ⁿ-dimensional Hilbert space, providing exponential representational capacity compared to the n-dimensional classical input.

## Implementation

The codebase implements multiple feature map architectures in `quantum_layer/quantum_feature_maps.py` and `quantum_layer/qml_encoder.py`:

- **Standard feature maps**: ZZFeatureMap, ZFeatureMap, and PauliFeatureMap from Qiskit, which apply Pauli-Z rotations and entangling gates
- **Data re-uploading**: `DataReuploadingFeatureMap` encodes the same features multiple times across layers, increasing expressivity without additional qubits
- **Variational feature maps**: `VariationalFeatureMap` combines fixed feature encoding with trainable parameters θ, enabling optimization of the encoding itself
- **Link prediction feature maps**: `LinkPredictionFeatureMap` splits qubits into head and tail groups, encoding entity embeddings separately and emphasizing their relationship through cross-entangling gates

The `QMLEncoder` class in `qml_encoder.py` handles feature preparation: it validates input dimensions, pads or truncates features to match the number of qubits, and binds feature values to circuit parameters. For a feature map with `num_qubits = 5` and `feature_map_reps = 2`, the circuit applies two layers of RY rotations and CZ entangling gates, creating a quantum state that encodes the 5-dimensional classical feature vector into a 32-dimensional quantum state space.

