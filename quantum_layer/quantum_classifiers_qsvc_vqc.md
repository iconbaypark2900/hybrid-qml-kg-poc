# Quantum Classifiers: QSVC and VQC

## Science

**Quantum Support Vector Classifier (QSVC)** leverages quantum kernels computed from feature maps. The quantum kernel measures the similarity between data points in the quantum feature space:

K(**x**ᵢ, **x**ⱼ) = |⟨φ(**x**ᵢ)|φ(**x**ⱼ)⟩|²

where |φ(**x**)⟩ is the quantum state produced by the feature map. This kernel captures quantum correlations that may be difficult to compute classically. QSVC then trains a classical support vector machine on the kernel matrix, solving:

min_{α} ½∑ᵢⱼ αᵢαⱼyᵢyⱼK(**x**ᵢ, **x**ⱼ) - ∑ᵢ αᵢ

subject to 0 ≤ αᵢ ≤ C and ∑ᵢ αᵢyᵢ = 0, where C is the regularization parameter and yᵢ are the labels.

**Variational Quantum Classifier (VQC)** uses a hybrid quantum-classical approach. The circuit consists of a feature map U_φ(**x**) followed by a trainable ansatz U(θ):

|ψ(**x**, θ)⟩ = U(θ)U_φ(**x**)|0⟩^⊗n

The ansatz (e.g., RealAmplitudes or EfficientSU2) applies parameterized rotations and entangling gates. The expectation value ⟨ψ(**x**, θ)|Z|ψ(**x**, θ)⟩ is measured, where Z is a Pauli-Z observable. This value is used to compute predictions, and the parameters θ are optimized via classical optimizers (COBYLA, SPSA) to minimize a loss function such as cross-entropy:

L(θ) = -∑ᵢ [yᵢ log(σ(⟨Z⟩ᵢ)) + (1-yᵢ)log(1-σ(⟨Z⟩ᵢ))]

where σ is the sigmoid function and ⟨Z⟩ᵢ is the expectation value for sample i.

## Implementation

The `QMLLinkPredictor` class in `quantum_layer/qml_model.py` implements both algorithms. For **QSVC**, it constructs a `FidelityQuantumKernel` (or `FidelityStatevectorKernel` for exact simulation) from the feature map and wraps it in Qiskit's `QSVC` class, which internally uses scikit-learn's SVC with the precomputed quantum kernel matrix. The implementation supports precomputed kernels for efficiency, computing K_train and K_test matrices once and reusing them during hyperparameter grid search over the regularization parameter C.

For **VQC**, the code builds a `VQC` instance with three components: (1) the feature map (ZZFeatureMap or ZFeatureMap), (2) a variational ansatz (RealAmplitudes or EfficientSU2 with configurable `ansatz_reps`), and (3) a classical optimizer (COBYLA or SPSA with `max_iter` iterations). The VQC trains by iteratively evaluating the quantum circuit, computing gradients or using derivative-free optimization to update θ, and measuring expectation values to compute predictions. Both models integrate with `QuantumExecutor` for execution on simulators or IBM Quantum hardware, with optional error mitigation techniques (dynamical decoupling, Pauli twirling, ZNE) configured through the executor.

