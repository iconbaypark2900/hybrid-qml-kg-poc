## 1. Quantum classifier algorithms QSVC and VQC with ZFeatureMap, ZZFeatureMap and PauliFeatureMap as quantum embeddings

Quantum support vector classifiers (QSVC) in Qiskit are scikit-learn SVC variants that use a quantum kernel built from a feature-map circuit $U_{\boldsymbol{x}}$, typically instantiated as ZFeatureMap, ZZFeatureMap, or PauliFeatureMap from the circuit library. Variational quantum classifiers (VQC) instead combine a data-encoding feature map with a trainable ansatz and optimize parameters via classical gradient-based or heuristic routines, but can be benchmarked with the same circuit-level and model-level metrics as QSVC. ZFeatureMap implements first‑order, single‑qubit $Z$-rotations with no entangling gates, while ZZFeatureMap introduces entangling $ZZ$ rotations via CNOT–RZ–CNOT patterns, and PauliFeatureMap generalizes this to arbitrary single- and multi-qubit Pauli terms, allowing systematic control of expressivity versus entangling-gate count and, hence, noise sensitivity.[^1][^2][^3][^4]

## 2. Quantum kernels, fidelity, similarity metrics, and mitigated vs unmitigated comparisons

A fidelity-based quantum kernel for QSVC is defined as

$$
K(\boldsymbol{x},\boldsymbol{y}) = \bigl|\langle \psi(\boldsymbol{x}) \mid \psi(\boldsymbol{y}) \rangle\bigr|^{2},
\quad \ket{\psi(\boldsymbol{x})}=U_{\boldsymbol{x}}\ket{0}^{\otimes n},
$$

and is efficiently estimated in Qiskit via the compute–uncompute method using the probability of measuring the all-zero outcome after preparing $U_{\boldsymbol{x}}^{\dagger}U_{\boldsymbol{y}}\ket{0}^{\otimes n}$. At the circuit level, state fidelity between ideal and noisy states is often quantified by $F(\rho,\sigma)=\bigl(\text{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\bigr)^{2}$, while kernel-matrix similarity under noise is captured with metrics such as relative Frobenius drift[^2][^4][^1]

$$
\Delta_{\mathrm{rel}} = \frac{\lVert K_{\text{noisy}} - K_{\text{ideal}}\rVert_{F}}{\lVert K_{\text{ideal}}\rVert_{F}}
$$

and kernel alignment

$$
A(K_{\text{noisy}}, K_{\text{ideal}}) =
\frac{\langle \mathrm{vec}(K_{\text{noisy}}), \mathrm{vec}(K_{\text{ideal}})\rangle}{
\lVert K_{\text{noisy}}\rVert_{F}\,\lVert K_{\text{ideal}}\rVert_{F}},
$$

which make it possible to compare unmitigated, mitigated, and ideal kernels across ideal simulators, noise-model simulators, and hardware backends.[^3][^4][^1][^2]

## 3. Circuit‑, kernel‑, and model‑level benchmarking metrics

The proposed QSVC benchmarking suites distinguish three layers of observables. Circuit-level metrics focus on a probe feature-map circuit and include the bitstring probability distribution $p(z)$, single-qubit observables such as[^1][^2][^3]

$$
\langle Z_j\rangle = \sum_{z\in\{0,1\}^{n}} (-1)^{z_j}\,p(z),
$$

and distribution-fidelity metrics between an observed distribution $p$ and an ideal one $p^{\star}$, for example the Hellinger fidelity

$$
F_{\mathrm{dist}}(p,p^{\star})=
\left(\sum_{z}\sqrt{p(z)p^{\star}(z)}\right)^{2},
$$

which detect bias and drift prior to full kernel evaluation. Kernel-level metrics operate on $K_{\text{ideal}},K_{\text{noisy}},K_{\text{mit}}$ using $\Delta_{\mathrm{rel}}$ and alignment to quantify geometry changes, while model-level metrics use standard classification scores such as accuracy, balanced accuracy, F1, and ROC–AUC for QSVC and classical baselines, enabling a joint view of performance and kernel quality.[^5][^2][^3][^1]

## 4. Efficient error mitigation methods in Qiskit

Noise-aware QSVC benchmarking emphasizes lightweight mitigation profiles that combine software “switches” in Qiskit Runtime and Aer with modest overhead. Key techniques include readout error mitigation (e.g., TREX at resilience level 1), dynamical decoupling via transpiler passes or runtime options, Pauli twirling and randomized compiling for gate errors, and noise-aware transpilation layouts that map logical qubits onto the least noisy physical qubits. These techniques can be systematically composed into named profiles—such as “none”, “DD+twirling”, or “resilience‑1+layout‑noise‑adaptive”—and ranked using a cost function $\text{cost} = \alpha t + \beta N_{\text{circuits}} + \gamma N_{\text{shots}}$ to select the least computationally expensive mitigation achieving acceptable kernel and QSVC metrics.[^4][^5][^2][^3][^1]

## 5. Circuit complexity, transpilation metrics, and compute resources

Because two-qubit errors dominate on current hardware, benchmarking pipelines explicitly track transpiled circuit complexity and execution resources for each feature map and mitigation profile. After transpilation to a concrete backend, Qiskit exposes depth, gate counts, and number of nonlocal gates (e.g., via $\text{depth}$, $\text{count\_ops}$, $\text{num\_nonlocal\_gates}$), while execution metadata and harness logic record wall-clock time, number of executed circuits, and total shots, which then feed into the same cost function used for mitigation ranking. Empirically, shallow, weakly entangling embeddings such as ZFeatureMap often yield lower depth and zero two-qubit gates, whereas ZZFeatureMap and expressive PauliFeatureMaps produce deeper circuits with many entanglers; correlating these complexity measures with kernel drift and QSVC accuracy under noise is central to designing scalable, noise-aware QSVC and VQC benchmarks.[^5][^2][^4][^1]

***

```python
# Box A) Device-Derived Noise Models
from qiskit.providers.aer.noise import NoiseModel

noise_model = NoiseModel.from_backend(some_backend)
```

```python
# Box B) Computing Fidelity Quantum Kernel with Compute-Uncompute Method in Qiskit
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Suppose `sampler` is a Sampler primitive configured to a backend (noisy sim or hardware)
fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=qfm)
```

```python
# Box C) Attaching Noise to Kernel Simulation
from qiskit_aer import AerSimulator

sim = AerSimulator.from_backend(device_backend)  # FakeVigo or real backend
from qiskit.primitives import Sampler
sampler = Sampler(backend=sim)

fidelity = ComputeUncompute(sampler=sampler)
quantum_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
```
