# quantum_layer/circuit_optimizer.py

"""
Circuit Optimization Framework Inspired by AlphaEvolve
Implements adaptive compilation, light cone pruning, and problem-specific optimization.

AlphaEvolve-inspired but doesn't require LLM - uses algorithmic approaches instead.
"""

import numpy as np
from typing import List, Callable, Dict, Any, Tuple, Optional
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import Operator
import logging

logger = logging.getLogger(__name__)


class LightConePruner:
    """
    Prune gates outside the light cone of measurement operators.
    
    Key insight from Google paper: Many gates don't affect final measurement,
    so removing them reduces circuit depth and error.
    """
    
    @staticmethod
    def compute_light_cone(
        circuit: QuantumCircuit,
        measurement_qubits: List[int],
        max_distance: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Compute the causal light cone from measurement qubits backward through circuit.
        
        Args:
            circuit: Quantum circuit to analyze
            measurement_qubits: Qubits where measurements occur
            max_distance: Optional maximum distance (depth) to trace back
            
        Returns:
            List of (gate_index, qubit) tuples that affect measurement
        """
        # Start with measurement qubits
        affected_qubits = set(measurement_qubits)
        light_cone_gates = []
        
        # Traverse circuit backward
        instructions = list(enumerate(circuit.data))
        instructions.reverse()
        
        for gate_idx, (gate, qubits, clbits) in instructions:
            qubit_indices = [circuit.qubits.index(q) for q in qubits]
            
            # If gate touches any affected qubit, it's in the light cone
            if any(q in affected_qubits for q in qubit_indices):
                light_cone_gates.append((gate_idx, tuple(qubit_indices)))
                # Gate's input qubits are now affected
                affected_qubits.update(qubit_indices)
        
        light_cone_gates.reverse()  # Restore forward order
        return light_cone_gates
    
    @staticmethod
    def prune_circuit(
        circuit: QuantumCircuit,
        measurement_qubits: List[int]
    ) -> QuantumCircuit:
        """
        Create a pruned circuit containing only gates in the light cone.
        """
        light_cone = LightConePruner.compute_light_cone(circuit, measurement_qubits)
        light_cone_indices = {idx for idx, _ in light_cone}
        
        # Create new circuit with only light cone gates
        pruned = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
        
        for gate_idx, (gate, qubits, clbits) in enumerate(circuit.data):
            if gate_idx in light_cone_indices:
                pruned.append(gate, qubits, clbits)
        
        original_depth = circuit.depth()
        pruned_depth = pruned.depth()
        reduction = 1 - (pruned_depth / original_depth) if original_depth > 0 else 0
        
        logger.info(f"Pruned circuit: {original_depth} → {pruned_depth} depth ({reduction:.1%} reduction)")
        
        return pruned


class AdaptiveTrotterization:
    """
    Adaptive time-step Trotterization based on local Hamiltonian structure.
    
    Google paper insight: Not all terms need same time resolution.
    Strong couplings get finer steps, weak couplings get coarser steps.
    """
    
    def __init__(self, hamiltonian_terms: List[Tuple[float, Any]], tolerance: float = 0.01):
        """
        Args:
            hamiltonian_terms: List of (coefficient, operator) pairs
            tolerance: Target Trotter error tolerance
        """
        self.terms = hamiltonian_terms
        self.tolerance = tolerance
        
    def compute_adaptive_steps(self, total_time: float) -> Dict[int, int]:
        """
        Compute number of Trotter steps for each term based on its strength.
        
        Returns:
            Dict mapping term_index → num_steps
        """
        # Sort terms by strength
        strengths = [abs(coeff) for coeff, _ in self.terms]
        max_strength = max(strengths)
        
        # Stronger terms need more steps for same error
        # Trotter error ~ (Δt)² * ||H||²
        # So steps ~ ||H|| / sqrt(tolerance)
        
        step_allocation = {}
        for i, strength in enumerate(strengths):
            relative_strength = strength / max_strength
            # Minimum 1 step, scale with sqrt of relative strength
            steps = max(1, int(np.ceil(relative_strength * 10 / np.sqrt(self.tolerance))))
            step_allocation[i] = steps
        
        logger.info(f"Adaptive Trotter steps: {step_allocation}")
        return step_allocation
    
    def generate_adaptive_circuit(
        self,
        total_time: float,
        num_qubits: int,
        term_circuits: List[Callable[[float], QuantumCircuit]]
    ) -> QuantumCircuit:
        """
        Generate Trotterized circuit with adaptive time steps.
        
        Args:
            total_time: Total evolution time
            num_qubits: Number of qubits
            term_circuits: List of functions that generate circuits for each term
                          Each function takes time as argument
            
        Returns:
            Optimized Trotterized circuit
        """
        steps = self.compute_adaptive_steps(total_time)
        circuit = QuantumCircuit(num_qubits)
        
        # Find LCM of all steps for proper interleaving
        max_steps = max(steps.values())
        
        for step in range(max_steps):
            for term_idx, num_steps in steps.items():
                # Apply this term if step is multiple of (max_steps / num_steps)
                if step % (max_steps // num_steps) == 0:
                    dt = total_time / num_steps
                    term_circuit = term_circuits[term_idx](dt)
                    circuit.compose(term_circuit, inplace=True)
        
        return circuit


class DistanceBasedRescaling:
    """
    Rescale coupling terms based on graph distance.
    
    Google paper technique: Terms between distant nodes can be approximated
    or handled differently than local terms.
    """
    
    @staticmethod
    def compute_graph_distances(num_nodes: int, edges: List[Tuple[int, int]]) -> np.ndarray:
        """
        Compute all-pairs shortest path distances on graph.
        
        Returns:
            Distance matrix (num_nodes x num_nodes)
        """
        # Initialize with infinity
        distances = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(distances, 0)
        
        # Set edge distances to 1
        for u, v in edges:
            distances[u, v] = 1
            distances[v, u] = 1
        
        # Floyd-Warshall algorithm
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    distances[i, j] = min(distances[i, j], distances[i, k] + distances[k, j])
        
        return distances
    
    @staticmethod
    def rescale_couplings(
        couplings: Dict[Tuple[int, int], float],
        distance_threshold: int = 3
    ) -> Dict[Tuple[int, int], float]:
        """
        Rescale or prune couplings based on graph distance.
        
        Args:
            couplings: Dict of (i, j) → coupling_strength
            distance_threshold: Prune couplings beyond this distance
            
        Returns:
            Rescaled coupling dict
        """
        # Extract edges
        edges = list(couplings.keys())
        nodes = set()
        for i, j in edges:
            nodes.add(i)
            nodes.add(j)
        
        num_nodes = max(nodes) + 1
        distances = DistanceBasedRescaling.compute_graph_distances(num_nodes, edges)
        
        rescaled = {}
        for (i, j), strength in couplings.items():
            dist = distances[i, j]
            
            if dist <= distance_threshold:
                # Keep strong local couplings as-is
                rescaled[(i, j)] = strength
            else:
                # Prune distant weak couplings
                if abs(strength) > 0.1:  # Only keep if strong enough
                    # Decay with distance
                    rescaled[(i, j)] = strength / dist
        
        pruned_count = len(couplings) - len(rescaled)
        logger.info(f"Distance-based rescaling: pruned {pruned_count}/{len(couplings)} couplings")
        
        return rescaled


class ProblemSpecificCompiler:
    """
    Compile circuits optimized for specific problem structure.
    
    Combines light cone pruning, adaptive Trotterization, and distance rescaling.
    """
    
    def __init__(
        self,
        num_qubits: int,
        measurement_qubits: List[int],
        coupling_graph: Optional[Dict[Tuple[int, int], float]] = None
    ):
        self.num_qubits = num_qubits
        self.measurement_qubits = measurement_qubits
        self.coupling_graph = coupling_graph or {}
        
    def compile_feature_map(
        self,
        features: np.ndarray,
        reps: int = 2,
        entanglement: str = "linear"
    ) -> QuantumCircuit:
        """
        Compile optimized feature map encoding.
        
        Standard approach:
            circuit = ZZFeatureMap(num_qubits, reps=reps)
            circuit = circuit.bind_parameters(features)
        
        Optimized approach:
            - Use distance-based rescaling for ZZ terms
            - Apply light cone pruning
            - Adaptive repetition based on feature variance
        """
        # Analyze feature importance (variance as proxy)
        feature_variance = np.var(features)
        
        # Adaptive reps: fewer for low-variance features
        effective_reps = max(1, int(reps * min(1.0, feature_variance / 0.1)))
        
        # Create base feature map
        base_circuit = ZZFeatureMap(
            feature_dimension=len(features),
            reps=effective_reps,
            entanglement=entanglement
        )
        circuit = base_circuit.bind_parameters(features)
        
        # Apply light cone pruning
        pruned_circuit = LightConePruner.prune_circuit(circuit, self.measurement_qubits)
        
        logger.info(f"Optimized feature map: reps {reps}→{effective_reps}, "
                   f"depth {circuit.depth()}→{pruned_circuit.depth()}")
        
        return pruned_circuit
    
    def compile_ansatz(
        self,
        parameters: np.ndarray,
        ansatz_type: str = "RealAmplitudes",
        reps: int = 3
    ) -> QuantumCircuit:
        """
        Compile optimized variational ansatz.
        """
        # Create base ansatz
        if ansatz_type == "RealAmplitudes":
            from qiskit.circuit.library import RealAmplitudes
            base_ansatz = RealAmplitudes(self.num_qubits, reps=reps)
        elif ansatz_type == "EfficientSU2":
            from qiskit.circuit.library import EfficientSU2
            base_ansatz = EfficientSU2(self.num_qubits, reps=reps)
        else:
            raise ValueError(f"Unknown ansatz type: {ansatz_type}")
        
        circuit = base_ansatz.bind_parameters(parameters)
        
        # Apply light cone pruning (critical for VQC)
        pruned = LightConePruner.prune_circuit(circuit, self.measurement_qubits)
        
        return pruned
    
    def optimize_full_circuit(
        self,
        feature_map: QuantumCircuit,
        ansatz: QuantumCircuit
    ) -> QuantumCircuit:
        """
        Apply final optimizations to complete circuit.
        """
        # Compose circuits
        full_circuit = feature_map.compose(ansatz)
        
        # Transpile with optimization
        # Level 3 = highest optimization (may be slow)
        optimized = transpile(
            full_circuit,
            optimization_level=3,
            seed_transpiler=42
        )
        
        # Calculate reduction
        original_depth = full_circuit.depth()
        original_gates = sum(full_circuit.count_ops().values())
        optimized_depth = optimized.depth()
        optimized_gates = sum(optimized.count_ops().values())
        
        logger.info(f"Full circuit optimization:")
        logger.info(f"  Depth: {original_depth} → {optimized_depth} ({optimized_depth/original_depth:.2%})")
        logger.info(f"  Gates: {original_gates} → {optimized_gates} ({optimized_gates/original_gates:.2%})")
        
        return optimized


# Benchmark function
def benchmark_optimization():
    """
    Compare standard vs optimized circuit compilation.
    """
    num_qubits = 5
    features = np.random.rand(5)
    parameters = np.random.rand(20)  # For RealAmplitudes(5, reps=3)
    measurement_qubits = [0]  # Measure only first qubit
    
    # Standard compilation
    logger.info("=" * 50)
    logger.info("STANDARD COMPILATION")
    logger.info("=" * 50)
    
    standard_fm = ZZFeatureMap(num_qubits, reps=2)
    standard_fm = standard_fm.bind_parameters(features)
    
    standard_ansatz = RealAmplitudes(num_qubits, reps=3)
    standard_ansatz = standard_ansatz.bind_parameters(parameters)
    
    standard_circuit = standard_fm.compose(standard_ansatz)
    
    print(f"Standard circuit:")
    print(f"  Depth: {standard_circuit.depth()}")
    print(f"  Gates: {sum(standard_circuit.count_ops().values())}")
    print(f"  Gate breakdown: {standard_circuit.count_ops()}")
    
    # Optimized compilation
    logger.info("=" * 50)
    logger.info("OPTIMIZED COMPILATION")
    logger.info("=" * 50)
    
    compiler = ProblemSpecificCompiler(
        num_qubits=num_qubits,
        measurement_qubits=measurement_qubits
    )
    
    optimized_fm = compiler.compile_feature_map(features, reps=2)
    optimized_ansatz = compiler.compile_ansatz(parameters, reps=3)
    optimized_circuit = compiler.optimize_full_circuit(optimized_fm, optimized_ansatz)
    
    print(f"\nOptimized circuit:")
    print(f"  Depth: {optimized_circuit.depth()}")
    print(f"  Gates: {sum(optimized_circuit.count_ops().values())}")
    print(f"  Gate breakdown: {optimized_circuit.count_ops()}")
    
    # Calculate improvement
    depth_reduction = 1 - (optimized_circuit.depth() / standard_circuit.depth())
    gate_reduction = 1 - (sum(optimized_circuit.count_ops().values()) / 
                         sum(standard_circuit.count_ops().values()))
    
    print(f"\nImprovement:")
    print(f"  Depth reduction: {depth_reduction:.1%}")
    print(f"  Gate reduction: {gate_reduction:.1%}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    benchmark_optimization()