"""
Quantum Circuit Optimization Techniques

Implements advanced optimization techniques for quantum circuits used in quantum machine learning,
including parameter optimization, gate synthesis, and topology-aware compilation.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from scipy.optimize import minimize
import itertools

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    try:
        from qiskit.circuit import Parameter
    except ImportError:
        from qiskit.circuit.parameter import Parameter
    from qiskit.compiler import transpile
    from qiskit.providers import Backend
    from qiskit.quantum_info import Operator
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGates, OptimizeSwapBeforeMeasure, RemoveResetInZeroState
    from qiskit.converters import circuit_to_dag, dag_to_circuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available. Quantum circuit optimization will be limited.")


class QuantumCircuitOptimizer:
    """
    Quantum Circuit Optimizer
    
    Optimizes quantum circuits for better performance on quantum hardware
    by reducing gate count, optimizing parameters, and adapting to hardware topology.
    """
    
    def __init__(
        self,
        backend: Optional[Backend] = None,
        optimization_level: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ):
        """
        Args:
            backend: Quantum backend for hardware-aware optimization
            optimization_level: Level of optimization (0-3)
            max_iterations: Maximum iterations for parameter optimization
            tolerance: Convergence tolerance
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum circuit optimization")
        
        self.backend = backend
        self.optimization_level = optimization_level
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Store optimization results
        self.optimization_history = []
        self.initial_circuit = None
        self.optimized_circuit = None
    
    def optimize_circuit(
        self,
        circuit: QuantumCircuit,
        objective_fn: Optional[Callable] = None,
        method: str = 'hybrid'
    ) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        """
        Optimize the quantum circuit using specified method.
        
        Args:
            circuit: Quantum circuit to optimize
            objective_fn: Objective function to minimize (optional)
            method: Optimization method ('transpile', 'parameter', 'hybrid')
            
        Returns:
            Optimized circuit and optimization metrics
        """
        self.initial_circuit = circuit.copy()
        
        if method == 'transpile':
            # Use Qiskit's built-in transpiler
            optimized_circuit = self._optimize_with_transpiler(circuit)
        elif method == 'parameter':
            # Optimize variational parameters
            if objective_fn is None:
                raise ValueError("Objective function required for parameter optimization")
            optimized_circuit = self._optimize_parameters(circuit, objective_fn)
        elif method == 'hybrid':
            # Combine both approaches
            circuit_transpiled = self._optimize_with_transpiler(circuit)
            if objective_fn is not None:
                optimized_circuit = self._optimize_parameters(circuit_transpiled, objective_fn)
            else:
                optimized_circuit = circuit_transpiled
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        self.optimized_circuit = optimized_circuit
        
        # Compute metrics
        metrics = self._compute_optimization_metrics(circuit, optimized_circuit)
        
        return optimized_circuit, metrics
    
    def _optimize_with_transpiler(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Optimize circuit using Qiskit's transpiler."""
        if self.backend is not None:
            # Optimize for specific backend
            optimized_circuit = transpile(
                circuit,
                backend=self.backend,
                optimization_level=self.optimization_level
            )
        else:
            # Optimize without specific backend
            optimized_circuit = transpile(
                circuit,
                optimization_level=self.optimization_level
            )
        
        return optimized_circuit
    
    def _optimize_parameters(
        self,
        circuit: QuantumCircuit,
        objective_fn: Callable
    ) -> QuantumCircuit:
        """Optimize parameters in parameterized circuits."""
        # Extract parameters from the circuit
        parameters = list(circuit.parameters)
        
        if not parameters:
            # If no parameters, return the circuit as is
            return circuit.copy()
        
        # Create a function to evaluate the objective with given parameter values
        def objective_wrapper(param_values):
            # Bind parameters to the circuit
            bound_circuit = circuit.bind_parameters(
                dict(zip(parameters, param_values))
            )
            
            # Evaluate the objective function
            return objective_fn(bound_circuit)
        
        # Initial parameter values (random initialization)
        initial_params = np.random.uniform(-np.pi, np.pi, len(parameters))
        
        # Optimize parameters
        result = minimize(
            objective_wrapper,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        # Store optimization history
        self.optimization_history.append(result)
        
        # Create final optimized circuit with optimal parameters
        optimized_circuit = circuit.bind_parameters(
            dict(zip(parameters, result.x))
        )
        
        return optimized_circuit
    
    def _compute_optimization_metrics(
        self,
        original_circuit: QuantumCircuit,
        optimized_circuit: QuantumCircuit
    ) -> Dict[str, Any]:
        """Compute metrics comparing original and optimized circuits."""
        original_depth = original_circuit.depth()
        optimized_depth = optimized_circuit.depth()
        
        original_size = original_circuit.size()
        optimized_size = optimized_circuit.size()
        
        original_count_ops = original_circuit.count_ops()
        optimized_count_ops = optimized_circuit.count_ops()
        
        metrics = {
            'original_depth': original_depth,
            'optimized_depth': optimized_depth,
            'depth_reduction': original_depth - optimized_depth,
            'depth_reduction_percentage': (original_depth - optimized_depth) / original_depth * 100 if original_depth > 0 else 0,
            'original_size': original_size,
            'optimized_size': optimized_size,
            'size_reduction': original_size - optimized_size,
            'size_reduction_percentage': (original_size - optimized_size) / original_size * 100 if original_size > 0 else 0,
            'original_ops': dict(original_count_ops),
            'optimized_ops': dict(optimized_count_ops),
            'optimization_successful': True
        }
        
        return metrics


class VariationalParameterOptimizer:
    """
    Variational Parameter Optimizer
    
    Optimizes parameters in variational quantum circuits using gradient-free
    and gradient-based optimization methods.
    """
    
    def __init__(
        self,
        method: str = 'cobyla',
        max_iter: int = 100,
        tol: float = 1e-6,
        learning_rate: float = 0.01
    ):
        """
        Args:
            method: Optimization method ('cobyla', 'bfgs', 'sgd', 'adam')
            max_iter: Maximum iterations
            tol: Tolerance for convergence
            learning_rate: Learning rate for gradient-based methods
        """
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        
        # Store optimization history
        self.history = {
            'loss': [],
            'params': [],
            'gradient_norm': []
        }
    
    def optimize(
        self,
        circuit: QuantumCircuit,
        objective_fn: Callable,
        initial_params: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize parameters in the variational circuit.
        
        Args:
            circuit: Parameterized quantum circuit
            objective_fn: Objective function to minimize
            initial_params: Initial parameter values (if None, random initialization)
            
        Returns:
            Optimized parameters and optimization metrics
        """
        # Extract parameters from circuit
        params = list(circuit.parameters)
        n_params = len(params)
        
        if n_params == 0:
            return np.array([]), {'iterations': 0, 'final_loss': 0.0, 'converged': True}
        
        # Initialize parameters
        if initial_params is None or len(initial_params) != n_params:
            initial_params = np.random.uniform(-np.pi, np.pi, n_params)
        
        # Create objective function wrapper
        def obj_wrapper(params_flat):
            # Reshape parameters if needed
            params_reshaped = params_flat if isinstance(params_flat, np.ndarray) else np.array(params_flat)
            
            # Bind parameters to circuit
            try:
                bound_circuit = circuit.bind_parameters(
                    dict(zip(params, params_reshaped))
                )
            except:
                # If binding fails, return a high loss
                return 1e6
            
            # Evaluate objective
            loss = objective_fn(bound_circuit)
            
            # Store in history
            self.history['loss'].append(float(loss))
            self.history['params'].append(params_reshaped.copy())
            
            return float(loss)
        
        if self.method in ['cobyla', 'bfgs', 'l-bfgs-b']:
            # Use scipy optimizers
            result = minimize(
                obj_wrapper,
                initial_params,
                method=self.method,
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )
            
            final_params = result.x
            converged = result.success
            n_iters = len(self.history['loss']) if self.history['loss'] else result.nit
            
        elif self.method == 'sgd':
            # Simple stochastic gradient descent
            current_params = initial_params.copy()
            converged = False
            n_iters = 0
            
            for i in range(self.max_iter):
                n_iters += 1
                
                # Compute gradient using finite differences
                grad = self._compute_gradient(obj_wrapper, current_params)
                self.history['gradient_norm'].append(np.linalg.norm(grad))
                
                # Update parameters
                current_params = current_params - self.learning_rate * grad
                
                # Check for convergence
                if np.linalg.norm(grad) < self.tol:
                    converged = True
                    break
                
                # Evaluate objective at new point
                current_loss = obj_wrapper(current_params)
                
                if len(self.history['loss']) > 0 and abs(current_loss - self.history['loss'][-1]) < self.tol:
                    converged = True
                    break
            
            final_params = current_params
            
        elif self.method == 'adam':
            # Adam optimizer implementation
            current_params = initial_params.copy()
            m = np.zeros_like(current_params)  # First moment
            v = np.zeros_like(current_params)  # Second moment
            beta1, beta2 = 0.9, 0.999
            epsilon = 1e-8
            
            converged = False
            n_iters = 0
            
            for t in range(1, self.max_iter + 1):
                n_iters += 1
                
                # Compute gradient
                grad = self._compute_gradient(obj_wrapper, current_params)
                self.history['gradient_norm'].append(np.linalg.norm(grad))
                
                # Update biased first moment estimate
                m = beta1 * m + (1 - beta1) * grad
                # Update biased second raw moment estimate
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = m / (1 - beta1 ** t)
                # Compute bias-corrected second raw moment estimate
                v_hat = v / (1 - beta2 ** t)
                
                # Update parameters
                current_params = current_params - (self.learning_rate * m_hat) / (np.sqrt(v_hat) + epsilon)
                
                # Check for convergence
                if np.linalg.norm(grad) < self.tol:
                    converged = True
                    break
                
                # Evaluate objective at new point
                current_loss = obj_wrapper(current_params)
                
                if len(self.history['loss']) > 0 and abs(current_loss - self.history['loss'][-1]) < self.tol:
                    converged = True
                    break
            
            final_params = current_params
            
        else:
            raise ValueError(f"Unknown optimization method: {self.method}")
        
        # Compute final metrics
        final_loss = obj_wrapper(final_params)
        metrics = {
            'final_params': final_params.tolist(),
            'final_loss': float(final_loss),
            'iterations': n_iters,
            'converged': converged,
            'history': {
                'loss': self.history['loss'],
                'gradient_norm': self.history['gradient_norm']
            }
        }
        
        return final_params, metrics
    
    def _compute_gradient(self, objective_fn: Callable, params: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Compute gradient using finite differences."""
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            
            params_plus[i] += eps
            params_minus[i] -= eps
            
            f_plus = objective_fn(params_plus)
            f_minus = objective_fn(params_minus)
            
            grad[i] = (f_plus - f_minus) / (2 * eps)
        
        return grad


class GateSynthesisOptimizer:
    """
    Gate Synthesis Optimizer
    
    Optimizes quantum circuits by synthesizing more efficient gate sequences
    for specific quantum operations.
    """
    
    def __init__(self):
        """Initialize the gate synthesis optimizer."""
        # Pre-computed optimal decompositions for common operations
        self.optimal_decompositions = {}
    
    def optimize_single_qubit_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize single-qubit gates by combining them into RZ and RX rotations.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            Optimized circuit
        """
        # Use Qiskit's built-in optimization pass
        from qiskit.transpiler.passes import Optimize1qGates
        from qiskit.transpiler import PassManager
        
        pass_manager = PassManager([Optimize1qGates()])
        optimized_circuit = pass_manager.run(circuit)
        
        return optimized_circuit
    
    def optimize_two_qubit_gates(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize two-qubit gates by decomposing into optimal CX sequences.
        
        Args:
            circuit: Quantum circuit to optimize
            
        Returns:
            Optimized circuit
        """
        # For now, we'll use Qiskit's built-in optimization
        # In practice, this would implement custom gate synthesis algorithms
        from qiskit.transpiler.passes import CXCancellation, CommutationAnalysis, CommutativeCancellation
        
        from qiskit.transpiler import PassManager
        pass_manager = PassManager([
            CommutationAnalysis(),
            CommutativeCancellation(),
            CXCancellation()
        ])
        
        optimized_circuit = pass_manager.run(circuit)
        
        return optimized_circuit
    
    def optimize_circuit_topology(
        self,
        circuit: QuantumCircuit,
        coupling_map: List[Tuple[int, int]],
        initial_layout: Optional[List[int]] = None
    ) -> QuantumCircuit:
        """
        Optimize circuit for specific hardware topology.
        
        Args:
            circuit: Quantum circuit to optimize
            coupling_map: Hardware coupling map as list of (control, target) qubit pairs
            initial_layout: Initial qubit layout mapping (optional)
            
        Returns:
            Optimized circuit adapted to hardware topology
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for topology optimization")
        
        from qiskit.transpiler import CouplingMap
        from qiskit.transpiler.passes import SabreLayout, SabreSwap
        
        # Create coupling map object
        coupling = CouplingMap(coupling_map)
        
        # Create pass manager with layout and routing passes
        from qiskit.transpiler import PassManager
        pass_manager = PassManager([
            SabreLayout(coupling, seed_alt=42),
            SabreSwap(coupling, seed_alt=42)
        ])
        
        # Apply optimization
        optimized_circuit = pass_manager.run(circuit)
        
        return optimized_circuit


class QuantumFeatureMapOptimizer:
    """
    Quantum Feature Map Optimizer
    
    Optimizes quantum feature maps for better expressibility and trainability
    in quantum machine learning models.
    """
    
    def __init__(
        self,
        num_qubits: int,
        feature_dimension: int,
        reps: int = 2,
        entanglement: str = 'full'
    ):
        """
        Args:
            num_qubits: Number of qubits in the feature map
            feature_dimension: Dimension of the input features
            reps: Number of repetitions of the feature map
            entanglement: Type of entanglement ('full', 'linear', 'pairwise')
        """
        self.num_qubits = num_qubits
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.entanglement = entanglement
        
        # Store optimized feature map
        self.optimized_feature_map = None
    
    def optimize_feature_map(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        optimization_metric: str = 'expressibility'
    ) -> Tuple[QuantumCircuit, Dict[str, Any]]:
        """
        Optimize the quantum feature map based on the training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            optimization_metric: Metric to optimize ('expressibility', 'trainability', 'overlap')
            
        Returns:
            Optimized feature map and optimization metrics
        """
        if optimization_metric == 'expressibility':
            # Optimize for maximum expressibility (ability to represent diverse states)
            optimized_map = self._optimize_for_expressibility(X_train)
        elif optimization_metric == 'trainability':
            # Optimize for better trainability (avoid barren plateaus)
            optimized_map = self._optimize_for_trainability(X_train, y_train)
        elif optimization_metric == 'overlap':
            # Optimize for better class separation (maximize overlap difference)
            optimized_map = self._optimize_for_overlap(X_train, y_train)
        else:
            raise ValueError(f"Unknown optimization metric: {optimization_metric}")
        
        self.optimized_feature_map = optimized_map
        
        # Compute metrics
        metrics = self._compute_feature_map_metrics(optimized_map, X_train)
        
        return optimized_map, metrics
    
    def _optimize_for_expressibility(self, X_train: np.ndarray) -> QuantumCircuit:
        """Optimize feature map for maximum expressibility."""
        # Create a parameterized feature map with rotation angles that can be optimized
        from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
        
        # For expressibility, we'll use ZZFeatureMap which has good theoretical properties
        feature_map = ZZFeatureMap(
            feature_dimension=self.feature_dimension,
            reps=self.reps,
            entanglement=self.entanglement
        )
        
        # To optimize for expressibility, we might want to adjust the entanglement pattern
        # or add trainable parameters to the rotation angles
        # For now, we'll return the standard ZZFeatureMap
        return feature_map
    
    def _optimize_for_trainability(self, X_train: np.ndarray, y_train: np.ndarray) -> QuantumCircuit:
        """Optimize feature map to avoid barren plateaus."""
        # To improve trainability, we can reduce the number of parameters
        # or adjust the structure to have more regular gradients
        from qiskit.circuit.library import ZFeatureMap
        
        # Use a simpler feature map with fewer parameters
        feature_map = ZFeatureMap(
            feature_dimension=self.feature_dimension,
            reps=max(1, self.reps // 2)  # Reduce repetitions to improve trainability
        )
        
        return feature_map
    
    def _optimize_for_overlap(self, X_train: np.ndarray, y_train: np.ndarray) -> QuantumCircuit:
        """Optimize feature map to maximize class separation."""
        # For class separation, we might want to emphasize features that distinguish classes
        from qiskit.circuit.library import PauliFeatureMap
        
        # Use PauliFeatureMap which can be tailored for specific problems
        feature_map = PauliFeatureMap(
            feature_dimension=self.feature_dimension,
            reps=self.reps,
            paulis=['X', 'Y', 'Z', 'ZZ'],  # Include 2-qubit interactions
            entanglement=self.entanglement
        )
        
        return feature_map
    
    def _compute_feature_map_metrics(
        self,
        feature_map: QuantumCircuit,
        X_train: np.ndarray
    ) -> Dict[str, Any]:
        """Compute metrics for the feature map."""
        # For now, we'll compute basic metrics
        metrics = {
            'num_qubits': feature_map.num_qubits,
            'num_parameters': len(feature_map.parameters),
            'depth': feature_map.depth(),
            'size': feature_map.size(),
            'feature_dimension': self.feature_dimension,
            'reps': self.reps
        }
        
        return metrics


def optimize_quantum_circuit(
    circuit: QuantumCircuit,
    backend: Optional[Backend] = None,
    method: str = 'hybrid',
    objective_fn: Optional[Callable] = None,
    **kwargs
) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    """
    Optimize a quantum circuit using specified method.
    
    Args:
        circuit: Quantum circuit to optimize
        backend: Quantum backend for hardware-aware optimization
        method: Optimization method ('circuit', 'parameters', 'synthesis', 'feature_map', 'hybrid')
        objective_fn: Objective function for parameter optimization (optional)
        **kwargs: Additional arguments for the optimization method
        
    Returns:
        Optimized circuit and metrics
    """
    if method == 'circuit':
        optimizer = QuantumCircuitOptimizer(backend=backend, **kwargs)
        return optimizer.optimize_circuit(circuit, objective_fn=objective_fn, method='transpile')
    
    elif method == 'parameters':
        if objective_fn is None:
            raise ValueError("Objective function required for parameter optimization")
        
        # Extract parameters and optimize
        params = list(circuit.parameters)
        if not params:
            return circuit, {'message': 'No parameters to optimize'}
        
        param_optimizer = VariationalParameterOptimizer(**kwargs)
        opt_params, metrics = param_optimizer.optimize(circuit, objective_fn)
        
        # Bind optimized parameters
        optimized_circuit = circuit.bind_parameters(
            dict(zip(params, opt_params))
        )
        
        return optimized_circuit, metrics
    
    elif method == 'synthesis':
        synth_optimizer = GateSynthesisOptimizer()
        
        # Apply different synthesis optimizations
        circuit_1q = synth_optimizer.optimize_single_qubit_gates(circuit)
        optimized_circuit = synth_optimizer.optimize_two_qubit_gates(circuit_1q)
        
        # Compute metrics
        original_ops = circuit.count_ops()
        optimized_ops = optimized_circuit.count_ops()
        
        metrics = {
            'original_ops': dict(original_ops),
            'optimized_ops': dict(optimized_ops),
            'optimization_successful': True
        }
        
        return optimized_circuit, metrics
    
    elif method == 'feature_map':
        # This method assumes the circuit is a feature map
        # We'll need feature dimension and other info from kwargs
        if 'num_qubits' not in kwargs or 'feature_dimension' not in kwargs:
            raise ValueError("num_qubits and feature_dimension required for feature_map optimization")
        
        fm_optimizer = QuantumFeatureMapOptimizer(**kwargs)
        
        # For this optimization we need training data
        if 'X_train' in kwargs and 'y_train' in kwargs:
            X_train = kwargs['X_train']
            y_train = kwargs['y_train']
            metric = kwargs.get('optimization_metric', 'expressibility')
            return fm_optimizer.optimize_feature_map(X_train, y_train, metric)
        else:
            # Create a default feature map
            from qiskit.circuit.library import ZZFeatureMap
            default_fm = ZZFeatureMap(
                feature_dimension=kwargs['feature_dimension'],
                reps=kwargs.get('reps', 2),
                entanglement=kwargs.get('entanglement', 'full')
            )
            return default_fm, {'message': 'Default feature map created'}
    
    elif method == 'hybrid':
        # Apply multiple optimization techniques
        # Start with transpilation optimization
        circuit_opt1 = transpile(circuit, optimization_level=kwargs.get('optimization_level', 2))
        
        # Apply gate synthesis optimization
        synth_optimizer = GateSynthesisOptimizer()
        circuit_opt2 = synth_optimizer.optimize_single_qubit_gates(circuit_opt1)
        circuit_opt2 = synth_optimizer.optimize_two_qubit_gates(circuit_opt2)
        
        # If parameters exist and objective function provided, optimize them
        if objective_fn is not None and len(circuit_opt2.parameters) > 0:
            param_optimizer = VariationalParameterOptimizer(**kwargs)
            params, _ = param_optimizer.optimize(circuit_opt2, objective_fn)
            final_circuit = circuit_opt2.bind_parameters(
                dict(zip(circuit_opt2.parameters, params))
            )
        else:
            final_circuit = circuit_opt2
        
        # Compute final metrics
        original_depth = circuit.depth()
        optimized_depth = final_circuit.depth()
        original_size = circuit.size()
        optimized_size = final_circuit.size()
        
        metrics = {
            'original_depth': original_depth,
            'optimized_depth': optimized_depth,
            'depth_reduction': original_depth - optimized_depth,
            'original_size': original_size,
            'optimized_size': optimized_size,
            'size_reduction': original_size - optimized_size,
            'optimization_successful': True
        }
        
        return final_circuit, metrics
    
    else:
        raise ValueError(f"Unknown optimization method: {method}")


# Example usage and testing
if __name__ == "__main__":
    if QISKIT_AVAILABLE:
        print("Testing Quantum Circuit Optimization Techniques...")
        
        # Create a parameterized test circuit
        qc = QuantumCircuit(3, 3)
        theta = Parameter('θ')
        phi = Parameter('φ')
        
        qc.h(0)
        qc.ry(theta, 0)
        qc.cx(0, 1)
        qc.rz(phi, 1)
        qc.cx(1, 2)
        qc.rx(theta, 2)
        qc.measure_all()
        
        print(f"Original circuit - Depth: {qc.depth()}, Size: {qc.size()}")
        
        # Mock objective function
        def mock_objective(circuit):
            # Simulate an objective that prefers shallower circuits
            return circuit.depth() * 0.1 + len(circuit.parameters) * 0.05
        
        # Test circuit optimization
        print("\n1. Testing Circuit Optimization:")
        try:
            opt_circuit, metrics = optimize_quantum_circuit(
                qc, method='circuit', optimization_level=2
            )
            print(f"   ✓ Circuit optimization completed")
            print(f"   Original: Depth={qc.depth()}, Size={qc.size()}")
            print(f"   Optimized: Depth={opt_circuit.depth()}, Size={opt_circuit.size()}")
            print(f"   Metrics: {metrics}")
        except Exception as e:
            print(f"   ✗ Circuit optimization failed: {e}")
        
        # Test parameter optimization
        print("\n2. Testing Parameter Optimization:")
        try:
            opt_circuit2, metrics2 = optimize_quantum_circuit(
                qc, method='parameters', objective_fn=mock_objective,
                max_iter=20
            )
            print(f"   ✓ Parameter optimization completed")
            print(f"   Final loss: {metrics2['final_loss']:.4f}")
            print(f"   Iterations: {metrics2['iterations']}")
        except Exception as e:
            print(f"   ✗ Parameter optimization failed: {e}")
        
        # Test synthesis optimization
        print("\n3. Testing Gate Synthesis Optimization:")
        try:
            opt_circuit3, metrics3 = optimize_quantum_circuit(
                qc, method='synthesis'
            )
            print(f"   ✓ Gate synthesis optimization completed")
            print(f"   Original ops: {dict(qc.count_ops())}")
            print(f"   Optimized ops: {metrics3['optimized_ops']}")
        except Exception as e:
            print(f"   ✗ Gate synthesis optimization failed: {e}")
        
        # Test hybrid optimization
        print("\n4. Testing Hybrid Optimization:")
        try:
            opt_circuit4, metrics4 = optimize_quantum_circuit(
                qc, method='hybrid', objective_fn=mock_objective,
                optimization_level=2, max_iter=10
            )
            print(f"   ✓ Hybrid optimization completed")
            print(f"   Original: Depth={qc.depth()}, Size={qc.size()}")
            print(f"   Optimized: Depth={opt_circuit4.depth()}, Size={opt_circuit4.size()}")
        except Exception as e:
            print(f"   ✗ Hybrid optimization failed: {e}")
        
        print("\nQuantum Circuit Optimization testing completed!")
    else:
        print("Qiskit not available. Circuit optimization techniques require Qiskit.")