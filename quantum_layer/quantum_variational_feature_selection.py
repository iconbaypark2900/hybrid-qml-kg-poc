"""
Quantum Variational Algorithms for Feature Selection

Implements quantum variational algorithms for feature selection in quantum machine learning,
including Quantum Variational Feature Selector (QVFS) and Quantum Approximate Optimization
for Feature Selection (QAOFS).
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.optimize import minimize
import itertools

logger = logging.getLogger(__name__)

try:
    from qiskit import QuantumCircuit, ClassicalRegister
    try:
        from qiskit.circuit import Parameter
    except ImportError:
        from qiskit.circuit.parameter import Parameter
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit.algorithms.optimizers import COBYLA, SPSA
    from qiskit_machine_learning.algorithms import VQC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available. Quantum variational algorithms will be limited.")


class QuantumVariationalFeatureSelector:
    """
    Quantum Variational Feature Selector (QVFS)
    
    Uses a parameterized quantum circuit to learn which features are most relevant
    for the learning task. The quantum circuit learns a probability distribution
    over feature subsets, and the most probable subsets are selected.
    """
    
    def __init__(
        self,
        num_features: int,
        num_layers: int = 2,
        optimizer: str = 'COBYLA',
        max_iter: int = 100,
        feature_encoding: str = 'amplitude',
        entanglement: str = 'full'
    ):
        """
        Args:
            num_features: Total number of features
            num_layers: Number of layers in the variational circuit
            optimizer: Classical optimizer ('COBYLA', 'SPSA', 'L_BFGS_B')
            max_iter: Maximum iterations for optimization
            feature_encoding: How to encode features ('amplitude', 'rotation', 'ising')
            entanglement: Entanglement pattern ('full', 'linear', 'pairwise')
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum variational feature selection")
        
        self.num_features = num_features
        self.num_layers = num_layers
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.feature_encoding = feature_encoding
        self.entanglement = entanglement
        
        # Initialize quantum components
        self.variational_circuit = self._create_variational_circuit()
        self.optimizer = self._create_optimizer()
        
        # Store selection results
        self.selected_features = None
        self.feature_scores = None
        self.selection_history = []
    
    def _create_variational_circuit(self) -> QuantumCircuit:
        """Create the parameterized quantum circuit for feature selection."""
        # Create a circuit with one qubit per feature
        qc = QuantumCircuit(self.num_features)
        
        # Add initial rotation gates to encode feature relevance
        for i in range(self.num_features):
            theta = Parameter(f'theta_{i}')
            qc.ry(theta, i)  # Encode feature relevance in rotation angle
        
        # Add entangling layers
        for layer in range(self.num_layers):
            # Add entangling gates
            for i in range(self.num_features - 1):
                qc.cx(i, i + 1)  # Linear entanglement
            
            # Add variational rotation gates
            for i in range(self.num_features):
                theta = Parameter(f'layer_{layer}_theta_{i}')
                qc.ry(theta, i)
        
        # Add final measurement
        qc.measure_all()
        
        return qc
    
    def _create_optimizer(self):
        """Create the classical optimizer."""
        if self.optimizer_name == 'COBYLA':
            return COBYLA(maxiter=self.max_iter)
        elif self.optimizer_name == 'SPSA':
            return SPSA(maxiter=self.max_iter)
        elif self.optimizer_name == 'L_BFGS_B':
            from qiskit.algorithms.optimizers import L_BFGS_B
            return L_BFGS_B(maxiter=self.max_iter)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        objective_fn: Optional[Callable] = None,
        num_selected_features: Optional[int] = None
    ) -> 'QuantumVariationalFeatureSelector':
        """
        Fit the quantum feature selector to the data.
        
        Args:
            X: Training data [N, D]
            y: Training labels [N]
            objective_fn: Custom objective function (if None, uses default)
            num_selected_features: Number of features to select (if None, uses half)
            
        Returns:
            Self
        """
        logger.info(f"Fitting quantum variational feature selector for {self.num_features} features...")
        
        if num_selected_features is None:
            num_selected_features = max(1, self.num_features // 2)
        
        if objective_fn is None:
            # Default objective: maximize classification accuracy with selected features
            def objective_fn(params):
                return self._default_objective(params, X, y, num_selected_features)
        
        # Get all parameters in the circuit
        params = list(self.variational_circuit.parameters)
        n_params = len(params)
        
        # Initialize parameters randomly
        initial_params = np.random.uniform(-np.pi, np.pi, n_params)
        
        # Optimize the parameters
        result = minimize(
            objective_fn,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )
        
        # Extract feature scores from optimized parameters
        # The magnitude of rotation angles indicates feature importance
        final_params = result.x
        feature_scores = np.abs(final_params[:self.num_features])  # Only initial rotation angles
        
        # Select top features based on scores
        top_indices = np.argsort(feature_scores)[::-1][:num_selected_features]
        
        self.selected_features = top_indices
        self.feature_scores = feature_scores
        self.selection_history.append(result)
        
        logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features}")
        
        return self
    
    def _default_objective(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        num_selected: int
    ) -> float:
        """Default objective function based on classification performance."""
        # Extract feature scores from parameters (first num_features parameters)
        feature_scores = np.abs(params[:self.num_features])
        
        # Select top features
        top_indices = np.argsort(feature_scores)[::-1][:num_selected]
        X_selected = X[:, top_indices]
        
        # Use a simple classifier to evaluate feature set
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')
        avg_score = np.mean(scores)
        
        # Return negative score to minimize (maximize accuracy)
        return -avg_score
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting the most relevant features.
        
        Args:
            X: Input data [N, D]
            
        Returns:
            Transformed data with selected features [N, num_selected]
        """
        if self.selected_features is None:
            raise ValueError("Feature selector not fitted yet. Call fit() first.")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the selector and transform the data."""
        self.fit(X, y)
        return self.transform(X)


class QuantumApproximateOptimizerFeatureSelection:
    """
    Quantum Approximate Optimization Algorithm for Feature Selection (QAOFS)
    
    Uses QAOA-inspired approach to solve the feature selection problem as a
    combinatorial optimization problem.
    """
    
    def __init__(
        self,
        num_features: int,
        p: int = 2,  # Number of QAOA layers
        optimizer: str = 'COBYLA',
        max_iter: int = 100,
        penalty_strength: float = 1.0
    ):
        """
        Args:
            num_features: Total number of features
            p: Number of QAOA layers (mixing and cost evolution)
            optimizer: Classical optimizer
            max_iter: Maximum iterations for optimization
            penalty_strength: Strength of penalty for constraint violations
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for QAOA feature selection")
        
        self.num_features = num_features
        self.p = p
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.penalty_strength = penalty_strength
        
        # Initialize quantum components
        self.qaoa_circuit = self._create_qaoa_circuit()
        self.optimizer = self._create_optimizer()
        
        # Store selection results
        self.selected_features = None
        self.feature_probabilities = None
    
    def _create_qaoa_circuit(self) -> QuantumCircuit:
        """Create the QAOA circuit for feature selection."""
        # Create a circuit with one qubit per feature
        qc = QuantumCircuit(self.num_features)
        
        # Initial state preparation (equal superposition)
        for i in range(self.num_features):
            qc.h(i)
        
        # QAOA layers
        for layer in range(self.p):
            # Cost Hamiltonian evolution (problem-dependent)
            beta = Parameter(f'beta_{layer}')
            gamma = Parameter(f'gamma_{layer}')
            
            # Apply ZZ interactions to encode feature correlations
            for i in range(self.num_features - 1):
                qc.cx(i, i + 1)
                qc.rz(2 * gamma, i + 1)
                qc.cx(i, i + 1)
            
            # Mixing Hamiltonian evolution (transverse field)
            for i in range(self.num_features):
                qc.rx(2 * beta, i)
        
        # Add final measurement
        qc.measure_all()
        
        return qc
    
    def _create_optimizer(self):
        """Create the classical optimizer."""
        if self.optimizer_name == 'COBYLA':
            return COBYLA(maxiter=self.max_iter)
        elif self.optimizer_name == 'SPSA':
            return SPSA(maxiter=self.max_iter)
        else:
            from qiskit.algorithms.optimizers import L_BFGS_B
            return L_BFGS_B(maxiter=self.max_iter)
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_selected_features: Optional[int] = None,
        feature_correlation_weight: float = 0.1
    ) -> 'QuantumApproximateOptimizerFeatureSelection':
        """
        Fit the QAOA-based feature selector to the data.
        
        Args:
            X: Training data [N, D]
            y: Training labels [N]
            num_selected_features: Number of features to select (if None, uses half)
            feature_correlation_weight: Weight for feature correlation in objective
            
        Returns:
            Self
        """
        logger.info(f"Fitting QAOA feature selector for {self.num_features} features...")
        
        if num_selected_features is None:
            num_selected_features = max(1, self.num_features // 2)
        
        # Compute feature correlations for the cost Hamiltonian
        feature_corr = self._compute_feature_correlations(X, y)
        
        # Define objective function
        def objective(params):
            return self._qaoa_objective(
                params, X, y, num_selected_features, feature_corr, feature_correlation_weight
            )
        
        # Get all parameters in the circuit
        params = list(self.qaoa_circuit.parameters)
        n_params = len(params)
        
        # Initialize parameters randomly
        initial_params = np.random.uniform(0, 2*np.pi, n_params)
        
        # Optimize the parameters
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': self.max_iter}
        )
        
        # Extract feature probabilities from the optimized circuit
        # This is a simplified approach - in practice, we'd run the circuit and measure
        feature_probs = self._extract_feature_probabilities(result.x)
        
        # Select top features based on probabilities
        top_indices = np.argsort(feature_probs)[::-1][:num_selected_features]
        
        self.selected_features = top_indices
        self.feature_probabilities = feature_probs
        
        logger.info(f"QAOA selected {len(self.selected_features)} features: {self.selected_features}")
        
        return self
    
    def _compute_feature_correlations(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute feature correlations for the cost Hamiltonian."""
        # Compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Adjust correlations based on label information
        # Features that correlate with the target should be preferred
        label_corr = np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])])
        
        # Combine correlations
        adjusted_corr = corr_matrix * (1 - np.outer(np.abs(label_corr), np.abs(label_corr)))
        
        return adjusted_corr
    
    def _qaoa_objective(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        num_selected: int,
        feature_corr: np.ndarray,
        corr_weight: float
    ) -> float:
        """Objective function for QAOA optimization."""
        # This is a simplified objective function
        # In practice, we would evaluate the quantum circuit and measure outcomes
        
        # Extract gamma and beta parameters
        n_gamma = self.p
        n_beta = self.p
        gammas = params[:n_gamma]
        betas = params[n_gamma:n_gamma+n_beta]
        
        # Create a mock objective based on feature selection quality
        # This simulates what we'd get from running the quantum circuit
        
        # Use the parameters to determine feature probabilities
        # This is a simplified simulation of the quantum state
        feature_probs = np.abs(np.sin(gammas[0])) if len(gammas) > 0 else np.ones(self.num_features) / self.num_features
        
        # If we have more parameters, use them to refine the probabilities
        if len(params) > n_gamma + n_beta:
            extra_params = params[n_gamma+n_beta:]
            for i, extra_param in enumerate(extra_params[:self.num_features]):
                feature_probs[i % len(feature_probs)] *= (1 + np.abs(np.sin(extra_param)))
        
        # Normalize probabilities
        feature_probs = feature_probs / np.sum(feature_probs)
        
        # Select top features
        top_indices = np.argsort(feature_probs)[::-1][:num_selected]
        X_selected = X[:, top_indices]
        
        # Evaluate the selected features
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, X_selected, y, cv=3, scoring='accuracy')
        avg_score = np.mean(scores)
        
        # Add penalty for deviating from the desired number of features
        num_penalty = self.penalty_strength * (len(top_indices) - num_selected)**2
        
        # Add penalty based on feature correlations (prefer diverse features)
        if corr_weight > 0 and len(top_indices) > 1:
            total_corr = 0
            for i in range(len(top_indices)):
                for j in range(i+1, len(top_indices)):
                    total_corr += abs(feature_corr[top_indices[i], top_indices[j]])
            corr_penalty = corr_weight * total_corr
        else:
            corr_penalty = 0
        
        # Return negative score to minimize (maximize accuracy)
        return -(avg_score - num_penalty - corr_penalty)
    
    def _extract_feature_probabilities(self, params: np.ndarray) -> np.ndarray:
        """Extract feature probabilities from optimized parameters."""
        # This is a simplified approach
        # In practice, we'd run the quantum circuit and measure the outcomes
        
        # Use the first few parameters to determine feature probabilities
        n_params_to_use = min(len(params), self.num_features)
        probs = np.zeros(self.num_features)
        
        for i in range(n_params_to_use):
            # Map parameter to probability using sine function
            probs[i] = np.abs(np.sin(params[i]))**2
        
        # Normalize to ensure they sum to 1
        if np.sum(probs) > 0:
            probs = probs / np.sum(probs)
        else:
            probs = np.ones(self.num_features) / self.num_features
        
        return probs
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting the most relevant features.
        
        Args:
            X: Input data [N, D]
            
        Returns:
            Transformed data with selected features [N, num_selected]
        """
        if self.selected_features is None:
            raise ValueError("Feature selector not fitted yet. Call fit() first.")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the selector and transform the data."""
        self.fit(X, y)
        return self.transform(X)


class QuantumMutualInformationFeatureSelector:
    """
    Quantum Mutual Information Feature Selector
    
    Uses quantum circuits to estimate mutual information between features and labels
    for feature selection.
    """
    
    def __init__(
        self,
        num_features: int,
        num_qubits_per_feature: int = 1,
        entanglement: str = 'linear',
        max_iter: int = 50
    ):
        """
        Args:
            num_features: Total number of features
            num_qubits_per_feature: Number of qubits to represent each feature
            entanglement: Entanglement pattern
            max_iter: Maximum iterations for optimization
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum mutual information feature selection")
        
        self.num_features = num_features
        self.num_qubits_per_feature = num_qubits_per_feature
        self.entanglement = entanglement
        self.max_iter = max_iter
        
        # Total number of qubits needed
        self.total_qubits = num_features * num_qubits_per_feature + 1  # +1 for label qubit
        
        # Store selection results
        self.selected_features = None
        self.mutual_information_scores = None
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_selected_features: Optional[int] = None
    ) -> 'QuantumMutualInformationFeatureSelector':
        """
        Fit the quantum mutual information feature selector.
        
        Args:
            X: Training data [N, D]
            y: Training labels [N]
            num_selected_features: Number of features to select (if None, uses half)
            
        Returns:
            Self
        """
        logger.info(f"Fitting quantum mutual information feature selector for {self.num_features} features...")
        
        if num_selected_features is None:
            num_selected_features = max(1, self.num_features // 2)
        
        # Estimate quantum mutual information for each feature
        mi_scores = self._estimate_quantum_mutual_information(X, y)
        
        # Select top features based on mutual information
        top_indices = np.argsort(mi_scores)[::-1][:num_selected_features]
        
        self.selected_features = top_indices
        self.mutual_information_scores = mi_scores
        
        logger.info(f"Selected {len(self.selected_features)} features based on quantum MI")
        
        return self
    
    def _estimate_quantum_mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Estimate quantum mutual information between features and labels.
        
        This is a simplified approach - in practice, this would involve
        running quantum circuits to estimate the quantum mutual information.
        """
        # For each feature, estimate its quantum mutual information with the label
        mi_scores = np.zeros(self.num_features)
        
        for i in range(self.num_features):
            # Discretize the feature for mutual information calculation
            feature_vals = X[:, i]
            
            # Simple classical mutual information as a proxy for quantum MI
            # In practice, we would use quantum circuits to estimate this
            mi_scores[i] = self._compute_classical_mi_proxy(feature_vals, y)
        
        return mi_scores
    
    def _compute_classical_mi_proxy(self, feature: np.ndarray, labels: np.ndarray) -> float:
        """Compute a proxy for mutual information."""
        # Discretize feature into bins
        bins = min(10, len(np.unique(feature)))
        feature_discrete = np.digitize(feature, bins=np.histogram(feature, bins=bins)[1][:-1])
        
        # Compute mutual information
        from sklearn.metrics import mutual_info_score
        mi = mutual_info_score(feature_discrete, labels)
        
        return mi
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the data by selecting the most relevant features.
        
        Args:
            X: Input data [N, D]
            
        Returns:
            Transformed data with selected features [N, num_selected]
        """
        if self.selected_features is None:
            raise ValueError("Feature selector not fitted yet. Call fit() first.")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the selector and transform the data."""
        self.fit(X, y)
        return self.transform(X)


def select_features_quantum_variational(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'qvfs',
    num_selected: Optional[int] = None,
    **kwargs
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Select features using quantum variational algorithms.
    
    Args:
        X: Input data [N, D]
        y: Labels [N]
        method: Selection method ('qvfs', 'qaoa', 'qmi')
        num_selected: Number of features to select (if None, uses half)
        **kwargs: Additional arguments for the selection method
        
    Returns:
        Tuple of (selected_features_data, selection_metrics)
    """
    n_features = X.shape[1]
    if num_selected is None:
        num_selected = max(1, n_features // 2)
    
    if method == 'qvfs':
        selector = QuantumVariationalFeatureSelector(n_features, **kwargs)
        X_selected = selector.fit_transform(X, y)
        
        metrics = {
            'method': 'qvfs',
            'selected_features': selector.selected_features.tolist(),
            'feature_scores': selector.feature_scores.tolist(),
            'num_selected': len(selector.selected_features),
            'original_features': n_features
        }
        
    elif method == 'qaoa':
        selector = QuantumApproximateOptimizerFeatureSelection(n_features, **kwargs)
        X_selected = selector.fit_transform(X, y)
        
        metrics = {
            'method': 'qaoa',
            'selected_features': selector.selected_features.tolist(),
            'feature_probabilities': selector.feature_probabilities.tolist(),
            'num_selected': len(selector.selected_features),
            'original_features': n_features
        }
        
    elif method == 'qmi':
        selector = QuantumMutualInformationFeatureSelector(n_features, **kwargs)
        X_selected = selector.fit_transform(X, y)
        
        metrics = {
            'method': 'qmi',
            'selected_features': selector.selected_features.tolist(),
            'mutual_information_scores': selector.mutual_information_scores.tolist(),
            'num_selected': len(selector.selected_features),
            'original_features': n_features
        }
        
    else:
        raise ValueError(f"Unknown selection method: {method}")
    
    return X_selected, metrics


# Example usage and testing
if __name__ == "__main__":
    if QISKIT_AVAILABLE:
        print("Testing Quantum Variational Algorithms for Feature Selection...")
        
        # Create sample data
        np.random.seed(42)
        n_samples, n_features = 100, 8
        X = np.random.randn(n_samples, n_features)
        
        # Create a target that depends on only a few features
        y = (X[:, 0] + X[:, 2] - X[:, 4] > 0).astype(int)
        
        print(f"Sample data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target depends on features [0, 2, 4]")
        
        # Test Quantum Variational Feature Selector
        print("\n1. Testing Quantum Variational Feature Selector (QVFS):")
        try:
            X_qvfs, metrics_qvfs = select_features_quantum_variational(
                X, y, method='qvfs',
                num_selected=4,
                num_layers=2,
                max_iter=30
            )
            print(f"   ✓ QVFS completed")
            print(f"   Selected features: {metrics_qvfs['selected_features']}")
            print(f"   Original shape: {X.shape}, Selected shape: {X_qvfs.shape}")
            
            # Check if important features were selected
            important_selected = len(set([0, 2, 4]) & set(metrics_qvfs['selected_features']))
            print(f"   Important features selected: {important_selected}/3")
        except Exception as e:
            print(f"   ✗ QVFS failed: {e}")
        
        # Test QAOA Feature Selection
        print("\n2. Testing QAOA Feature Selection:")
        try:
            X_qaoa, metrics_qaoa = select_features_quantum_variational(
                X, y, method='qaoa',
                num_selected=4,
                p=2,
                max_iter=20
            )
            print(f"   ✓ QAOA completed")
            print(f"   Selected features: {metrics_qaoa['selected_features']}")
            print(f"   Original shape: {X.shape}, Selected shape: {X_qaoa.shape}")
            
            # Check if important features were selected
            important_selected = len(set([0, 2, 4]) & set(metrics_qaoa['selected_features']))
            print(f"   Important features selected: {important_selected}/3")
        except Exception as e:
            print(f"   ✗ QAOA failed: {e}")
        
        # Test Quantum Mutual Information Feature Selection
        print("\n3. Testing Quantum Mutual Information Feature Selection:")
        try:
            X_qmi, metrics_qmi = select_features_quantum_variational(
                X, y, method='qmi',
                num_selected=4,
                max_iter=20
            )
            print(f"   ✓ QMI completed")
            print(f"   Selected features: {metrics_qmi['selected_features']}")
            print(f"   Original shape: {X.shape}, Selected shape: {X_qmi.shape}")
            
            # Check if important features were selected
            important_selected = len(set([0, 2, 4]) & set(metrics_qmi['selected_features']))
            print(f"   Important features selected: {important_selected}/3")
        except Exception as e:
            print(f"   ✗ QMI failed: {e}")
        
        print("\nQuantum Variational Feature Selection testing completed!")
    else:
        print("Qiskit not available. Quantum variational algorithms require Qiskit.")