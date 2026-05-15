# quantum_layer/advanced_error_mitigation.py

"""
Advanced Error Mitigation Techniques Inspired by Google Quantum AI
Implements Pauli Path-inspired ZNE and adaptive noise modeling.

Per preregistration §5.7 / §8.6, three zero-noise extrapolations are
computed at noise scales [1×, 3×, 5×] and reported separately:

  * Analytical Pauli-path fit (the primary, implemented in PauliPathZNE)
  * Linear polyfit (linear_zero_noise_extrapolation)
  * Richardson extrapolation (richardson_zero_noise_extrapolation)

The three values are emitted as separate observable columns by
quantum_layer/qml_trainer.py for downstream §8.6 sensitivity analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from scipy.optimize import minimize, differential_evolution
from scipy.special import erf
import logging

logger = logging.getLogger(__name__)


def linear_zero_noise_extrapolation(
    noise_scales: np.ndarray, measurements: np.ndarray
) -> float:
    """Zero-noise value from a degree-1 polynomial fit.

    Solves ``y = a*lambda + b`` by ordinary least squares and returns
    the intercept ``b`` (which is the extrapolation to ``lambda=0``).
    Standard Richardson would handle a higher-order fit; this helper
    exists separately so reviewers see the *first-order* extrapolation
    explicitly per preregistration §5.7 step 4.

    Args:
        noise_scales: 1-D array of noise factors, length n >= 2.
        measurements: 1-D array of measured values at each scale.

    Returns:
        The zero-noise intercept (a float).
    """
    scales = np.asarray(noise_scales, dtype=float).ravel()
    meas = np.asarray(measurements, dtype=float).ravel()
    if scales.shape != meas.shape:
        raise ValueError(
            f"linear ZNE: scales {scales.shape} != measurements {meas.shape}"
        )
    if scales.shape[0] < 2:
        raise ValueError("linear ZNE: need at least 2 noise scales")
    # polyfit returns highest-degree coeff first; for deg=1 -> [slope, intercept]
    slope, intercept = np.polyfit(scales, meas, deg=1)
    return float(intercept)


def richardson_zero_noise_extrapolation(
    noise_scales: np.ndarray, measurements: np.ndarray
) -> float:
    """Zero-noise value via classical Richardson extrapolation.

    For ``n`` noise scales, computes coefficients ``c_i`` such that
    ``sum(c_i * lambda_i^k) = 1 if k == 0 else 0`` for ``k = 0..n-1``,
    then returns ``sum(c_i * y_i)``. This cancels error terms up to
    order ``n-1`` in ``lambda`` and matches the canonical form used
    in preregistration §5.7 step 5.

    With three scales ``[1, 3, 5]`` (the preregistered defaults), the
    closed-form coefficients are ``c = (1.875, -1.25, 0.375)`` so that
    ``f(0) ≈ 1.875 f(1) − 1.25 f(3) + 0.375 f(5)``.

    Args:
        noise_scales: 1-D array of noise factors, length n >= 2.
        measurements: 1-D array of measured values at each scale.

    Returns:
        The zero-noise extrapolated value (a float).

    Raises:
        ValueError: shape mismatch or fewer than 2 scales.
        np.linalg.LinAlgError: singular Vandermonde (e.g. duplicate scales).
    """
    scales = np.asarray(noise_scales, dtype=float).ravel()
    meas = np.asarray(measurements, dtype=float).ravel()
    if scales.shape != meas.shape:
        raise ValueError(
            f"Richardson ZNE: scales {scales.shape} != measurements {meas.shape}"
        )
    n = scales.shape[0]
    if n < 2:
        raise ValueError("Richardson ZNE: need at least 2 noise scales")
    # Vandermonde-like system: A[k, i] = lambda_i ** k for k = 0..n-1.
    A = np.vstack([scales ** k for k in range(n)])
    rhs = np.zeros(n)
    rhs[0] = 1.0
    coeffs = np.linalg.solve(A, rhs)
    return float(np.dot(coeffs, meas))


def all_zero_noise_extrapolations(
    pauli_path_zne: "PauliPathZNE",
    noise_scales: np.ndarray,
    measurements: np.ndarray,
    measurement_errors: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute all three pre-registered ZNE methods on the same data.

    Returns a dict with keys ``analytical``, ``linear``, ``richardson``
    plus the analytical ``params`` sub-dict. Used by qml_trainer.py to
    emit per-method observable columns (preregistration §8.6).

    Each method is wrapped in try/except so a single failing method
    does not block the others; failed methods produce ``float("nan")``.
    """
    out: Dict[str, float] = {"analytical": float("nan"), "linear": float("nan"),
                             "richardson": float("nan")}
    params: Dict[str, float] = {}
    try:
        c0_analytical, params = pauli_path_zne.fit_noise_model(
            noise_scales, measurements, measurement_errors
        )
        out["analytical"] = float(c0_analytical)
    except Exception as e:
        logger.warning(f"analytical ZNE failed: {e}")
    try:
        out["linear"] = linear_zero_noise_extrapolation(noise_scales, measurements)
    except Exception as e:
        logger.warning(f"linear ZNE failed: {e}")
    try:
        out["richardson"] = richardson_zero_noise_extrapolation(noise_scales, measurements)
    except Exception as e:
        logger.warning(f"Richardson ZNE failed: {e}")
    out["params"] = params  # type: ignore[assignment]
    return out


class PauliPathZNE:
    """
    Pauli Path Zero-Noise Extrapolation
    
    Based on the insight that noisy expectation values are Laplace transforms
    of the Hamming weight distribution in the Pauli path picture.
    
    Reference: Google Quantum AI, "Quantum computation of molecular geometry..."
    """
    
    def __init__(self, use_bayesian_priors: bool = True):
        self.use_bayesian_priors = use_bayesian_priors
        self.prior_params = None
        
    def model_function(self, lambda_noise: float, C0: float, H_bar: float, sigma: float) -> float:
        """
        Analytical model for noisy expectation value.
        
        Args:
            lambda_noise: Noise scaling factor (1.0 = physical noise)
            C0: Zero-noise extrapolated value
            H_bar: Mean Hamming weight (noise accumulation)
            sigma: Standard deviation of Hamming weight
            
        Returns:
            C(λ): Expected noisy measurement value
        """
        beta = sigma / H_bar if H_bar > 0 else 0.1
        
        # Error function correction for finite H ≥ 0 constraint
        erf_numerator = H_bar - lambda_noise * sigma**2
        erf_correction_num = 1 + erf(erf_numerator / sigma) if sigma > 0 else 1
        erf_correction_den = 1 + erf(H_bar / sigma) if sigma > 0 else 1
        
        # Full model with quadratic and erf terms
        result = C0 * (erf_correction_num / erf_correction_den) * \
                 np.exp(-lambda_noise * H_bar) * \
                 np.exp(lambda_noise**2 * sigma**2 / 2)
        
        return result
    
    def fit_noise_model(
        self, 
        noise_scales: np.ndarray, 
        measurements: np.ndarray,
        measurement_errors: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Fit the Pauli Path model to noisy measurement data.
        
        Args:
            noise_scales: Array of noise scaling factors (e.g., [1.0, 1.5, 2.0, 2.5, 3.0])
            measurements: Noisy measurement values at each scale
            measurement_errors: Optional standard errors for weighted fitting
            
        Returns:
            C0: Zero-noise extrapolated value
            params: Dict with H_bar, sigma, beta
        """
        if measurement_errors is None:
            measurement_errors = np.ones_like(measurements) * 0.01
        
        # Objective function: weighted least squares
        def objective(params):
            C0, H_bar, sigma = params
            if H_bar <= 0 or sigma <= 0 or C0 < 0 or C0 > 1:
                return 1e10  # Penalty for invalid params
            
            predicted = np.array([
                self.model_function(lam, C0, H_bar, sigma) 
                for lam in noise_scales
            ])
            residuals = (measurements - predicted) / measurement_errors
            return np.sum(residuals**2)
        
        # Initial guess or use Bayesian prior
        if self.use_bayesian_priors and self.prior_params is not None:
            x0 = [
                self.prior_params['C0'],
                self.prior_params['H_bar'],
                self.prior_params['sigma']
            ]
        else:
            # Heuristic initial guess
            x0 = [
                measurements[0],  # C0 ~ first measurement
                2.0,              # H_bar ~ typical noise accumulation
                0.5               # sigma ~ spread
            ]
        
        # Bounds: C0 in [0, 1], H_bar > 0, sigma > 0
        bounds = [(0, 1), (0.1, 10), (0.01, 5)]
        
        # Global optimization followed by local refinement
        result_global = differential_evolution(
            objective, bounds, seed=42, maxiter=500, tol=1e-6
        )
        result_local = minimize(
            objective, result_global.x, method='L-BFGS-B', bounds=bounds
        )
        
        C0_fit, H_bar_fit, sigma_fit = result_local.x
        beta_fit = sigma_fit / H_bar_fit
        
        params = {
            'C0': float(C0_fit),
            'H_bar': float(H_bar_fit),
            'sigma': float(sigma_fit),
            'beta': float(beta_fit),
            'fit_error': float(result_local.fun)
        }
        
        # Store as prior for next fitting (Bayesian update)
        if self.use_bayesian_priors:
            self.prior_params = params
        
        logger.info(f"Fitted Pauli Path ZNE: C0={C0_fit:.4f}, H̄={H_bar_fit:.3f}, σ={sigma_fit:.3f}, β={beta_fit:.3f}")
        
        return C0_fit, params
    
    def extrapolate_zero_noise(
        self,
        circuit_executor: Callable,
        base_noise_scale: float = 1.0,
        num_scales: int = 5,
        max_scale: float = 3.0
    ) -> Tuple[float, Dict]:
        """
        Execute circuit at multiple noise scales and extrapolate to zero noise.
        
        Args:
            circuit_executor: Function that runs circuit and returns expectation value
                             Must accept noise_scale parameter
            base_noise_scale: Physical noise level (typically 1.0)
            num_scales: Number of noise scaling points
            max_scale: Maximum noise amplification
            
        Returns:
            mitigated_value: Zero-noise extrapolated expectation value
            diagnostics: Fitting diagnostics and intermediate results
        """
        # Generate noise scales (including base)
        noise_scales = np.linspace(base_noise_scale, max_scale, num_scales)
        
        # Execute at each scale
        measurements = []
        errors = []
        
        for scale in noise_scales:
            # Run multiple shots to estimate error
            shots_per_scale = 5
            results = [circuit_executor(noise_scale=scale) for _ in range(shots_per_scale)]
            measurements.append(np.mean(results))
            errors.append(np.std(results) / np.sqrt(shots_per_scale))
        
        measurements = np.array(measurements)
        errors = np.array(errors)
        
        # Fit model and extrapolate
        C0, params = self.fit_noise_model(noise_scales, measurements, errors)
        
        diagnostics = {
            'noise_scales': noise_scales.tolist(),
            'raw_measurements': measurements.tolist(),
            'measurement_errors': errors.tolist(),
            'fit_params': params,
            'extrapolated_value': C0
        }
        
        return C0, diagnostics


class AdaptiveErrorMitigation:
    """
    Time-correlated error mitigation using Bayesian priors.
    
    Uses results from shallow circuits to inform mitigation of deeper circuits.
    """
    
    def __init__(self):
        self.circuit_history = []
        self.mitigation_models = {}
        
    def mitigate_time_series(
        self,
        circuits: List[Callable],
        circuit_depths: List[int],
        initial_priors: Optional[Dict] = None
    ) -> List[Tuple[float, Dict]]:
        """
        Mitigate a sequence of circuits with increasing depth.
        
        Args:
            circuits: List of circuit executors (callables)
            circuit_depths: Depth of each circuit (for sorting)
            initial_priors: Optional starting priors for first circuit
            
        Returns:
            List of (mitigated_value, diagnostics) tuples
        """
        # Sort by depth (shallow first)
        sorted_indices = np.argsort(circuit_depths)
        
        results = []
        zne = PauliPathZNE(use_bayesian_priors=True)
        
        if initial_priors:
            zne.prior_params = initial_priors
        
        for idx in sorted_indices:
            circuit = circuits[idx]
            depth = circuit_depths[idx]
            
            logger.info(f"Mitigating circuit {idx+1}/{len(circuits)} (depth={depth})")
            
            # Run ZNE with Bayesian priors from previous circuits
            mitigated, diagnostics = zne.extrapolate_zero_noise(circuit)
            
            results.append((mitigated, diagnostics))
            
            # Prior for next circuit is automatically updated in zne
        
        # Restore original order
        ordered_results = [None] * len(results)
        for original_idx, sorted_idx in enumerate(sorted_indices):
            ordered_results[sorted_idx] = results[original_idx]
        
        return ordered_results


class DynamicalDecouplingEnhanced:
    """
    Advanced dynamical decoupling sequences.
    
    Implements XY-4, KDD (Knill Dynamical Decoupling), and adaptive DD.
    """
    
    @staticmethod
    def insert_xy4_sequence(circuit, idle_qubits: List[int], idle_duration: float):
        """
        Insert XY-4 pulse sequence during idle periods.
        
        XY-4 sequence: X - τ - Y - 2τ - X - 2τ - Y - τ
        Better than simple X-X as it cancels more error terms.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import XGate, YGate
        
        tau = idle_duration / 8  # 8 segments in XY-4
        
        # Create DD subcircuit
        dd_circ = QuantumCircuit(len(idle_qubits))
        for i, qubit in enumerate(idle_qubits):
            dd_circ.x(i)
            dd_circ.delay(tau, i)
            dd_circ.y(i)
            dd_circ.delay(2*tau, i)
            dd_circ.x(i)
            dd_circ.delay(2*tau, i)
            dd_circ.y(i)
            dd_circ.delay(tau, i)
        
        return dd_circ
    
    @staticmethod
    def optimize_dd_spacing(circuit, backend):
        """
        Optimize DD pulse spacing based on backend characteristics.
        
        Adapts to T1, T2, gate times for optimal error suppression.
        """
        # Get backend properties
        T1 = backend.properties().t1(0) if hasattr(backend, 'properties') else 100e-6
        T2 = backend.properties().t2(0) if hasattr(backend, 'properties') else 100e-6
        
        # Optimal spacing: balance coherence protection vs gate error introduction
        optimal_spacing = min(T1, T2) / 4
        
        return optimal_spacing


# Example usage
if __name__ == "__main__":
    # Simulate a noisy quantum expectation value
    def mock_circuit_executor(noise_scale=1.0):
        """Mock circuit that returns noisy expectation value."""
        true_value = 0.75
        noise = noise_scale * 0.1 * np.random.randn()
        decay = np.exp(-0.5 * noise_scale)  # Exponential damping
        return true_value * decay + noise
    
    # Apply Pauli Path ZNE
    zne = PauliPathZNE()
    mitigated_value, diagnostics = zne.extrapolate_zero_noise(
        mock_circuit_executor,
        num_scales=5,
        max_scale=3.0
    )
    
    print(f"Raw measurement (λ=1.0): {mock_circuit_executor(1.0):.4f}")
    print(f"Mitigated value (λ=0.0): {mitigated_value:.4f}")
    print(f"True value: 0.7500")
    print(f"\nFit parameters:")
    for k, v in diagnostics['fit_params'].items():
        print(f"  {k}: {v:.4f}")