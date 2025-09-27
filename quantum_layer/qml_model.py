# quantum_layer/qml_model.py

import numpy as np
import logging
from typing import Optional, Union, Dict, Any
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.primitives import Sampler
from qiskit.algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from .qml_encoder import QMLEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QMLLinkPredictor:
    """
    Quantum Machine Learning model for knowledge graph link prediction.
    
    Supports two approaches:
      1. Variational Quantum Classifier (VQC) — end-to-end quantum model
      2. Quantum Kernel + Classical SVM (QSVC) — hybrid kernel method
    
    Designed for biomedical tasks with small feature dimensions (5-8D).
    """
    
    def __init__(
        self,
        model_type: str = "VQC",
        encoding_method: str = "feature_map",  # "amplitude" or "feature_map"
        num_qubits: int = 5,
        ansatz_type: str = "RealAmplitudes",   # for VQC
        ansatz_reps: int = 3,
        optimizer: str = "COBYLA",
        max_iter: int = 100,
        feature_map_type: str = "ZZ",          # for feature_map encoding
        feature_map_reps: int = 2,
        random_state: int = 42,
        quantum_config_path: str = "config/quantum_config.yaml"
    ):
        """
        Initialize QML model.
        
        Args:
            model_type: "VQC" or "QSVC"
            encoding_method: "amplitude" or "feature_map" (basis not supported for QML models)
            num_qubits: Number of qubits (must match encoder)
            ansatz_type: "RealAmplitudes" or "EfficientSU2"
            ansatz_reps: Repetition blocks in ansatz
            optimizer: "COBYLA" or "SPSA"
            max_iter: Max optimization iterations
            feature_map_type: "ZZ", "Z", or "Pauli" (if using feature_map)
            feature_map_reps: Repetition blocks in feature map
            random_state: For reproducibility
            quantum_config_path: Path to quantum configuration file
        """
        if model_type not in ["VQC", "QSVC"]:
            raise ValueError("model_type must be 'VQC' or 'QSVC'")
        
        if encoding_method == "basis":
            raise ValueError("Basis encoding not supported for QML models. Use 'amplitude' or 'feature_map'.")
        
        self.model_type = model_type
        self.encoding_method = encoding_method
        self.num_qubits = num_qubits
        self.random_state = random_state
        self.max_iter = max_iter
        self.feature_map_type = feature_map_type
        self.feature_map_reps = feature_map_reps
        self.quantum_config_path = quantum_config_path
        
        # Initialize encoder
        self.encoder = QMLEncoder(
            encoding_method=encoding_method,
            num_qubits=num_qubits,
            feature_map_type=feature_map_type,
            feature_map_reps=feature_map_reps
        )
        
        # Set up optimizer
        if optimizer == "COBYLA":
            self.optimizer = COBYLA(maxiter=max_iter)
        elif optimizer == "SPSA":
            self.optimizer = SPSA(maxiter=max_iter)
        else:
            raise ValueError("optimizer must be 'COBYLA' or 'SPSA'")
        
        # Build ansatz (for VQC)
        if ansatz_type == "RealAmplitudes":
            self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=ansatz_reps)
        elif ansatz_type == "EfficientSU2":
            self.ansatz = EfficientSU2(num_qubits=num_qubits, reps=ansatz_reps)
        else:
            raise ValueError("ansatz_type must be 'RealAmplitudes' or 'EfficientSU2'")
        
        self.model = None
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}
    
    def _prepare_quantum_kernel(self) -> QuantumKernel:
        """Prepare quantum kernel for QSVC."""
        if self.encoding_method == "amplitude":
            raise NotImplementedError("Amplitude encoding not directly supported in Qiskit QuantumKernel. Use feature_map.")
        
        # For feature_map, the encoder's feature map is the kernel's feature map
        if self.feature_map_type == "ZZ":
            from qiskit.circuit.library import ZZFeatureMap
            feature_map = ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                entanglement="linear"
            )
        elif self.feature_map_type == "Z":
            from qiskit.circuit.library import ZFeatureMap
            feature_map = ZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps
            )
        else:
            raise ValueError(f"Unsupported feature_map_type for kernel: {self.feature_map_type}")
        
        # Import quantum executor for unified sampler
        try:
            from .quantum_executor import QuantumExecutor
            quantum_executor = QuantumExecutor(self.quantum_config_path)
            sampler, _ = quantum_executor.get_sampler()
        except Exception as e:
            logger.warning(f"Could not initialize quantum executor: {e}. Using local sampler.")
            sampler = Sampler()
        
        return QuantumKernel(
            feature_map=feature_map,
            sampler=sampler
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QMLLinkPredictor":
        """
        Train the QML model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels (0 or 1)
        """
        logger.info(f"Training {self.model_type} with {self.encoding_method} encoding...")
        
        # Initialize quantum executor for unified execution
        try:
            from .quantum_executor import QuantumExecutor
            self.quantum_executor = QuantumExecutor(self.quantum_config_path)
            sampler, backend_name = self.quantum_executor.get_sampler()
            logger.info(f"Using quantum backend: {backend_name}")
        except Exception as e:
            logger.warning(f"Quantum executor not available: {e}. Using local sampler.")
            sampler = Sampler()
            self.quantum_executor = None
        
        if self.model_type == "VQC":
            # For VQC, we need to handle amplitude encoding carefully
            if self.encoding_method == "amplitude":
                # Amplitude encoding requires a different approach
                # We'll use the encoder to create initial states, but VQC in Qiskit ML
                # expects a feature map. So we fall back to custom circuit construction.
                raise NotImplementedError(
                    "Amplitude encoding with VQC requires custom implementation. "
                    "Use 'feature_map' encoding for VQC, or try QSVC with amplitude (not recommended)."
                )
            else:
                # Use feature map encoding (standard VQC workflow)
                quantum_kernel = self._prepare_quantum_kernel()
                self.model = VQC(
                    sampler=sampler,
                    feature_map=quantum_kernel.feature_map,
                    ansatz=self.ansatz,
                    optimizer=self.optimizer,
                    callback=lambda x: logger.debug(f"VQC loss: {x}")
                )
        
        elif self.model_type == "QSVC":
            if self.encoding_method == "amplitude":
                raise NotImplementedError(
                    "Amplitude encoding not supported in Qiskit QSVC. Use 'feature_map'."
                )
            else:
                quantum_kernel = self._prepare_quantum_kernel()
                self.model = QSVC(
                    quantum_kernel=quantum_kernel
                )
        
        # Fit the model
        try:
            self.model.fit(X, y)
            self.is_fitted = True
            logger.info(f"{self.model_type} training completed.")
        except Exception as e:
            # Close quantum session on error to avoid costs
            if hasattr(self, 'quantum_executor') and self.quantum_executor:
                self.quantum_executor.close_session()
            raise RuntimeError(f"QML model training failed: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback: use decision function and sigmoid
            from scipy.special import expit
            decision = self.model.decision_function(X)
            proba = expit(decision)
            return np.vstack([1 - proba, proba]).T
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy score."""
        if not self.is_fitted:
            raise RuntimeError("Model not trained. Call fit() first.")
        return self.model.score(X, y)


# Example usage (uncomment to test)
# if __name__ == "__main__":
#     # Generate dummy data (5D features, binary labels)
#     np.random.seed(42)
#     X = np.random.rand(20, 5)  # 20 samples, 5 features
#     y = np.random.randint(0, 2, 20)
#     
#     # Train VQC with feature map encoding
#     predictor = QMLLinkPredictor(
#         model_type="VQC",
#         encoding_method="feature_map",
#         num_qubits=5,
#         feature_map_type="ZZ",
#         feature_map_reps=2,
#         ansatz_type="RealAmplitudes",
#         ansatz_reps=2,
#         max_iter=50
#     )
#     
#     predictor.fit(X, y)
#     accuracy = predictor.score(X, y)
#     proba = predictor.predict_proba(X)
#     
#     print(f"Training accuracy: {accuracy:.4f}")
#     print(f"Predicted probabilities shape: {proba.shape}")