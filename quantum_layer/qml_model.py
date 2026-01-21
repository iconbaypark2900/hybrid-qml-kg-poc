# quantum_layer/qml_model.py
"""
Hybrid QML link predictor (QSVC / VQC) compatible with Qiskit 1.x and
qiskit-machine-learning >= 0.8.

- QSVC uses fidelity kernels:
    * statevector → FidelityStatevectorKernel(feature_map)
    * sampling/hardware → FidelityQuantumKernel(feature_map, fidelity=ComputeUncompute(sampler))
- VQC uses feature-map encoding + parametric ansatz + optimizer.

Assumes upstream prepares QML-ready features with dim == num_qubits
(e.g., |h - t| in PCA-reduced space, qml_dim == num_qubits).
"""

from __future__ import annotations

import logging
from typing import Optional, Dict
from pathlib import Path
import yaml

import numpy as np

# Core Qiskit imports (tested with qiskit~=1.2, qiskit-ml~=0.8.4, qiskit-algorithms~=0.3)
from qiskit.circuit.library import RealAmplitudes, EfficientSU2, ZZFeatureMap, ZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

logger = logging.getLogger(__name__)


def load_quantum_config(config_path: str = "config/quantum_layer_config.yaml") -> Dict:
    """
    Load quantum layer configuration from YAML file.

    Args:
        config_path: Path to the quantum layer config YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    if not Path(config_path).exists():
        logger.warning(f"Config file not found at {config_path}, using defaults")
        return {
            "model": {
                "model_type": "QSVC",
                "encoding_method": "feature_map",
                "num_qubits": 5,
                "random_state": 42
            },
            "feature_map": {
                "feature_map_type": "ZZ",
                "feature_map_reps": 2
            },
            "vqc": {
                "ansatz_type": "RealAmplitudes",
                "ansatz_reps": 3,
                "optimizer": "COBYLA",
                "max_iter": 50
            },
            "quantum_executor": {
                "quantum_config_path": "config/quantum_config.yaml"
            }
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

# Try to import the local encoder (not strictly required for this class to run)
try:
    from .qml_encoder import QMLEncoder  # noqa: F401
except Exception:
    QMLEncoder = None  # type: ignore


class QMLLinkPredictor:
    """
    Wrapper for quantum classifiers.

    Parameters
    ----------
    model_type : {"QSVC","VQC"}
    encoding_method : {"feature_map"}
    num_qubits : int
    ansatz_type : {"RealAmplitudes","EfficientSU2"}
    ansatz_reps : int
    optimizer : {"COBYLA","SPSA"}
    max_iter : int
    feature_map_type : {"ZZ","Z"}
    feature_map_reps : int
    random_state : int
    quantum_config_path : str (YAML; optional)
    """

    def __init__(
        self,
        model_type: Optional[str] = None,
        encoding_method: Optional[str] = None,
        num_qubits: Optional[int] = None,
        ansatz_type: Optional[str] = None,
        ansatz_reps: Optional[int] = None,
        optimizer: Optional[str] = None,
        max_iter: Optional[int] = None,
        feature_map_type: Optional[str] = None,
        feature_map_reps: Optional[int] = None,
        random_state: Optional[int] = None,
        quantum_config_path: Optional[str] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        # Load config if not provided
        if config is None:
            if config_path is None:
                config_path = "config/quantum_layer_config.yaml"
            config = load_quantum_config(config_path)

        # Use provided parameters or fall back to config
        self.model_type = (model_type if model_type is not None else config["model"]["model_type"]).upper()
        self.encoding_method = encoding_method if encoding_method is not None else config["model"]["encoding_method"]
        self.num_qubits = int(num_qubits if num_qubits is not None else config["model"]["num_qubits"])
        self.ansatz_type = ansatz_type if ansatz_type is not None else config["vqc"]["ansatz_type"]
        self.ansatz_reps = int(ansatz_reps if ansatz_reps is not None else config["vqc"]["ansatz_reps"])
        self.optimizer_name = optimizer if optimizer is not None else config["vqc"]["optimizer"]
        self.max_iter = int(max_iter if max_iter is not None else config["vqc"]["max_iter"])
        self.feature_map_type = feature_map_type if feature_map_type is not None else config["feature_map"]["feature_map_type"]
        self.feature_map_reps = int(feature_map_reps if feature_map_reps is not None else config["feature_map"]["feature_map_reps"])
        self.random_state = int(random_state if random_state is not None else config["model"]["random_state"])
        self.quantum_config_path = quantum_config_path if quantum_config_path is not None else config["quantum_executor"]["quantum_config_path"]

        # Store config for reference
        self.config = config

        # runtime fields
        self.model = None
        self.is_fitted = False
        self.metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Sanity / builders
    # ------------------------------------------------------------------
    def _ensure_qiskit_available(self):
        """
        Verifies that Qiskit libraries are installed and available.

        Returns
        -------
        bool
            True if Qiskit is available, otherwise raises an ImportError.
        """
        # Import checks are already at file scope; this method exists for clarity & future guards.
        return True

    def _build_optimizer(self):
        """
        Constructs a Qiskit optimizer instance based on the specified optimizer name.

        Returns
        -------
        qiskit_algorithms.optimizers.Optimizer
            An instance of a Qiskit optimizer (e.g., COBYLA, SPSA).
        """
        name = self.optimizer_name.upper()
        if name == "COBYLA":
            return COBYLA(maxiter=self.max_iter)
        if name == "SPSA":
            return SPSA(maxiter=self.max_iter)
        raise ValueError("optimizer must be one of: {'COBYLA','SPSA'}")

    def _build_ansatz(self):
        """
        Constructs a Qiskit ansatz (variational form) instance.

        Returns
        -------
        qiskit.circuit.library.Ansatz
            An instance of a Qiskit ansatz (e.g., RealAmplitudes, EfficientSU2).
        """
        if self.ansatz_type == "RealAmplitudes":
            return RealAmplitudes(num_qubits=self.num_qubits, reps=self.ansatz_reps)
        if self.ansatz_type == "EfficientSU2":
            return EfficientSU2(num_qubits=self.num_qubits, reps=self.ansatz_reps)
        raise ValueError("ansatz_type must be one of: {'RealAmplitudes','EfficientSU2'}")

    def _make_feature_map(self):
        """
        Constructs a Qiskit feature map instance.

        Returns
        -------
        qiskit.circuit.library.FeatureMap
            An instance of a Qiskit feature map (e.g., ZZFeatureMap, ZFeatureMap).
        """
        if self.encoding_method != "feature_map":
            raise NotImplementedError("Only 'feature_map' encoding is supported.")
        if self.feature_map_type == "ZZ":
            return ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                entanglement="linear",
            )
        if self.feature_map_type == "Z":
            return ZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
            )
        raise ValueError("feature_map_type must be one of: {'ZZ','Z'}")

    def _prepare_quantum_kernel(self, sampler: Sampler, exec_mode: str):
        """
        Construct a QSVC fidelity kernel consistent with execution mode.

        Parameters
        ----------
        sampler : qiskit.primitives.Sampler
            The sampler to use for quantum kernel evaluation.
        exec_mode : str
            The execution mode, e.g., "simulator_statevector", "simulator", etc.

        Returns
        -------
        qiskit_machine_learning.kernels.QuantumKernel
            The constructed quantum kernel.
        """
        fm = self._make_feature_map()
        if exec_mode in ("statevector", "simulator_statevector"):
            return FidelityStatevectorKernel(feature_map=fm)
        # Decompose composite feature maps so Aer/backends don't error on unknown instructions
        fm_exec = fm.decompose(reps=10)
        fidelity = ComputeUncompute(sampler=sampler)
        return FidelityQuantumKernel(feature_map=fm_exec, fidelity=fidelity)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "QMLLinkPredictor":
        """
        Train the selected model.

        Parameters
        ----------
        X : np.ndarray
            The training data.
        y : np.ndarray
            The training labels.

        Returns
        -------
        QMLLinkPredictor
            The fitted QMLLinkPredictor instance.
        """
        self._ensure_qiskit_available()

        # Choose sampler + exec mode (prefer exact sims)
        sampler: Optional[Sampler]
        exec_mode = "simulator_statevector"
        try:
            from .quantum_executor import QuantumExecutor
            qx = QuantumExecutor(self.quantum_config_path)
            sampler, exec_mode = qx.get_sampler()
        except Exception as e:
            logger.info(f"QuantumExecutor fallback to local sampler: {e}")
            try:
                from qiskit.primitives import StatevectorSampler
                sampler = StatevectorSampler()
                exec_mode = "simulator_statevector"
            except Exception:
                # Very defensive: use Aer SamplerV2 for a shot-based ideal fallback
                from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
                sampler = AerSamplerV2(default_shots=1024)
                exec_mode = "simulator"

        if self.model_type == "VQC":
            if self.encoding_method != "feature_map":
                raise NotImplementedError("VQC supports only 'feature_map' in this project.")
            feature_map = self._make_feature_map()
            if exec_mode not in ("statevector", "simulator_statevector"):
                feature_map = feature_map.decompose(reps=10)
            ansatz = self._build_ansatz()
            optimizer = self._build_optimizer()
            self.model = VQC(
                sampler=sampler,
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
            )
            logger.info("Training VQC with feature_map encoding...")

        elif self.model_type == "QSVC":
            if self.encoding_method != "feature_map":
                raise NotImplementedError("QSVC supports only 'feature_map' in this project.")
            quantum_kernel = self._prepare_quantum_kernel(sampler=sampler, exec_mode=exec_mode)
            self.model = QSVC(quantum_kernel=quantum_kernel)
            logger.info("Training QSVC with feature_map encoding...")

        else:
            raise ValueError("model_type must be 'QSVC' or 'VQC'")

        if X.ndim != 2 or X.shape[1] != self.num_qubits:
            raise ValueError(
                f"Input feature dim must equal num_qubits; got X.shape={X.shape}, num_qubits={self.num_qubits}"
            )

        self.model.fit(X, y)
        self.is_fitted = True
        logger.info(f"{self.model_type} training completed.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions on new data.

        Parameters
        ----------
        X : np.ndarray
            The input data to make predictions on.

        Returns
        -------
        np.ndarray
            The predicted labels.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns probability estimates for new data.

        Parameters
        ----------
        X : np.ndarray
            The input data to make predictions on.

        Returns
        -------
        np.ndarray
            The probability estimates of the classes.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # VQC usually exposes predict_proba; QSVC may not (SVC w/ kernel)
        if hasattr(self.model, "predict_proba"):
            try:
                return self.model.predict_proba(X)
            except Exception:
                pass
        # fallback: map decision function to probability-like scores
        if hasattr(self.model, "decision_function"):
            df = self.model.decision_function(X)
            df = np.asarray(df, dtype=float)
            proba_pos = 1.0 / (1.0 + np.exp(-df))
            return np.vstack([1 - proba_pos, proba_pos]).T
        # last resort: hard labels to 0/1 probs
        y = self.model.predict(X)
        return np.vstack([1 - y, y]).T.astype(float)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : np.ndarray
            The test data.
        y : np.ndarray
            The true labels for the test data.

        Returns
        -------
        float
            The mean accuracy.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return float(self.model.score(X, y))
