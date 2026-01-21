# quantum_layer/qml_encoder.py

import numpy as np
from typing import Union, Tuple, Optional, Dict
from pathlib import Path
import yaml
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
import logging

logging.basicConfig(level=logging.INFO)
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
                "encoding_method": "feature_map",
                "num_qubits": 5,
                "random_state": 42
            },
            "feature_map": {
                "feature_map_type": "ZZ",
                "feature_map_reps": 2
            }
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


class QMLEncoder:
    """
    Encodes classical feature vectors into quantum states for QML.

    Supports:
      - Amplitude Encoding (optimal for high-dimensional data)
      - Basis Encoding (for very low-dimensional data)
      - Quantum Feature Maps (ZZ, Z, Pauli) for kernel methods

    Designed for link prediction features from biomedical KGs.
    """

    def __init__(
        self,
        encoding_method: Optional[str] = None,
        num_qubits: Optional[int] = None,
        feature_map_type: Optional[str] = None,
        feature_map_reps: Optional[int] = None,
        config_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize encoder.

        Args:
            encoding_method: "amplitude", "basis", or "feature_map" (overrides config)
            num_qubits: Target number of qubits (determines max feature dim) (overrides config)
            feature_map_type: "ZZ", "Z", or "Pauli" (used if encoding_method="feature_map") (overrides config)
            feature_map_reps: Number of repetition blocks in feature map (overrides config)
            config_path: Path to quantum layer config YAML file (default: "config/quantum_layer_config.yaml")
            config: Configuration dictionary (if provided, config_path is ignored)
        """
        # Load config if not provided
        if config is None:
            if config_path is None:
                config_path = "config/quantum_layer_config.yaml"
            config = load_quantum_config(config_path)

        # Use provided parameters or fall back to config
        self.encoding_method = encoding_method if encoding_method is not None else config["model"]["encoding_method"]
        self.num_qubits = int(num_qubits if num_qubits is not None else config["model"]["num_qubits"])
        self.feature_map_type = feature_map_type if feature_map_type is not None else config["feature_map"]["feature_map_type"]
        self.feature_map_reps = int(feature_map_reps if feature_map_reps is not None else config["feature_map"]["feature_map_reps"])

        if self.encoding_method not in ["amplitude", "basis", "feature_map"]:
            raise ValueError("encoding_method must be 'amplitude', 'basis', or 'feature_map'")

        # Store config for reference
        self.config = config

        # Max feature dimensions based on encoding
        if encoding_method == "amplitude":
            self.max_features = 2 ** num_qubits
        elif encoding_method == "basis":
            self.max_features = num_qubits
        else:  # feature_map
            self.max_features = num_qubits  # Feature maps use 1 feature per qubit

        logger.info(f"Initialized {encoding_method} encoder with {num_qubits} qubits "
                    f"(max features: {self.max_features})")

    def _validate_and_prepare_features(self, x: np.ndarray) -> np.ndarray:
        """
        Validate input features and adjust dimensionality to match encoder capacity.

        Args:
            x: Input feature vector (1D numpy array)

        Returns:
            Prepared feature vector of length self.max_features
        """
        if x.ndim != 1:
            raise ValueError("Input features must be a 1D array")

        if len(x) == 0:
            raise ValueError("Input features cannot be empty")

        # Handle dimension mismatch
        if len(x) > self.max_features:
            logger.warning(f"Feature dimension ({len(x)}) exceeds max ({self.max_features}). Truncating.")
            x = x[:self.max_features]
        elif len(x) < self.max_features:
            if self.encoding_method == "amplitude":
                # Pad with zeros for amplitude encoding
                padding = self.max_features - len(x)
                x = np.pad(x, (0, padding), mode='constant')
                logger.debug(f"Padded features to {self.max_features} dimensions for amplitude encoding.")
            else:
                # For basis/feature_map, pad or repeat cyclically
                repeats = int(np.ceil(self.max_features / len(x)))
                x = np.tile(x, repeats)[:self.max_features]
                logger.debug(f"Repeated features to {self.max_features} dimensions.")

        # Normalize for amplitude encoding (critical!)
        if self.encoding_method == "amplitude":
            norm = np.linalg.norm(x)
            if norm == 0:
                raise ValueError("Cannot encode zero vector with amplitude encoding")
            x = x / norm
            logger.debug("Normalized feature vector to unit length for amplitude encoding.")

        return x

    def amplitude_encoding(self, x: np.ndarray) -> QuantumCircuit:
        """
        Amplitude encoding: maps x to quantum state sum_i x_i |i>.
        Uses Qiskit's Initialize instruction (decomposes to gates automatically).

        Args:
            x: Input feature vector (1D numpy array)

        Returns:
            QuantumCircuit object
        """
        x = self._validate_and_prepare_features(x)

        qc = QuantumCircuit(self.num_qubits)
        try:
            qc.initialize(x, range(self.num_qubits))
        except Exception as e:
            raise RuntimeError(f"Amplitude encoding failed: {e}")

        qc = qc.decompose()  # Decompose initialize into native gates
        logger.debug(f"Created amplitude-encoded circuit with {qc.num_qubits} qubits, {qc.size()} gates.")
        return qc

    def basis_encoding(self, x: np.ndarray) -> QuantumCircuit:
        """
        Basis encoding: maps binary features to computational basis states.
        Non-binary features are thresholded at 0.5.

        Args:
            x: Input feature vector (1D numpy array)

        Returns:
            QuantumCircuit object
        """
        x = self._validate_and_prepare_features(x)

        # Threshold to binary
        x_bin = (x > 0.5).astype(int)

        qc = QuantumCircuit(self.num_qubits)
        for i, bit in enumerate(x_bin):
            if bit == 1:
                qc.x(i)

        logger.debug(f"Created basis-encoded circuit: |{''.join(map(str, x_bin[::-1]))}>")
        return qc

    def feature_map_encoding(self, x: np.ndarray) -> QuantumCircuit:
        """
        Quantum feature map encoding (for kernel methods or VQC).
        Maps features to parameterized rotations.

        Args:
            x: Input feature vector (1D numpy array)

        Returns:
            QuantumCircuit object
        """
        x = self._validate_and_prepare_features(x)

        if self.feature_map_type == "ZZ":
            feature_map = ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                entanglement="linear"
            )
        elif self.feature_map_type == "Z":
            feature_map = ZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps
            )
        elif self.feature_map_type == "Pauli":
            feature_map = PauliFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                paulis=["Z", "ZZ"]
            )
        else:
            raise ValueError(f"Unsupported feature_map_type: {self.feature_map_type}")

        qc = feature_map.bind_parameters(x)
        logger.debug(f"Created {self.feature_map_type} feature map circuit with {qc.num_qubits} qubits.")
        return qc

    def encode(self, x: np.ndarray) -> QuantumCircuit:
        """
        Main encoding interface. Returns a parameterized or fixed quantum circuit.

        Args:
            x: Input feature vector (1D numpy array)

        Returns:
            QuantumCircuit object
        """
        if self.encoding_method == "amplitude":
            return self.amplitude_encoding(x)
        elif self.encoding_method == "basis":
            return self.basis_encoding(x)
        elif self.encoding_method == "feature_map":
            return self.feature_map_encoding(x)
        else:
            raise RuntimeError("Unexpected encoding method")

    def batch_encode(self, X: np.ndarray) -> list:
        """
        Encode a batch of feature vectors.
        Returns list of QuantumCircuit objects.

        Args:
            X: 2D numpy array of shape (num_samples, num_features)
        Returns:
            List of QuantumCircuit objects
        """
        circuits = []
        for i, x in enumerate(X):
            try:
                qc = self.encode(x)
                circuits.append(qc)
            except Exception as e:
                logger.error(f"Failed to encode sample {i}: {e}")
                # Append empty circuit or handle as needed
                circuits.append(QuantumCircuit(self.num_qubits))
        return circuits


# Example usage (uncomment to test)
# if __name__ == "__main__":
#     # Simulate a 5D feature vector from drug-disease embedding
#     x = np.random.rand(5)
#
#     # Amplitude encoding (requires power-of-2 dimensions)
#     encoder = QMLEncoder(encoding_method="amplitude", num_qubits=3)  # 2^3 = 8 dims
#     qc_amp = encoder.encode(x)
#     print("Amplitude circuit:")
#     print(qc_amp)
#
#     # Feature map encoding (1 feature per qubit)
#     encoder_fm = QMLEncoder(encoding_method="feature_map", num_qubits=5, feature_map_type="ZZ")
#     qc_fm = encoder_fm.encode(x)
#     print("\nFeature map circuit:")
#     print(qc_fm)