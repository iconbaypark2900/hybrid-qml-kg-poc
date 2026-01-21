"""
Quantum-Specific Feature Engineering

Creates features optimized for quantum models using:
1. Quantum kernel-based feature selection
2. Quantum amplitude/phase encoding features
3. Quantum entanglement-inspired features
4. Quantum feature map expressivity features
"""

import logging
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

try:
    from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
    from qiskit_machine_learning.kernels import FidelityStatevectorKernel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logger.warning("Qiskit not available. Quantum feature engineering will be disabled.")


class QuantumFeatureEngineer:
    """
    Quantum-specific feature engineering for link prediction.
    
    Creates features that are optimized for quantum models by:
    1. Using quantum kernel importance for feature selection
    2. Creating quantum-native features (amplitude, phase, entanglement)
    3. Maximizing quantum feature map expressivity
    """
    
    def __init__(
        self,
        num_qubits: int = 12,
        feature_map_type: str = 'ZZ',
        feature_map_reps: int = 3,
        entanglement: str = 'full',
        use_quantum_selection: bool = True,
        random_state: int = 42
    ):
        """
        Args:
            num_qubits: Number of qubits for quantum feature map
            feature_map_type: Type of feature map ('ZZ', 'Z', 'Pauli')
            feature_map_reps: Number of feature map repetitions
            entanglement: Entanglement pattern
            use_quantum_selection: Use quantum kernel for feature selection
            random_state: Random seed
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for quantum feature engineering")
        
        self.num_qubits = num_qubits
        self.feature_map_type = feature_map_type
        self.feature_map_reps = feature_map_reps
        self.entanglement = entanglement
        self.use_quantum_selection = use_quantum_selection
        self.random_state = random_state
        
        # Initialize feature map
        self.feature_map = self._create_feature_map()
        
        # Initialize quantum kernel
        self.quantum_kernel = FidelityStatevectorKernel(feature_map=self.feature_map)
        
        # Scalers and PCA
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=num_qubits, random_state=random_state)
        
        # Feature selection
        self.selected_features_ = None
        self.feature_importances_ = None
    
    def _create_feature_map(self):
        """Create quantum feature map."""
        if self.feature_map_type == 'ZZ':
            return ZZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                entanglement=self.entanglement
            )
        elif self.feature_map_type == 'Z':
            return ZFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps
            )
        elif self.feature_map_type == 'Pauli':
            return PauliFeatureMap(
                feature_dimension=self.num_qubits,
                reps=self.feature_map_reps,
                paulis=["Z", "ZZ"],
                entanglement=self.entanglement
            )
        else:
            raise ValueError(f"Unknown feature_map_type: {self.feature_map_type}")
    
    def _compute_quantum_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_features_to_evaluate: int = 100,
        sample_size: int = 500
    ) -> np.ndarray:
        """
        Compute feature importance using quantum kernel separability (fast approximation).
        
        Uses sampling and approximation to make it computationally feasible.
        
        Args:
            X: Features [n_samples, n_features]
            y: Labels [n_samples]
            max_features_to_evaluate: Maximum number of features to evaluate (for speed)
            sample_size: Number of samples to use for kernel computation (for speed)
        """
        logger.info(f"Computing quantum feature importance (fast approximation)...")
        logger.info(f"  Total features: {X.shape[1]}, Evaluating top {max_features_to_evaluate}")
        logger.info(f"  Using {sample_size} samples for kernel computation (for speed)")
        
        # Use a subset of samples for faster computation
        if X.shape[0] > sample_size:
            np.random.seed(self.random_state)
            sample_indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
        else:
            X_sample = X
            y_sample = y
            sample_indices = np.arange(X.shape[0])
        
        # Reduce features to num_qubits for kernel computation
        X_scaled = self.scaler.fit_transform(X_sample)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        # Compute baseline kernel separability
        logger.info("  Computing baseline kernel separability...")
        K_full = self.quantum_kernel.evaluate(X_reduced)
        baseline_sep = self._compute_kernel_separability(K_full, y_sample)
        logger.info(f"  Baseline separability: {baseline_sep:.6f}")
        
        # Use classical variance/separability as proxy for initial ranking
        # This is much faster than computing quantum kernels for all features
        from sklearn.feature_selection import mutual_info_classif
        logger.info("  Using mutual information for initial feature ranking...")
        mi_scores = mutual_info_classif(X_sample, y_sample, random_state=self.random_state)
        
        # Select top features to evaluate with quantum kernel
        top_indices = np.argsort(mi_scores)[-max_features_to_evaluate:][::-1]
        
        # Compute importance for selected features only
        importances = np.zeros(X.shape[1])
        importances[:] = mi_scores  # Initialize with MI scores
        
        logger.info(f"  Evaluating top {len(top_indices)} features with quantum kernel...")
        for idx, i in enumerate(top_indices):
            if (idx + 1) % 10 == 0:
                logger.info(f"    Progress: {idx+1}/{len(top_indices)} features evaluated")
            
            # Remove feature i
            X_removed = np.delete(X_sample, i, axis=1)
            
            # Recompute PCA and kernel
            X_removed_scaled = self.scaler.fit_transform(X_removed)
            X_removed_reduced = self.pca.fit_transform(X_removed_scaled)
            
            K_removed = self.quantum_kernel.evaluate(X_removed_reduced)
            removed_sep = self._compute_kernel_separability(K_removed, y_sample)
            
            # Importance = how much separability decreases when feature is removed
            importances[i] = baseline_sep - removed_sep
        
        logger.info(f"  ✓ Quantum feature importance computed")
        
        return importances
    
    def _compute_kernel_separability(self, K: np.ndarray, y: np.ndarray) -> float:
        """Compute kernel separability score."""
        pos_mask = y == 1
        neg_mask = y == 0
        
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return 0.0
        
        pos_pos_kernel = K[np.ix_(pos_mask, pos_mask)]
        neg_neg_kernel = K[np.ix_(neg_mask, neg_mask)]
        pos_neg_kernel = K[np.ix_(pos_mask, neg_mask)]
        
        pos_pos_mean = np.mean(pos_pos_kernel[~np.eye(pos_pos_kernel.shape[0], dtype=bool)])
        neg_neg_mean = np.mean(neg_neg_kernel[~np.eye(neg_neg_kernel.shape[0], dtype=bool)])
        pos_neg_mean = np.mean(pos_neg_kernel)
        
        within_class_mean = (pos_pos_mean + neg_neg_mean) / 2
        separability = within_class_mean / (pos_neg_mean + 1e-8)
        
        return separability
    
    def create_quantum_native_features(
        self,
        head_embeddings: np.ndarray,
        tail_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Create quantum-native features optimized for quantum circuits.
        
        Features include:
        1. Amplitude encoding features (normalized amplitudes)
        2. Phase encoding features (angles/phases)
        3. Entanglement-inspired features (correlations)
        4. Quantum distance features (fidelity-like measures)
        """
        logger.info("Creating quantum-native features...")
        
        # Concatenate head and tail embeddings
        link_embeddings = np.concatenate([head_embeddings, tail_embeddings], axis=1)
        
        features = []
        
        # 1. Amplitude encoding features (normalize for quantum amplitude encoding)
        # Quantum amplitude encoding requires normalized vectors
        norms = np.linalg.norm(link_embeddings, axis=1, keepdims=True)
        normalized_embs = link_embeddings / (norms + 1e-8)
        features.append(normalized_embs)
        
        # 2. Phase encoding features (angles/arguments)
        # Convert to polar coordinates (amplitude, phase)
        phases = np.angle(link_embeddings.astype(complex))
        features.append(phases)
        
        # 3. Entanglement-inspired features (correlations between head and tail)
        # Compute correlations between head and tail embeddings
        head_norm = np.linalg.norm(head_embeddings, axis=1, keepdims=True)
        tail_norm = np.linalg.norm(tail_embeddings, axis=1, keepdims=True)
        
        # Dot product (cosine similarity)
        dot_products = np.sum(head_embeddings * tail_embeddings, axis=1, keepdims=True)
        cosine_similarity = dot_products / ((head_norm * tail_norm) + 1e-8)
        features.append(cosine_similarity)
        
        # Element-wise product (for entanglement)
        element_wise_product = head_embeddings * tail_embeddings
        features.append(element_wise_product)
        
        # Element-wise difference
        element_wise_diff = np.abs(head_embeddings - tail_embeddings)
        features.append(element_wise_diff)
        
        # 4. Quantum distance features (fidelity-like measures)
        # Fidelity between head and tail embeddings
        head_normalized = head_embeddings / (head_norm + 1e-8)
        tail_normalized = tail_embeddings / (tail_norm + 1e-8)
        fidelity = np.sum(head_normalized * tail_normalized, axis=1, keepdims=True) ** 2
        features.append(fidelity)
        
        # Quantum overlap (inner product squared)
        overlap = np.sum(head_embeddings * tail_embeddings, axis=1, keepdims=True) ** 2
        features.append(overlap)
        
        # Combine all features
        quantum_features = np.hstack(features)
        
        logger.info(f"  Created {quantum_features.shape[1]} quantum-native features from {link_embeddings.shape[1]} base features")
        
        return quantum_features
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: np.ndarray,
        head_embeddings: Optional[np.ndarray] = None,
        tail_embeddings: Optional[np.ndarray] = None,
        max_features: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit quantum feature engineer and transform features.
        
        Args:
            X: Input features [n_samples, n_features]
            y: Labels [n_samples]
            head_embeddings: Head entity embeddings [n_samples, dim] (optional)
            tail_embeddings: Tail entity embeddings [n_samples, dim] (optional)
            max_features: Maximum number of features to select (None = all)
        
        Returns:
            Transformed features
        """
        logger.info("Fitting quantum feature engineer...")
        
        # Create quantum-native features if embeddings provided
        if head_embeddings is not None and tail_embeddings is not None:
            quantum_features = self.create_quantum_native_features(head_embeddings, tail_embeddings)
            # Combine with original features
            X_enhanced = np.hstack([X, quantum_features])
        else:
            X_enhanced = X
        
        # Quantum-based feature selection
        if self.use_quantum_selection and X_enhanced.shape[1] > self.num_qubits:
            logger.info("  Using quantum kernel for feature selection (fast approximation)...")
            try:
                # Limit number of features to evaluate for speed
                max_features_to_eval = min(100, X_enhanced.shape[1] // 2)
                importances = self._compute_quantum_feature_importance(
                    X_enhanced, y,
                    max_features_to_evaluate=max_features_to_eval,
                    sample_size=min(500, X_enhanced.shape[0])
                )
                self.feature_importances_ = importances
                
                # Select top features
                if max_features is None:
                    max_features = min(X_enhanced.shape[1], X_enhanced.shape[0] // 2)
                
                top_indices = np.argsort(importances)[-max_features:][::-1]
                self.selected_features_ = top_indices
                
                X_selected = X_enhanced[:, top_indices]
                logger.info(f"  Selected {len(top_indices)} features from {X_enhanced.shape[1]}")
            except Exception as e:
                logger.warning(f"  Quantum feature selection failed: {e}. Using all features.")
                X_selected = X_enhanced
                self.selected_features_ = np.arange(X_enhanced.shape[1])
        else:
            X_selected = X_enhanced
            self.selected_features_ = np.arange(X_enhanced.shape[1])
        
        return X_selected
    
    def transform(
        self,
        X: np.ndarray,
        head_embeddings: Optional[np.ndarray] = None,
        tail_embeddings: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Transform features using fitted engineer."""
        # Create quantum-native features if embeddings provided
        if head_embeddings is not None and tail_embeddings is not None:
            quantum_features = self.create_quantum_native_features(head_embeddings, tail_embeddings)
            X_enhanced = np.hstack([X, quantum_features])
        else:
            X_enhanced = X
        
        # Apply feature selection
        if self.selected_features_ is not None:
            X_selected = X_enhanced[:, self.selected_features_]
        else:
            X_selected = X_enhanced
        
        return X_selected

