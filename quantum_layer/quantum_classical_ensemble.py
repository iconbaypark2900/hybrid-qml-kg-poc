"""
Quantum-Classical Ensemble for Knowledge Graph Link Prediction

This module implements a hybrid ensemble that combines quantum and classical models
for improved link prediction performance. The ensemble can use various combination
strategies such as weighted averaging, stacking, or voting.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

from .qml_model import QMLLinkPredictor
from classical_baseline.train_baseline import ClassicalLinkPredictor

logger = logging.getLogger(__name__)


class QuantumClassicalEnsemble:
    """
    A hybrid ensemble that combines quantum and classical models for link prediction.
    
    The ensemble can use various strategies to combine predictions from quantum
    and classical models, including weighted averaging, stacking, or voting.
    """
    
    def __init__(
        self,
        quantum_model: Optional[QMLLinkPredictor] = None,
        classical_model: Optional[ClassicalLinkPredictor] = None,
        ensemble_method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
        use_stacking: bool = False,
        stacking_model: Optional = None,
        calibration_method: Optional[str] = None,
        random_state: int = 42
    ):
        """
        Initialize the quantum-classical ensemble.
        
        Args:
            quantum_model: Pre-trained quantum model (optional)
            classical_model: Pre-trained classical model (optional)
            ensemble_method: Method for combining predictions ('weighted_average', 'voting', 'stacking')
            weights: Weights for each model in weighted average
            use_stacking: Whether to use stacking for ensemble
            stacking_model: Model to use for stacking (default: LogisticRegression)
            calibration_method: Calibration method for probabilities ('isotonic', 'sigmoid', None)
            random_state: Random state for reproducibility
        """
        self.quantum_model = quantum_model
        self.classical_model = classical_model
        self.ensemble_method = ensemble_method
        self.weights = weights or {"quantum": 0.5, "classical": 0.5}
        self.use_stacking = use_stacking
        self.stacking_model = stacking_model or LogisticRegression(random_state=random_state)
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.is_fitted = False
        self.scaler = StandardScaler()
        
        # Validate ensemble method
        valid_methods = ["weighted_average", "voting", "stacking"]
        if self.ensemble_method not in valid_methods:
            raise ValueError(f"Invalid ensemble_method: {self.ensemble_method}. Valid options: {valid_methods}")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_quantum: Optional[np.ndarray] = None,
        X_classical: Optional[np.ndarray] = None
    ) -> 'QuantumClassicalEnsemble':
        """
        Fit the ensemble model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_quantum: Quantum-specific features (if different from X_train)
            X_classical: Classical-specific features (if different from X_train)
        
        Returns:
            Self
        """
        logger.info(f"Fitting quantum-classical ensemble using method: {self.ensemble_method}")
        
        # Use provided features or default to X_train
        X_quantum = X_quantum if X_quantum is not None else X_train
        X_classical = X_classical if X_classical is not None else X_train
        
        # Fit individual models if not already fitted
        if self.quantum_model and not self.quantum_model.is_fitted:
            logger.info("Fitting quantum model...")
            self.quantum_model.fit(X_quantum, y_train)
        
        if self.classical_model and not hasattr(self.classical_model, 'model') or not self.classical_model.model:
            logger.info("Fitting classical model...")
            # Need to train classical model differently since it's designed for KG data
            # For now, we'll use a simplified approach
            if hasattr(self.classical_model, 'scaler'):
                X_classical_scaled = self.classical_model.scaler.transform(X_classical) if self.classical_model.scaler else X_classical
            else:
                X_classical_scaled = self.scaler.fit_transform(X_classical)
            
            if hasattr(self.classical_model, 'model'):
                self.classical_model.model.fit(X_classical_scaled, y_train)
        
        # If using stacking, train the meta-learner
        if self.use_stacking:
            logger.info("Training stacking meta-learner...")
            self._train_stacking_model(X_train, y_train, X_quantum, X_classical)
        
        self.is_fitted = True
        logger.info("Quantum-classical ensemble fitted successfully")
        return self
    
    def _train_stacking_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_quantum: np.ndarray,
        X_classical: np.ndarray
    ):
        """Train the stacking meta-learner using base model predictions."""
        # Get predictions from base models
        quantum_probs = self._predict_quantum(X_quantum)
        classical_probs = self._predict_classical(X_classical)
        
        # Create meta-features
        meta_features = np.column_stack([quantum_probs, classical_probs])
        
        # Fit the stacking model
        self.stacking_model.fit(meta_features, y_train)
    
    def _predict_quantum(self, X_quantum: np.ndarray) -> np.ndarray:
        """Get predictions from quantum model."""
        if not self.quantum_model or not self.quantum_model.is_fitted:
            raise ValueError("Quantum model not fitted or not provided")
        
        if hasattr(self.quantum_model, 'predict_proba'):
            return self.quantum_model.predict_proba(X_quantum)[:, 1]
        else:
            # Fallback to decision function if predict_proba not available
            if hasattr(self.quantum_model, 'model') and hasattr(self.quantum_model.model, 'decision_function'):
                scores = self.quantum_model.model.decision_function(X_quantum)
                # Convert to probabilities using sigmoid
                return 1.0 / (1.0 + np.exp(-scores))
            else:
                # Use hard predictions as fallback
                predictions = self.quantum_model.predict(X_quantum)
                return predictions.astype(float)
    
    def _predict_classical(self, X_classical: np.ndarray) -> np.ndarray:
        """Get predictions from classical model."""
        if not self.classical_model or not hasattr(self.classical_model, 'model') or not self.classical_model.model:
            raise ValueError("Classical model not fitted or not provided")
        
        # Apply scaling if needed
        if hasattr(self.classical_model, 'scaler') and self.classical_model.scaler:
            X_scaled = self.classical_model.scaler.transform(X_classical)
        else:
            X_scaled = self.scaler.transform(X_classical)
        
        if hasattr(self.classical_model.model, 'predict_proba'):
            return self.classical_model.model.predict_proba(X_scaled)[:, 1]
        else:
            # Fallback to decision function if predict_proba not available
            if hasattr(self.classical_model.model, 'decision_function'):
                scores = self.classical_model.model.decision_function(X_scaled)
                # Convert to probabilities using sigmoid
                return 1.0 / (1.0 + np.exp(-scores))
            else:
                # Use hard predictions as fallback
                predictions = self.classical_model.model.predict(X_scaled)
                return predictions.astype(float)
    
    def predict_proba(self, X: np.ndarray, X_quantum: Optional[np.ndarray] = None, X_classical: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get probability predictions from the ensemble.
        
        Args:
            X: Input features
            X_quantum: Quantum-specific features (if different from X)
            X_classical: Classical-specific features (if different from X)
        
        Returns:
            Array of probabilities for positive class
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Use provided features or default to X
        X_quantum = X_quantum if X_quantum is not None else X
        X_classical = X_classical if X_classical is not None else X
        
        if self.ensemble_method == "weighted_average":
            return self._weighted_average_predict(X_quantum, X_classical)
        elif self.ensemble_method == "voting":
            return self._voting_predict(X_quantum, X_classical)
        elif self.ensemble_method == "stacking":
            return self._stacking_predict(X, X_quantum, X_classical)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _weighted_average_predict(self, X_quantum: np.ndarray, X_classical: np.ndarray) -> np.ndarray:
        """Combine predictions using weighted average."""
        quantum_probs = self._predict_quantum(X_quantum)
        classical_probs = self._predict_classical(X_classical)
        
        q_weight = self.weights.get("quantum", 0.5)
        c_weight = self.weights.get("classical", 0.5)
        
        # Normalize weights
        total_weight = q_weight + c_weight
        if total_weight > 0:
            q_weight /= total_weight
            c_weight /= total_weight
        
        ensemble_probs = q_weight * quantum_probs + c_weight * classical_probs
        return ensemble_probs
    
    def _voting_predict(self, X_quantum: np.ndarray, X_classical: np.ndarray) -> np.ndarray:
        """Combine predictions using voting."""
        quantum_probs = self._predict_quantum(X_quantum)
        classical_probs = self._predict_classical(X_classical)
        
        # For soft voting, average the probabilities
        ensemble_probs = (quantum_probs + classical_probs) / 2.0
        return ensemble_probs
    
    def _stacking_predict(self, X: np.ndarray, X_quantum: np.ndarray, X_classical: np.ndarray) -> np.ndarray:
        """Combine predictions using stacking."""
        quantum_probs = self._predict_quantum(X_quantum)
        classical_probs = self._predict_classical(X_classical)
        
        # Create meta-features
        meta_features = np.column_stack([quantum_probs, classical_probs])
        
        # Get final prediction from stacking model
        if hasattr(self.stacking_model, 'predict_proba'):
            return self.stacking_model.predict_proba(meta_features)[:, 1]
        else:
            # Use decision function and convert to probabilities
            scores = self.stacking_model.decision_function(meta_features)
            return 1.0 / (1.0 + np.exp(-scores))
    
    def predict(self, X: np.ndarray, X_quantum: Optional[np.ndarray] = None, X_classical: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get binary predictions from the ensemble.
        
        Args:
            X: Input features
            X_quantum: Quantum-specific features (if different from X)
            X_classical: Classical-specific features (if different from X)
        
        Returns:
            Array of binary predictions
        """
        probas = self.predict_proba(X, X_quantum, X_classical)
        return (probas >= 0.5).astype(int)
    
    def evaluate_ensemble_diversity(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        X_quantum: Optional[np.ndarray] = None,
        X_classical: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate the diversity of the ensemble members.
        
        Args:
            X_test: Test features
            y_test: Test labels
            X_quantum: Quantum-specific test features
            X_classical: Classical-specific test features
            
        Returns:
            Dictionary with diversity metrics
        """
        X_quantum = X_quantum if X_quantum is not None else X_test
        X_classical = X_classical if X_classical is not None else X_test
        
        # Get predictions from individual models
        quantum_probs = self._predict_quantum(X_quantum)
        classical_probs = self._predict_classical(X_classical)
        
        # Calculate correlations between model predictions
        correlation = np.corrcoef(quantum_probs, classical_probs)[0, 1]
        
        # Calculate disagreement rate
        quantum_preds = (quantum_probs >= 0.5).astype(int)
        classical_preds = (classical_probs >= 0.5).astype(int)
        disagreement_rate = np.mean(quantum_preds != classical_preds)
        
        # Calculate individual model performances
        quantum_ap = average_precision_score(y_test, quantum_probs)
        classical_ap = average_precision_score(y_test, classical_probs)
        
        # Calculate ensemble performance
        ensemble_probs = self.predict_proba(X_test, X_quantum, X_classical)
        ensemble_ap = average_precision_score(y_test, ensemble_probs)
        
        return {
            "correlation": correlation,
            "disagreement_rate": disagreement_rate,
            "quantum_ap": quantum_ap,
            "classical_ap": classical_ap,
            "ensemble_ap": ensemble_ap,
            "ensemble_improvement": ensemble_ap - max(quantum_ap, classical_ap)
        }


def create_optimized_quantum_classical_ensemble(
    quantum_config: Optional[Dict] = None,
    classical_config: Optional[Dict] = None,
    ensemble_method: str = "weighted_average",
    random_state: int = 42
) -> QuantumClassicalEnsemble:
    """
    Factory function to create an optimized quantum-classical ensemble.
    
    Args:
        quantum_config: Configuration for quantum model
        classical_config: Configuration for classical model
        ensemble_method: Method for combining predictions
        random_state: Random state for reproducibility
        
    Returns:
        Optimized QuantumClassicalEnsemble
    """
    # Default quantum model configuration
    if quantum_config is None:
        quantum_config = {
            "model_type": "QSVC",
            "encoding_method": "feature_map",
            "num_qubits": 5,
            "feature_map_type": "ZZ",
            "feature_map_reps": 2,
            "random_state": random_state
        }
    
    # Default classical model configuration
    if classical_config is None:
        classical_config = {
            "model_type": "LogisticRegression",
            "random_state": random_state
        }
    
    # Create models
    quantum_model = QMLLinkPredictor(**quantum_config)
    classical_model = ClassicalLinkPredictor(**classical_config)
    
    # Create ensemble
    ensemble = QuantumClassicalEnsemble(
        quantum_model=quantum_model,
        classical_model=classical_model,
        ensemble_method=ensemble_method,
        random_state=random_state
    )
    
    return ensemble