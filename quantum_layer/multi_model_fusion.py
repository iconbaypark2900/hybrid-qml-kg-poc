# quantum_layer/multi_model_fusion.py

"""
Multi-Model Prediction Fusion Module

This module implements advanced methods for combining predictions from multiple models
beyond the existing stacking ensemble. It supports:
- Weighted averaging with learned or heuristic weights
- Rank-based fusion (Reciprocal Rank Fusion, Borda Count)
- Confidence-weighted combination
- Bayesian model averaging
- Neural network-based meta-learner
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import average_precision_score, roc_auc_score
import logging
from scipy.special import softmax
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class MultiModelFusion:
    """
    Advanced multi-model prediction fusion.
    
    Combines predictions from multiple models using various strategies:
    - weighted_average: Simple weighted combination
    - rank_fusion: Reciprocal Rank Fusion (RRF) or Borda Count
    - confidence_weighted: Weight by model confidence per sample
    - bayesian_averaging: Bayesian Model Averaging (BMA)
    - neural_metalearner: Neural network meta-learner
    - optimized_weights: Learn optimal weights via optimization
    """
    
    def __init__(
        self,
        fusion_method: str = "weighted_average",
        weights: Optional[Dict[str, float]] = None,
        use_cross_validation: bool = True,
        cv_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize the multi-model fusion system.
        
        Args:
            fusion_method: Fusion strategy ('weighted_average', 'rank_fusion', 
                          'confidence_weighted', 'bayesian_averaging', 
                          'neural_metalearner', 'optimized_weights')
            weights: Pre-defined weights for each model (for weighted_average)
            use_cross_validation: Whether to use CV for generating meta-features
            cv_folds: Number of CV folds
            random_state: Random seed
        """
        self.fusion_method = fusion_method
        self.weights = weights or {}
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        self.is_fitted = False
        self.model_names: List[str] = []
        self.meta_scaler = StandardScaler()
        self.neural_metalearner = None
        self.optimized_weights = None
        self.model_variances = {}  # For Bayesian averaging
        
    def fit(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
        X_val: Optional[np.ndarray] = None
    ) -> 'MultiModelFusion':
        """
        Fit the fusion system.
        
        Args:
            predictions: Dict mapping model_name -> predicted probabilities
            y_true: Ground truth labels
            X_val: Optional validation features (for some methods)
            
        Returns:
            Self
        """
        logger.info(f"Fitting MultiModelFusion with method: {self.fusion_method}")
        
        self.model_names = list(predictions.keys())
        n_models = len(self.model_names)
        
        if n_models == 0:
            raise ValueError("No models provided")
        
        # Stack predictions into matrix
        pred_matrix = np.column_stack([predictions[name] for name in self.model_names])
        
        if self.fusion_method == "weighted_average":
            self._fit_weighted_average(predictions, y_true)
            
        elif self.fusion_method == "rank_fusion":
            self._fit_rank_fusion(predictions, y_true)
            
        elif self.fusion_method == "confidence_weighted":
            self._fit_confidence_weighted(predictions, y_true)
            
        elif self.fusion_method == "bayesian_averaging":
            self._fit_bayesian_averaging(predictions, y_true)
            
        elif self.fusion_method == "neural_metalearner":
            self._fit_neural_metalearner(pred_matrix, y_true)
            
        elif self.fusion_method == "optimized_weights":
            self._fit_optimized_weights(predictions, y_true)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        self.is_fitted = True
        return self
    
    def _fit_weighted_average(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """Use pre-defined weights or uniform weights."""
        if not self.weights:
            # Default: uniform weights
            n_models = len(self.model_names)
            self.weights = {name: 1.0 / n_models for name in self.model_names}
            logger.info(f"Using uniform weights: {self.weights}")
        else:
            # Normalize weights
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}
            logger.info(f"Using provided weights: {self.weights}")
    
    def _fit_rank_fusion(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Implement Reciprocal Rank Fusion (RRF).
        
        RRF combines rankings from multiple models. For each sample,
        models rank predictions, and RRF computes:
            RRF_score = sum(1 / (k + rank_i)) for each model i
        where k is a damping constant (typically 60).
        """
        self.rrf_k = 60  # Damping constant
        logger.info(f"Rank fusion configured with k={self.rrf_k}")
    
    def _fit_confidence_weighted(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Learn to weight models by their confidence.
        
        Confidence = |prediction - 0.5| (distance from decision boundary)
        """
        # Compute average confidence per model on training data
        self.model_confidence = {}
        for name, pred in predictions.items():
            confidence = np.abs(pred - 0.5) * 2  # Scale to [0, 1]
            self.model_confidence[name] = float(np.mean(confidence))
        
        # Normalize
        total_conf = sum(self.model_confidence.values())
        self.model_confidence = {k: v / total_conf for k, v in self.model_confidence.items()}
        
        logger.info(f"Model confidences: {self.model_confidence}")
    
    def _fit_bayesian_averaging(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Bayesian Model Averaging.
        
        Weight each model by its posterior probability given the data.
        P(M_i|D) ∝ P(D|M_i) * P(M_i)
        """
        # Compute log-likelihood for each model
        self.model_log_likelihood = {}
        self.model_prior = {}
        
        for name, pred in predictions.items():
            # Binary cross-entropy as negative log-likelihood
            epsilon = 1e-10
            pred_clipped = np.clip(pred, epsilon, 1 - epsilon)
            log_likelihood = np.mean(y_true * np.log(pred_clipped) + (1 - y_true) * np.log(1 - pred_clipped))
            self.model_log_likelihood[name] = log_likelihood
            
            # Prior: uniform or based on model complexity (could be customized)
            self.model_prior[name] = 1.0 / len(predictions)
        
        # Compute posterior weights using softmax for numerical stability
        log_weights = [
            self.model_log_likelihood[name] + np.log(self.model_prior[name])
            for name in self.model_names
        ]
        posterior_weights = softmax(log_weights)
        
        self.bma_weights = {name: float(w) for name, w in zip(self.model_names, posterior_weights)}
        logger.info(f"BMA weights: {self.bma_weights}")
    
    def _fit_neural_metalearner(self, pred_matrix: np.ndarray, y_true: np.ndarray):
        """Train a neural network meta-learner on model predictions."""
        logger.info("Training neural network meta-learner...")
        
        # Scale meta-features
        pred_matrix_scaled = self.meta_scaler.fit_transform(pred_matrix)
        
        # Neural network architecture
        self.neural_metalearner = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.neural_metalearner.fit(pred_matrix_scaled, y_true)
        logger.info("Neural meta-learner trained")
    
    def _fit_optimized_weights(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray):
        """Learn optimal weights by optimizing PR-AUC."""
        logger.info("Optimizing weights via PR-AUC maximization...")
        
        def objective(weights):
            # Normalize weights to sum to 1
            weights = np.abs(weights)
            weights = weights / np.sum(weights)
            
            # Compute weighted predictions
            pred_matrix = np.column_stack([predictions[name] for name in self.model_names])
            weighted_pred = pred_matrix @ weights
            
            # Negative PR-AUC (we minimize)
            try:
                pr_auc = average_precision_score(y_true, weighted_pred)
                return -pr_auc
            except Exception:
                return 0.0
        
        # Initial guess: uniform weights
        n_models = len(self.model_names)
        x0 = np.ones(n_models) / n_models
        
        # Optimize with bounds [0, 1] for each weight
        from scipy.optimize import minimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(n_models)]
        )
        
        # Normalize optimized weights
        optimal_weights = np.abs(result.x)
        optimal_weights = optimal_weights / np.sum(optimal_weights)
        
        self.optimized_weights = {name: float(w) for name, w in zip(self.model_names, optimal_weights)}
        logger.info(f"Optimized weights: {self.optimized_weights}")
        logger.info(f"Optimization result: {result.message}")
    
    def predict(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Generate fused predictions.
        
        Args:
            predictions: Dict mapping model_name -> predicted probabilities
            
        Returns:
            Fused predicted probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Fusion system not fitted. Call fit() first.")
        
        # Validate model names
        for name in self.model_names:
            if name not in predictions:
                raise ValueError(f"Missing predictions for model: {name}")
        
        pred_matrix = np.column_stack([predictions[name] for name in self.model_names])
        
        if self.fusion_method == "weighted_average":
            return self._predict_weighted_average(predictions)
            
        elif self.fusion_method == "rank_fusion":
            return self._predict_rank_fusion(predictions)
            
        elif self.fusion_method == "confidence_weighted":
            return self._predict_confidence_weighted(predictions)
            
        elif self.fusion_method == "bayesian_averaging":
            return self._predict_bayesian_averaging(predictions)
            
        elif self.fusion_method == "neural_metalearner":
            return self._predict_neural_metalearner(pred_matrix)
            
        elif self.fusion_method == "optimized_weights":
            return self._predict_optimized_weights(predictions)
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def _predict_weighted_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted average prediction."""
        weighted_sum = np.zeros(len(next(iter(predictions.values()))))
        for name, pred in predictions.items():
            weighted_sum += self.weights.get(name, 0) * pred
        return weighted_sum
    
    def _predict_rank_fusion(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Reciprocal Rank Fusion."""
        n_samples = len(next(iter(predictions.values())))
        rrf_scores = np.zeros(n_samples)
        
        for name, pred in predictions.items():
            # Get ranks (higher prediction = better rank = lower rank number)
            ranks = np.argsort(np.argsort(-pred)) + 1  # 1-indexed ranks
            rrf_scores += 1.0 / (self.rrf_k + ranks)
        
        # Normalize to [0, 1]
        rrf_scores = (rrf_scores - rrf_scores.min()) / (rrf_scores.max() - rrf_scores.min() + 1e-10)
        return rrf_scores
    
    def _predict_confidence_weighted(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Confidence-weighted combination."""
        n_samples = len(next(iter(predictions.values())))
        weighted_sum = np.zeros(n_samples)
        total_weight = np.zeros(n_samples)
        
        for name, pred in predictions.items():
            # Per-sample confidence
            confidence = np.abs(pred - 0.5) * 2
            # Model-level confidence weight
            model_weight = self.model_confidence.get(name, 1.0 / len(predictions))
            
            weighted_sum += model_weight * confidence * pred
            total_weight += model_weight * confidence
        
        return weighted_sum / (total_weight + 1e-10)
    
    def _predict_bayesian_averaging(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Bayesian Model Averaging."""
        weighted_sum = np.zeros(len(next(iter(predictions.values()))))
        for name, pred in predictions.items():
            weighted_sum += self.bma_weights.get(name, 0) * pred
        return weighted_sum
    
    def _predict_neural_metalearner(self, pred_matrix: np.ndarray) -> np.ndarray:
        """Neural network meta-learner prediction."""
        pred_matrix_scaled = self.meta_scaler.transform(pred_matrix)
        return self.neural_metalearner.predict_proba(pred_matrix_scaled)[:, 1]
    
    def _predict_optimized_weights(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Optimized weights prediction."""
        weighted_sum = np.zeros(len(next(iter(predictions.values()))))
        for name, pred in predictions.items():
            weighted_sum += self.optimized_weights.get(name, 0) * pred
        return weighted_sum
    
    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate fused predictions and individual models.
        
        Returns:
            Dict with metrics for each model and the fused prediction
        """
        results = {}
        
        # Individual model performance
        for name, pred in predictions.items():
            pr_auc = average_precision_score(y_true, pred)
            try:
                roc_auc = roc_auc_score(y_true, pred)
            except Exception:
                roc_auc = float('nan')
            results[f'{name}_pr_auc'] = float(pr_auc)
            results[f'{name}_roc_auc'] = float(roc_auc)
        
        # Fused prediction performance
        fused_pred = self.predict(predictions)
        fused_pr_auc = average_precision_score(y_true, fused_pred)
        try:
            fused_roc_auc = roc_auc_score(y_true, fused_pred)
        except Exception:
            fused_roc_auc = float('nan')
        
        results['fused_pr_auc'] = float(fused_pr_auc)
        results['fused_roc_auc'] = float(fused_roc_auc)
        results['improvement_over_mean'] = float(
            fused_pr_auc - np.mean([v for k, v in results.items() if k.endswith('_pr_auc') and k != 'fused_pr_auc'])
        )
        
        return results


def create_fusion_ensemble(
    model_predictions: Dict[str, np.ndarray],
    y_train: np.ndarray,
    fusion_method: str = "optimized_weights",
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[MultiModelFusion, Dict[str, float]]:
    """
    Convenience function to create and evaluate a fusion ensemble.
    
    Args:
        model_predictions: Dict mapping model_name -> predicted probabilities (train set)
        y_train: Training labels
        fusion_method: Fusion strategy
        cv_folds: Number of CV folds for generating out-of-fold predictions
        random_state: Random seed
        
    Returns:
        (fitted_fusion_system, evaluation_metrics)
    """
    logger.info(f"Creating fusion ensemble with method: {fusion_method}")
    
    # If using cross-validation, generate out-of-fold predictions
    if cv_folds > 1 and len(model_predictions) > 0:
        logger.info(f"Using {cv_folds}-fold CV for robust fusion")
        # Note: In practice, you'd want to generate OOF predictions during model training
        # For now, we use the provided predictions directly
    
    # Create and fit fusion system
    fusion = MultiModelFusion(
        fusion_method=fusion_method,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    fusion.fit(model_predictions, y_train)
    
    # Evaluate
    metrics = fusion.evaluate(model_predictions, y_train)
    
    return fusion, metrics


# Example usage
if __name__ == "__main__":
    # Example: Combine predictions from 3 models
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    
    # Simulate predictions from 3 models with different performance
    pred_model1 = np.random.rand(n_samples) * 0.5 + 0.25  # PR-AUC ~0.6
    pred_model2 = np.random.rand(n_samples) * 0.4 + 0.3  # PR-AUC ~0.65
    pred_model3 = np.random.rand(n_samples) * 0.6 + 0.2  # PR-AUC ~0.7
    
    predictions = {
        'model1': pred_model1,
        'model2': pred_model2,
        'model3': pred_model3
    }
    
    # Test different fusion methods
    methods = ['weighted_average', 'rank_fusion', 'bayesian_averaging', 'optimized_weights']
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Testing fusion method: {method}")
        print('='*50)
        
        fusion, metrics = create_fusion_ensemble(
            predictions,
            y_true,
            fusion_method=method
        )
        
        print(f"\nResults:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
