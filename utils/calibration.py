"""
Probability calibration utilities for improving model confidence estimates.

Calibration ensures that predicted probabilities match empirical frequencies.
For example, among all predictions where the model says "70% positive",
approximately 70% should actually be positive.
"""

import logging
from typing import Optional
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class CalibratedModel:
    """
    Wrapper for any classifier that adds probability calibration.

    Supports two calibration methods:
    1. Platt scaling: Fits a logistic regression on classifier outputs
    2. Isotonic regression: Non-parametric, monotonic calibration
    """

    def __init__(
        self,
        base_model,
        method: str = 'isotonic',
        cv: int = 5
    ):
        """
        Args:
            base_model: Base classifier (must have predict_proba or decision_function)
            method: Calibration method ('isotonic' or 'sigmoid' for Platt scaling)
            cv: Number of CV folds for calibration fitting
        """
        self.base_model = base_model
        self.method = method
        self.cv = cv
        self.calibrated_model = None

    def fit(self, X, y):
        """Fit base model and calibration."""
        logger.info(f"Fitting model with {self.method} calibration...")

        # Use sklearn's built-in calibrated classifier
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            method=self.method,
            cv=self.cv
        )
        self.calibrated_model.fit(X, y)

        logger.info("Calibration complete")
        return self

    def predict(self, X):
        """Predict class labels."""
        if self.calibrated_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.calibrated_model.predict(X)

    def predict_proba(self, X):
        """Predict calibrated probabilities."""
        if self.calibrated_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self.calibrated_model.predict_proba(X)

    def decision_function(self, X):
        """Decision function (for compatibility)."""
        if self.calibrated_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Return calibrated probabilities as decision scores
        proba = self.calibrated_model.predict_proba(X)
        return proba[:, 1]  # Probability of positive class


def calibrate_probabilities(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    method: str = 'isotonic'
) -> np.ndarray:
    """
    Calibrate probability scores post-hoc.

    Args:
        y_true: True labels (0/1)
        y_scores: Uncalibrated scores/probabilities
        method: 'isotonic' or 'platt' (Platt scaling)

    Returns:
        Calibrated probabilities
    """
    if method == 'isotonic':
        calibrator = IsotonicRegression(out_of_bounds='clip')
    elif method == 'platt':
        # Platt scaling: fit logistic regression
        calibrator = LogisticRegression()
        y_scores = y_scores.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    calibrator.fit(y_scores, y_true)

    if method == 'platt':
        calibrated = calibrator.predict_proba(y_scores)[:, 1]
    else:
        calibrated = calibrator.predict(y_scores)

    logger.info(f"Calibrated {len(y_scores)} scores using {method}")
    return calibrated


def evaluate_calibration(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10):
    """
    Evaluate calibration quality using Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and empirical accuracy.

    Args:
        y_true: True labels (0/1)
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dict with calibration metrics
    """
    from sklearn.calibration import calibration_curve

    # Compute calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )

    # Expected Calibration Error (ECE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bin_edges[1:-1])

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            bin_size = mask.sum() / len(y_true)
            ece += bin_size * np.abs(bin_acc - bin_conf)

    # Maximum Calibration Error (MCE)
    mce = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_proba[mask].mean()
            mce = max(mce, np.abs(bin_acc - bin_conf))

    results = {
        'ece': ece,  # Expected Calibration Error
        'mce': mce,  # Maximum Calibration Error
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value
    }

    logger.info(f"Calibration evaluation: ECE={ece:.4f}, MCE={mce:.4f}")
    return results


def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str = "Model"):
    """
    Plot calibration curve (reliability diagram).

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        model_name: Name for plot title

    Returns:
        matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, marker='o', label=model_name)

        # Plot perfectly calibrated line
        ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return fig
    except ImportError:
        logger.warning("matplotlib not available, skipping calibration plot")
        return None


# Example usage
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    # Generate example data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Train uncalibrated model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X[:800], y[:800])

    # Get uncalibrated probabilities
    y_proba_uncal = rf.predict_proba(X[800:])[:, 1]

    # Calibrate using wrapper
    calibrated_rf = CalibratedModel(rf, method='isotonic')
    calibrated_rf.fit(X[:800], y[:800])
    y_proba_cal = calibrated_rf.predict_proba(X[800:])[:, 1]

    # Evaluate calibration
    print("Uncalibrated:")
    evaluate_calibration(y[800:], y_proba_uncal)

    print("\nCalibrated:")
    evaluate_calibration(y[800:], y_proba_cal)
