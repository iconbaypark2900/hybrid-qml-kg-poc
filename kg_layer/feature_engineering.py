"""
Feature engineering utilities for quantum machine learning.
Provides various encoding strategies for transforming entity embeddings into QML-ready features.
"""

import numpy as np
from typing import Callable, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sk_normalize


def make_qml_features(
    h: np.ndarray,
    t: np.ndarray,
    strategy: str = "diff",
    qml_dim: int = 5,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Create QML-ready features from head and tail embeddings.
    
    Args:
        h: Head entity embedding vector
        t: Tail entity embedding vector
        strategy: Feature encoding strategy
            - "diff": |h - t|
            - "hadamard": h ⊙ t (element-wise product)
            - "concat": concatenate([h, t]) then truncate/project to qml_dim
            - "diff_prod": concatenate([|h-t|, h⊙t]) then project to qml_dim
            - "poly": polynomial features (degree 2 interactions)
        qml_dim: Target dimension for QML features
        normalize: Optional normalization strategy ('l2', 'minmax', 'zscore', 'tanh')
    
    Returns:
        Feature vector of shape (qml_dim,)
    """
    # Normalize inputs if requested
    if normalize == 'l2':
        h = h / (np.linalg.norm(h) + 1e-9)
        t = t / (np.linalg.norm(t) + 1e-9)
    elif normalize == 'minmax':
        h_min, h_max = h.min(), h.max()
        t_min, t_max = t.min(), t.max()
        if h_max > h_min:
            h = (h - h_min) / (h_max - h_min)
        if t_max > t_min:
            t = (t - t_min) / (t_max - t_min)
    elif normalize == 'zscore':
        h_mean, h_std = h.mean(), h.std()
        t_mean, t_std = t.mean(), t.std()
        if h_std > 0:
            h = (h - h_mean) / h_std
        if t_std > 0:
            t = (t - t_mean) / t_std
    elif normalize == 'tanh':
        h = np.tanh(h)
        t = np.tanh(t)
    
    # Apply encoding strategy
    if strategy == "diff":
        feat = np.abs(h - t)
    elif strategy == "hadamard":
        feat = h * t
    elif strategy == "concat":
        feat = np.concatenate([h, t])
    elif strategy == "diff_prod":
        feat = np.concatenate([np.abs(h - t), h * t])
    elif strategy == "poly":
        # Polynomial features (degree 2): include h, t, h*t, h^2, t^2
        feat = np.concatenate([
            h[:min(3, len(h))],
            t[:min(3, len(t))],
            (h * t)[:min(3, len(h))],
            (h ** 2)[:min(3, len(h))],
            (t ** 2)[:min(3, len(t))]
        ])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Project to qml_dim if needed
    if len(feat) > qml_dim:
        # Use PCA to reduce dimension
        # For single vector, we'll pad and use a pre-fit PCA or truncate
        # In practice, this should be fitted on training data
        if len(feat) > qml_dim:
            # Simple truncation (better: use PCA fitted on training set)
            feat = feat[:qml_dim]
    elif len(feat) < qml_dim:
        # Pad with zeros
        feat = np.pad(feat, (0, qml_dim - len(feat)), mode='constant')
    
    return feat[:qml_dim].astype(np.float32)


def polynomial_features(h: np.ndarray, t: np.ndarray, degree: int = 2) -> np.ndarray:
    """
    Generate polynomial features up to specified degree.
    
    Args:
        h: Head embedding
        t: Tail embedding
        degree: Maximum polynomial degree
    
    Returns:
        Polynomial feature vector
    """
    features = []
    
    if degree >= 1:
        features.extend([h, t])
    if degree >= 2:
        features.extend([h * t, h ** 2, t ** 2])
    if degree >= 3:
        features.extend([h ** 3, t ** 3, (h ** 2) * t, h * (t ** 2)])
    
    return np.concatenate(features) if features else np.array([])

