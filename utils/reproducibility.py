"""
Reproducibility utilities for centralizing random seed management.

This module ensures all random number generators (RNGs) across the project
are properly seeded for reproducible results.
"""

import os
import random
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int = 42):
    """
    Set seed for all random number generators used in the project.

    This includes:
    - Python's built-in random module
    - NumPy
    - PyTorch (if available)
    - Qiskit (if available)

    Args:
        seed: Random seed value
    """
    logger.info(f"Setting global random seed to {seed}")

    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # For deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("PyTorch seed set")
    except ImportError:
        pass

    # Qiskit (if available)
    try:
        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = seed
        logger.info("Qiskit seed set")
    except ImportError:
        pass

    # Set environment variables for additional libraries
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info(f"✓ All RNGs seeded with {seed}")


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Get a NumPy random number generator with optional seed.

    Args:
        seed: Optional seed value. If None, uses default RNG.

    Returns:
        NumPy random Generator
    """
    if seed is not None:
        return np.random.default_rng(seed)
    return np.random.default_rng()
