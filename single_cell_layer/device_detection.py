from __future__ import annotations

import logging
from typing import Literal

logger = logging.getLogger(__name__)

BackendTag = Literal["gpu_rapids", "cpu_scanpy"]


def detect_backend(prefer_gpu: bool = True) -> BackendTag:
    """
    Detect available compute backend for single-cell processing.

    Returns 'gpu_rapids' if RAPIDS-singlecell is importable and a CUDA
    device is available, otherwise 'cpu_scanpy'.
    """
    if prefer_gpu:
        if _rapids_available():
            logger.info("Backend: gpu_rapids (RAPIDS-singlecell)")
            return "gpu_rapids"
        logger.info("RAPIDS not available; falling back to cpu_scanpy.")

    logger.info("Backend: cpu_scanpy")
    return "cpu_scanpy"


def _rapids_available() -> bool:
    try:
        import rapids_singlecell  # noqa: F401
        import cupy               # noqa: F401
        return True
    except ImportError:
        return False


def _cuda_available() -> bool:
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


def backend_info() -> dict:
    """Return a dict summarising the detected backend and library versions."""
    info: dict = {}

    # Scanpy (always attempted)
    try:
        import scanpy as sc
        info["scanpy"] = sc.__version__
    except ImportError:
        info["scanpy"] = None

    # RAPIDS
    try:
        import rapids_singlecell as rsc
        info["rapids_singlecell"] = rsc.__version__
    except ImportError:
        info["rapids_singlecell"] = None

    try:
        import cupy as cp
        info["cupy"] = cp.__version__
        info["cuda_devices"] = cp.cuda.runtime.getDeviceCount()
    except Exception:
        info["cupy"] = None
        info["cuda_devices"] = 0

    info["recommended_backend"] = detect_backend()
    return info
