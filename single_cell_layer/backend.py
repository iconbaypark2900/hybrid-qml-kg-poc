from __future__ import annotations

import logging
from typing import Dict, Optional

from single_cell_layer.device_detection import BackendTag, detect_backend

logger = logging.getLogger(__name__)


def get_backend(config: Optional[Dict] = None):
    """
    Return the active backend module (cpu_scanpy or gpu_rapids).

    Dispatches based on config["single_cell"]["backend"]:
      - "auto"  → detect_backend() picks the best available
      - "gpu"   → force gpu_rapids (raises if unavailable)
      - "cpu"   → force cpu_scanpy

    Returns the backend module, which exposes:
      - preprocess(adata, config) → AnnData
      - run_pca(adata, n_pcs) → AnnData
      - run_neighbors(adata, n_neighbors) → AnnData
      - run_umap(adata) → AnnData
      - run_leiden(adata, resolution) → AnnData
    """
    sc_cfg = (config or {}).get("single_cell", {})
    mode: str = sc_cfg.get("backend", "auto").lower()
    prefer_gpu: bool = sc_cfg.get("prefer_gpu", True)
    fallback_to_cpu: bool = sc_cfg.get("fallback_to_cpu", True)

    if mode == "gpu":
        return _load_gpu(fallback_to_cpu=fallback_to_cpu)
    if mode == "cpu":
        return _load_cpu()

    # auto
    tag: BackendTag = detect_backend(prefer_gpu=prefer_gpu)
    if tag == "gpu_rapids":
        return _load_gpu(fallback_to_cpu=fallback_to_cpu)
    return _load_cpu()


def _load_gpu(fallback_to_cpu: bool = True):
    try:
        from single_cell_layer import gpu_rapids_backend
        return gpu_rapids_backend
    except ImportError as e:
        if fallback_to_cpu:
            logger.warning(f"GPU backend unavailable ({e}); falling back to cpu_scanpy.")
            return _load_cpu()
        raise


def _load_cpu():
    from single_cell_layer import cpu_scanpy_backend
    return cpu_scanpy_backend
