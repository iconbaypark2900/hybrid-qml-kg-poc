from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """
    In-memory registry that tracks loaded AnnData datasets by name.

    Used by the pipeline to avoid reloading the same dataset multiple times
    and to expose a consistent reference across pipeline stages.
    """

    def __init__(self) -> None:
        self._store: Dict[str, object] = {}  # name → AnnData
        self._meta: Dict[str, Dict] = {}     # name → metadata dict

    def register(self, name: str, adata, source_path: str = "", notes: str = "") -> None:
        """Register an AnnData object under a logical name."""
        if name in self._store:
            logger.warning(f"Dataset '{name}' already registered; overwriting.")
        self._store[name] = adata
        self._meta[name] = {
            "name": name,
            "source_path": source_path,
            "n_obs": getattr(adata, "n_obs", None),
            "n_vars": getattr(adata, "n_vars", None),
            "notes": notes,
        }
        logger.info(
            f"Registered dataset '{name}': "
            f"{getattr(adata, 'n_obs', '?')} cells × {getattr(adata, 'n_vars', '?')} genes"
        )

    def get(self, name: str):
        """Return AnnData for a registered name, or raise KeyError."""
        if name not in self._store:
            raise KeyError(
                f"Dataset '{name}' not in registry. "
                f"Available: {list(self._store.keys())}"
            )
        return self._store[name]

    def list_datasets(self) -> Dict[str, Dict]:
        """Return metadata dict for all registered datasets."""
        return dict(self._meta)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)
