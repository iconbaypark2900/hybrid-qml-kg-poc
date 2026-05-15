from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PerturbationRegistry:
    """
    In-memory registry of loaded DrugSignature objects, keyed by compound name.
    """

    def __init__(self) -> None:
        self._store: Dict[str, object] = {}  # compound_name → DrugSignature

    def register_many(self, signatures: Dict[str, object]) -> None:
        before = len(self._store)
        self._store.update(signatures)
        logger.info(
            f"Registered {len(signatures)} signatures "
            f"(total: {len(self._store)}, new: {len(self._store) - before})"
        )

    def get(self, compound: str) -> object:
        if compound not in self._store:
            raise KeyError(f"Compound '{compound}' not in registry.")
        return self._store[compound]

    def list_compounds(self) -> list:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, compound: str) -> bool:
        return compound in self._store

    def export_json(self, out_path: str) -> Path:
        """Write all signatures to a JSON file."""
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._store.items()}
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info(f"Registry exported to {p} ({len(data)} compounds)")
        return p
