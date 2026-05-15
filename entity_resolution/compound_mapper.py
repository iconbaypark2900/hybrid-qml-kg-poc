from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from entity_resolution.hetionet_resolver import HetionetResolver

logger = logging.getLogger(__name__)

_DB_ID = re.compile(r"^(DB\d{5})$", re.IGNORECASE)   # DrugBank ID e.g. DB00945


class CompoundMapper:
    """
    Map compound identifiers to Hetionet Compound node IDs.

    Hetionet Compound nodes use DrugBank IDs:
        Compound::DB00945  (aspirin)

    Resolution order:
      1. DrugBank ID (DB00000 pattern) → direct construction
      2. Human-readable / brand name lookup in HetionetResolver
      3. External-ID lookup
    """

    def __init__(self, resolver: Optional[HetionetResolver] = None) -> None:
        self._resolver = resolver or HetionetResolver()
        self._cache: Dict[str, Optional[str]] = {}

    def _ensure_loaded(self) -> None:
        if not self._resolver._loaded:
            self._resolver.load()

    def map(self, query: str) -> Optional[str]:
        """
        Resolve a compound identifier (DrugBank ID, INN name, brand name)
        to a Hetionet node ID, or None if unresolved.
        """
        self._ensure_loaded()
        q = query.strip()

        if q in self._cache:
            return self._cache[q]

        # Try DrugBank format
        m = _DB_ID.match(q)
        if m:
            candidate = f"Compound::{m.group(1).upper()}"
            if candidate in self._resolver._id_to_node:
                self._cache[q] = candidate
                return candidate
            # ID pattern matches but not in graph
            logger.debug(f"DrugBank ID {q} not in Hetionet compound nodes.")

        # Try name / brand name lookup
        result = self._resolver.resolve_name(q)
        if result and result.startswith("Compound::"):
            self._cache[q] = result
            return result

        # Try external-ID lookup
        result = self._resolver.resolve_external_id(q)
        if result and result.startswith("Compound::"):
            self._cache[q] = result
            return result

        logger.debug(f"Compound '{q}' could not be resolved to a Hetionet node.")
        self._cache[q] = None
        return None

    def map_many(self, queries: List[str]) -> Dict[str, Optional[str]]:
        return {q: self.map(q) for q in queries}

    def filter_resolved(self, queries: List[str]) -> List[str]:
        return [q for q in queries if self.map(q) is not None]
