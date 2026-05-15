from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from entity_resolution.hetionet_resolver import HetionetResolver

logger = logging.getLogger(__name__)

# Common external-format DOID patterns
_DOID_BARE = re.compile(r"^(\d+)$")            # e.g. "9351"
_DOID_PREFIXED = re.compile(r"^DOID:(\d+)$", re.IGNORECASE)   # e.g. "DOID:9351"


def _normalise_doid(query: str) -> Optional[str]:
    """Return Hetionet Disease node ID from a DOID-style string, or None."""
    q = query.strip()
    m = _DOID_PREFIXED.match(q)
    if m:
        return f"Disease::DOID:{m.group(1)}"
    m = _DOID_BARE.match(q)
    if m:
        return f"Disease::DOID:{m.group(1)}"
    return None


class DiseaseMapper:
    """
    Map disease identifiers to Hetionet Disease node IDs.

    Hetionet Disease nodes use Disease Ontology IDs:
        Disease::DOID:9351  (type 1 diabetes mellitus)

    Resolution order:
      1. DOID integer or "DOID:NNNN" format → direct construction
      2. Human-readable name lookup in HetionetResolver
      3. Exact external-ID lookup
    """

    def __init__(self, resolver: Optional[HetionetResolver] = None) -> None:
        self._resolver = resolver or HetionetResolver()
        self._cache: Dict[str, Optional[str]] = {}

    def _ensure_loaded(self) -> None:
        if not self._resolver._loaded:
            self._resolver.load()

    def map(self, query: str) -> Optional[str]:
        """
        Resolve a disease identifier (DOID, name, synonym) to a Hetionet node ID,
        or None if unresolved.
        """
        self._ensure_loaded()
        q = query.strip()

        if q in self._cache:
            return self._cache[q]

        # Try DOID numeric format
        candidate = _normalise_doid(q)
        if candidate and candidate in self._resolver._id_to_node:
            self._cache[q] = candidate
            return candidate

        # Try name lookup
        result = self._resolver.resolve_name(q)
        if result and result.startswith("Disease::"):
            self._cache[q] = result
            return result

        # Try external-ID lookup
        result = self._resolver.resolve_external_id(q)
        if result and result.startswith("Disease::"):
            self._cache[q] = result
            return result

        logger.debug(f"Disease '{q}' could not be resolved to a Hetionet node.")
        self._cache[q] = None
        return None

    def map_many(self, queries: List[str]) -> Dict[str, Optional[str]]:
        return {q: self.map(q) for q in queries}

    def filter_resolved(self, queries: List[str]) -> List[str]:
        return [q for q in queries if self.map(q) is not None]
