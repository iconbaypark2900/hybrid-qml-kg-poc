from __future__ import annotations

import logging
from typing import Dict, List, Optional

from entity_resolution.hetionet_resolver import HetionetResolver

logger = logging.getLogger(__name__)


class GeneMapper:
    """
    Map gene identifiers to Hetionet Gene node IDs.

    Hetionet Gene nodes use Entrez (NCBI) integer IDs:
        Gene::1956  (EGFR)

    Resolution order for each query:
      1. Direct Entrez integer → "Gene::<id>"  (if query is all digits)
      2. HGNC symbol lookup via HetionetResolver name table
      3. Fallback external-ID lookup (uppercase strip)
    """

    def __init__(self, resolver: Optional[HetionetResolver] = None) -> None:
        self._resolver = resolver or HetionetResolver()
        # symbol → entrez from resolver name table; populated lazily
        self._symbol_cache: Dict[str, Optional[str]] = {}

    def _ensure_loaded(self) -> None:
        if not self._resolver._loaded:
            self._resolver.load()

    def map(self, query: str) -> Optional[str]:
        """
        Resolve a gene identifier (HGNC symbol or Entrez integer string)
        to a Hetionet node ID like 'Gene::1956', or None if unresolved.
        """
        self._ensure_loaded()
        q = query.strip()

        # Fast path: pure Entrez integer
        if q.isdigit():
            candidate = f"Gene::{q}"
            if candidate in self._resolver._id_to_node:
                return candidate
            logger.debug(f"Entrez ID {q} not found in Hetionet nodes.")
            return None

        # Check symbol cache
        if q in self._symbol_cache:
            return self._symbol_cache[q]

        # Try name lookup (case-insensitive in resolver)
        result = self._resolver.resolve_name(q)
        if result and result.startswith("Gene::"):
            self._symbol_cache[q] = result
            return result

        # Try external-ID lookup
        result = self._resolver.resolve_external_id(q)
        if result and result.startswith("Gene::"):
            self._symbol_cache[q] = result
            return result

        logger.debug(f"Gene '{q}' could not be resolved to a Hetionet node.")
        self._symbol_cache[q] = None
        return None

    def map_many(self, queries: List[str]) -> Dict[str, Optional[str]]:
        """Batch resolve; returns {query: hetionet_id_or_None}."""
        return {q: self.map(q) for q in queries}

    def filter_resolved(self, queries: List[str]) -> List[str]:
        """Return only queries that successfully resolve."""
        return [q for q in queries if self.map(q) is not None]
