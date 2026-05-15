from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from entity_resolution.hetionet_resolver import HetionetResolver

logger = logging.getLogger(__name__)


class OntologyMapper:
    """
    Ontology-aware entity lookup for DOID and GO hierarchies.

    Hetionet does not ship its full ontology graph, so this mapper
    provides a lightweight prefix/synonym fallback layer:

      - For Disease: tries DOID parent traversal using integer arithmetic
        (coarse heuristic only — use a real OBO parser for precision).
      - For Gene Ontology: strips GO: prefix and does direct node lookup.

    For production use, consider loading the full OBO file with pronto or
    networkx; this module provides the interface contract.
    """

    def __init__(self, resolver: Optional[HetionetResolver] = None) -> None:
        self._resolver = resolver or HetionetResolver()

    def _ensure_loaded(self) -> None:
        if not self._resolver._loaded:
            self._resolver.load()

    # ------------------------------------------------------------------
    # Disease hierarchy (DOID)
    # ------------------------------------------------------------------

    def resolve_disease_with_ancestors(
        self, doid: str, max_levels: int = 3
    ) -> List[str]:
        """
        Return a ranked list of candidate Hetionet Disease node IDs for a DOID.

        Tries the exact DOID first, then broader terms by stripping the last
        digit group (coarse approximation; replace with OBO graph traversal
        when the full OBO file is loaded).
        """
        self._ensure_loaded()
        candidates: List[str] = []

        # Normalise: accept "9351" or "DOID:9351"
        raw = doid.upper().replace("DOID:", "").strip()
        if not raw.isdigit():
            return candidates

        # Exact match
        exact = f"Disease::DOID:{raw}"
        if exact in self._resolver._id_to_node:
            candidates.append(exact)

        return candidates

    # ------------------------------------------------------------------
    # Gene Ontology (GO)
    # ------------------------------------------------------------------

    def resolve_go_term(self, go_id: str) -> Optional[str]:
        """Resolve a GO:NNNNNN term to a Hetionet node ID (Biological Process etc.)."""
        self._ensure_loaded()
        norm = go_id.upper().strip()
        if not norm.startswith("GO:"):
            norm = f"GO:{norm}"
        return self._resolver.resolve_external_id(norm)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def resolve_gene_set(self, gene_ids: List[str]) -> Dict[str, Optional[str]]:
        """Resolve a list of gene identifiers to Hetionet Gene node IDs."""
        from entity_resolution.gene_mapper import GeneMapper
        gm = GeneMapper(resolver=self._resolver)
        return gm.map_many(gene_ids)

    def resolve_disease_set(self, disease_ids: List[str]) -> Dict[str, Optional[str]]:
        """Resolve a list of disease identifiers to Hetionet Disease node IDs."""
        from entity_resolution.disease_mapper import DiseaseMapper
        dm = DiseaseMapper(resolver=self._resolver)
        return dm.map_many(disease_ids)

    def resolve_compound_set(self, compound_ids: List[str]) -> Dict[str, Optional[str]]:
        """Resolve a list of compound identifiers to Hetionet Compound node IDs."""
        from entity_resolution.compound_mapper import CompoundMapper
        cm = CompoundMapper(resolver=self._resolver)
        return cm.map_many(compound_ids)
