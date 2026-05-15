from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Namespace inference logic adapted from scripts/build_node_metadata_stub.py
def _node_type(node_id: str) -> str:
    return node_id.split("::", 1)[0] if "::" in node_id else "Unknown"


def infer_namespace_and_url(node_id: str) -> Tuple[str, str]:
    """Return (namespace, resolver_url) for a Hetionet node ID without API calls."""
    t = _node_type(node_id)
    local_id = node_id.split("::", 1)[1] if "::" in node_id else node_id

    if t == "Compound" and local_id.startswith("DB"):
        return "DrugBank", f"https://go.drugbank.com/drugs/{local_id}"
    if t == "Disease" and local_id.startswith("DOID:"):
        return "DOID", f"https://disease-ontology.org/?id={local_id}"
    if t == "Gene" and local_id.isdigit():
        return "Entrez", f"https://www.ncbi.nlm.nih.gov/gene/{local_id}"
    if t == "Anatomy" and local_id.startswith("UBERON:"):
        uberon_key = local_id.replace(":", "_")
        return "UBERON", f"https://www.ebi.ac.uk/ols/ontologies/uberon/terms?iri=http://purl.obolibrary.org/obo/{uberon_key}"
    if t in ("Biological Process", "Molecular Function", "Cellular Component") and local_id.startswith("GO:"):
        return "GO", f"https://www.ebi.ac.uk/QuickGO/term/{local_id}"
    if t == "Side Effect" and local_id.startswith("C"):
        return "UMLS", f"https://uts.nlm.nih.gov/uts/umls/concept/{local_id}"
    if t == "Symptom" and local_id.startswith("D"):
        return "MeSH", f"https://meshb.nlm.nih.gov/record/ui?ui={local_id}"
    if t == "Pathway":
        return "Pathway", ""
    if t == "Pharmacologic Class" and local_id.startswith("N"):
        return "NDF-RT", ""
    return "", ""


class HetionetResolver:
    """
    Static entity resolver backed by the Hetionet nodes TSV.

    Builds three lookup tables on first call to load():
      - name_to_id: lowercased human-readable name → Hetionet node ID
      - id_to_node: Hetionet node ID → (name, kind) tuple
      - external_to_id: DrugBank/DOID/Entrez local ID → Hetionet node ID

    No network calls; all lookups are O(1) dict lookups after loading.
    """

    def __init__(
        self,
        nodes_tsv: str = "data/hetionet-v1.0-nodes.tsv",
        extra_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        self.nodes_tsv = Path(nodes_tsv)
        self._extra_aliases: Dict[str, str] = extra_aliases or {}

        # populated by load()
        self._name_to_id: Dict[str, str] = {}
        self._id_to_node: Dict[str, Tuple[str, str]] = {}  # id → (name, kind)
        self._external_to_id: Dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "HetionetResolver":
        """Parse nodes TSV and build lookup tables. Idempotent."""
        if self._loaded:
            return self

        if not self.nodes_tsv.exists():
            logger.warning(
                f"Hetionet nodes file not found at {self.nodes_tsv}; "
                "resolver will only use extra_aliases."
            )
        else:
            self._parse_tsv()

        # Seed with extra aliases (overrides TSV if same key)
        for alias, node_id in self._extra_aliases.items():
            self._name_to_id[alias.lower().strip()] = node_id

        self._loaded = True
        logger.info(
            f"HetionetResolver loaded: {len(self._id_to_node)} nodes, "
            f"{len(self._name_to_id)} name lookups, "
            f"{len(self._external_to_id)} external-ID lookups."
        )
        return self

    def _parse_tsv(self) -> None:
        with self.nodes_tsv.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                node_id = row.get("id", "").strip()
                name = row.get("name", "").strip()
                kind = row.get("kind", "").strip()
                if not node_id:
                    continue

                self._id_to_node[node_id] = (name, kind)

                if name:
                    self._name_to_id[name.lower()] = node_id

                # Build external-ID index from the local part of the Hetionet ID
                local_id = node_id.split("::", 1)[1] if "::" in node_id else ""
                if local_id:
                    self._external_to_id[local_id.upper()] = node_id

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve_name(self, name: str) -> Optional[str]:
        """Return Hetionet node ID for a human-readable name, or None."""
        if not self._loaded:
            self.load()
        return self._name_to_id.get(name.lower().strip())

    def resolve_external_id(self, external_id: str) -> Optional[str]:
        """Return Hetionet node ID for a DrugBank/DOID/Entrez local ID, or None."""
        if not self._loaded:
            self.load()
        return self._external_to_id.get(external_id.upper().strip())

    def resolve(self, query: str) -> Optional[str]:
        """
        Try name lookup first, then external-ID lookup.
        Returns Hetionet node ID or None if unresolved.
        """
        return self.resolve_name(query) or self.resolve_external_id(query)

    def get_name(self, node_id: str) -> Optional[str]:
        if not self._loaded:
            self.load()
        entry = self._id_to_node.get(node_id)
        return entry[0] if entry else None

    def get_kind(self, node_id: str) -> Optional[str]:
        if not self._loaded:
            self.load()
        entry = self._id_to_node.get(node_id)
        return entry[1] if entry else None

    def list_by_kind(self, kind: str) -> List[str]:
        """Return all Hetionet node IDs of a given kind (e.g. 'Compound')."""
        if not self._loaded:
            self.load()
        return [nid for nid, (_, k) in self._id_to_node.items() if k == kind]
