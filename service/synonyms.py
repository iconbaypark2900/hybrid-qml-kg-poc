"""Curated synonym table for human-friendly entity resolution.

Layered loader:
  1. Hardcoded high-confidence aliases for the entities the demo dataset
     actually contains (DB00178, DB00997, etc. + DOID:14330, etc.). This
     ships in the codebase so the service has *some* synonym support
     without external data files.
  2. Optional `data/drugbank_synonyms.tsv` (TSV: drugbank_id<TAB>synonym)
     loaded if present.
  3. Optional `data/doid_synonyms.tsv` (TSV: doid<TAB>synonym) loaded if
     present.

The resolver still rejects unknown ids. This table just expands what
counts as a known id by mapping aliases → canonical IDs upfront.

Matches are case-insensitive on the alias side. Canonical IDs are returned
unchanged ("DB00178", "DOID:14330").
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# Hardcoded high-confidence aliases. Sources are human-curated from
# DrugBank.ca and Disease Ontology pages — limited to entities present in
# the smoke embedding set. Add to data/*_synonyms.tsv to scale further.
_BUILTIN_DRUG_ALIASES: dict[str, list[str]] = {
    "DB00178": ["ramipril", "altace", "tritace"],
    "DB00305": ["mycophenolate mofetil", "mycophenolate", "cellcept"],
    "DB00635": ["prednisone", "deltasone", "rayos"],
    "DB00641": ["simvastatin", "zocor"],
    "DB00678": ["losartan", "cozaar"],
    "DB00705": ["amlodipine", "norvasc"],
    "DB00938": ["salmeterol", "serevent"],
    "DB00997": ["doxorubicin", "adriamycin"],
    "DB01003": ["cromolyn", "cromolyn sodium"],
    "DB01098": ["rosuvastatin", "crestor"],
    "DB01120": ["gliclazide", "diamicron"],
}

_BUILTIN_DISEASE_ALIASES: dict[str, list[str]] = {
    "DOID:635": ["acquired immunodeficiency syndrome", "aids"],
    "DOID:1936": ["atherosclerosis"],
    "DOID:2841": ["asthma"],
    "DOID:3571": ["liver cirrhosis"],
    "DOID:9352": ["type 2 diabetes mellitus", "type 2 diabetes", "t2dm",
                   "diabetes mellitus type 2", "non-insulin-dependent diabetes mellitus"],
    "DOID:10534": ["stomach cancer", "gastric cancer"],
    "DOID:13189": ["panic disorder"],
}


class SynonymIndex:
    """Resolve a free-text query to a canonical id, or None."""

    def __init__(self):
        self._drug_alias_to_id: dict[str, str] = {}
        self._disease_alias_to_id: dict[str, str] = {}
        self._load_builtin()

    def _load_builtin(self) -> None:
        for canonical, aliases in _BUILTIN_DRUG_ALIASES.items():
            self._drug_alias_to_id[canonical.lower()] = canonical
            for a in aliases:
                self._drug_alias_to_id[a.lower()] = canonical
        for canonical, aliases in _BUILTIN_DISEASE_ALIASES.items():
            self._disease_alias_to_id[canonical.lower()] = canonical
            for a in aliases:
                self._disease_alias_to_id[a.lower()] = canonical

    def load_external(self, data_dir: Path) -> None:
        """Layer additional aliases from data/*_synonyms.tsv if present."""
        drug_file = data_dir / "drugbank_synonyms.tsv"
        disease_file = data_dir / "doid_synonyms.tsv"
        if drug_file.exists():
            self._load_tsv(drug_file, self._drug_alias_to_id)
            log.info("loaded drug synonyms from %s", drug_file)
        if disease_file.exists():
            self._load_tsv(disease_file, self._disease_alias_to_id)
            log.info("loaded disease synonyms from %s", disease_file)

    def _load_tsv(self, path: Path, target: dict[str, str]) -> None:
        with path.open(encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 2:
                    continue
                canonical, alias = row[0].strip(), row[1].strip()
                if not canonical or not alias:
                    continue
                target[alias.lower()] = canonical
                target[canonical.lower()] = canonical

    def resolve_drug(self, query: str) -> Optional[str]:
        if not query:
            return None
        return self._drug_alias_to_id.get(query.strip().lower())

    def resolve_disease(self, query: str) -> Optional[str]:
        if not query:
            return None
        return self._disease_alias_to_id.get(query.strip().lower())

    def stats(self) -> dict[str, int]:
        return {
            "drug_aliases": len(self._drug_alias_to_id),
            "disease_aliases": len(self._disease_alias_to_id),
        }
