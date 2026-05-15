from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_DB_ID = re.compile(r"^DB\d{5}$")
_HETIONET_PREFIX = "Compound::"


# Seed dictionary — DrugBank ID → list of disease MeSH / DOID terms it is
# approved/indicated for. This is intentionally small; the real lookup
# goes through a downloaded DrugBank XML or the licensed API. Kept here
# so check_drugbank_indication() works offline for smoke tests.
_DRUGBANK_SEED_INDICATIONS: Dict[str, Set[str]] = {
    "DB00331": {"DOID:9352", "type 2 diabetes mellitus"},        # metformin
    "DB00945": {"DOID:9351", "diabetes mellitus"},                # aspirin
    "DB01098": {"DOID:1387", "hyperlipidemia"},                   # rosuvastatin
    "DB00682": {"DOID:0060224", "atrial fibrillation"},           # warfarin
    "DB00641": {"DOID:2487", "hypercholesterolemia"},             # simvastatin
    "DB00381": {"DOID:10763", "hypertension"},                    # amlodipine
    "DB00451": {"DOID:1459", "hypothyroidism"},                   # levothyroxine
    "DB00619": {"DOID:8552", "chronic myeloid leukemia"},         # imatinib
    "DB00675": {"DOID:1612", "breast cancer"},                    # tamoxifen
    "DB00678": {"DOID:10763", "hypertension"},                    # losartan
    "DB01104": {"DOID:1470", "major depressive disorder"},        # sertraline
    "DB00338": {"DOID:8534", "gastroesophageal reflux disease"},  # omeprazole
}


def _normalize_drugbank_id(query: str) -> Optional[str]:
    """Pull a bare DrugBank ID (DBxxxxx) out of common Hetionet prefixes."""
    if not query:
        return None
    q = query.strip()
    if q.startswith(_HETIONET_PREFIX):
        q = q[len(_HETIONET_PREFIX):]
    return q if _DB_ID.match(q) else None


def check_drugbank_indication(
    compound_id: str,
    disease_term: str,
    indication_table: Optional[Dict[str, Set[str]]] = None,
) -> bool:
    """
    Check whether `compound_id` has an approved indication matching `disease_term`.

    Matches loosely: any string membership in the indication set is a hit
    (DOID, MeSH term, or free-text label).
    """
    table = indication_table if indication_table is not None else _DRUGBANK_SEED_INDICATIONS
    db_id = _normalize_drugbank_id(compound_id)
    if db_id is None:
        return False
    indications = table.get(db_id, set())
    if not indications:
        return False

    term_lc = disease_term.strip().lower()
    for ind in indications:
        if ind.lower() == term_lc or term_lc in ind.lower() or ind.lower() in term_lc:
            return True
    return False


def get_drugbank_indications(
    compound_id: str,
    indication_table: Optional[Dict[str, Set[str]]] = None,
) -> List[str]:
    """Return all known indications for a DrugBank ID, or []."""
    table = indication_table if indication_table is not None else _DRUGBANK_SEED_INDICATIONS
    db_id = _normalize_drugbank_id(compound_id)
    if db_id is None:
        return []
    return sorted(table.get(db_id, set()))


def load_drugbank_table_from_tsv(path: str) -> Dict[str, Set[str]]:
    """
    Load a TSV of DrugBank indications:  drugbank_id<TAB>indication_term

    Returns a {drugbank_id: {indications}} dict suitable for passing to
    `check_drugbank_indication(indication_table=...)`.
    """
    p = Path(path)
    if not p.exists():
        logger.warning(f"DrugBank TSV not found at {p}; returning seed table only.")
        return dict(_DRUGBANK_SEED_INDICATIONS)

    table: Dict[str, Set[str]] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            db_id, term = parts[0].strip(), parts[1].strip()
            if not _DB_ID.match(db_id):
                continue
            table.setdefault(db_id, set()).add(term)
    logger.info(f"Loaded {len(table)} DrugBank IDs from {p}")
    return table
