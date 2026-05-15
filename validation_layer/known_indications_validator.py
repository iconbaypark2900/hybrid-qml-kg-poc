from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Small hard-coded seed of well-known approved indications (DrugBank ID → set of DOID).
# This is a fallback; the full validator queries DrugCentral or ChEMBL at runtime.
_KNOWN_SEED: Dict[str, Set[str]] = {
    "DB00331": {"DOID:9352"},   # metformin → type 2 diabetes
    "DB00945": {"DOID:1307"},   # aspirin → pain
    "DB01076": {"DOID:10763"},  # atorvastatin → hyperlipidaemia (via CVD)
}


def check_known_indication(
    compound_hetionet_id: str,
    disease_hetionet_id: str,
    extra_known: Optional[Dict[str, Set[str]]] = None,
) -> bool:
    """
    Return True if compound is a known approved treatment for disease.

    Checks the seed dict and any extra_known provided. For full coverage,
    call with DrugCentral/ChEMBL data loaded into extra_known.
    """
    db_id = compound_hetionet_id.replace("Compound::", "") if compound_hetionet_id else ""
    doid = disease_hetionet_id.replace("Disease::", "") if disease_hetionet_id else ""

    seed = dict(_KNOWN_SEED)
    if extra_known:
        for k, v in extra_known.items():
            seed.setdefault(k, set()).update(v)

    return doid in seed.get(db_id, set())


def batch_check_known_indications(
    pairs: List[Dict],
    extra_known: Optional[Dict[str, Set[str]]] = None,
) -> List[Dict]:
    """
    Annotate a list of {compound_hetionet_id, disease_hetionet_id, ...} dicts.

    Adds 'known_indication': bool to each dict in-place.
    """
    for pair in pairs:
        pair["known_indication"] = check_known_indication(
            pair.get("compound_hetionet_id", ""),
            pair.get("disease_hetionet_id", ""),
            extra_known=extra_known,
        )
    n_known = sum(1 for p in pairs if p.get("known_indication"))
    logger.info(f"Known indication check: {n_known}/{len(pairs)} are known")
    return pairs
