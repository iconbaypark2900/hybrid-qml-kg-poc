from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def map_perturbation_ids_to_hetionet(
    compound_names: List[str],
    compound_mapper,
) -> Dict[str, Optional[str]]:
    """
    Map a list of perturbation compound names to Hetionet node IDs.

    Uses entity_resolution.CompoundMapper. Returns {name: hetionet_id_or_None}.
    Logs a summary of resolved vs. unresolved.
    """
    mapping = compound_mapper.map_many(compound_names)
    n_resolved = sum(1 for v in mapping.values() if v is not None)
    logger.info(
        f"Compound → Hetionet mapping: {n_resolved}/{len(compound_names)} resolved "
        f"({n_resolved / max(len(compound_names), 1) * 100:.1f}%)"
    )
    return mapping
