from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"


def query_opentargets_evidence(
    drug_chembl_id: Optional[str],
    disease_efo_id: Optional[str],
    timeout: int = 10,
) -> Dict:
    """
    Query Open Targets Platform for a drug–disease evidence score.

    Args:
        drug_chembl_id: ChEMBL ID (e.g. "CHEMBL1431") — Open Targets uses ChEMBL,
            not DrugBank. Callers must map first.
        disease_efo_id: EFO ID (e.g. "EFO_0000400") — not DOID.

    Returns:
        {
          'overall_score': float in [0,1] (0 if not found / failed),
          'datatype_scores': {datatype: score},
          'source': 'open_targets' or 'unavailable',
        }
    """
    result: Dict = {"overall_score": 0.0, "datatype_scores": {}, "source": "unavailable"}

    if not drug_chembl_id or not disease_efo_id:
        logger.debug("Open Targets query skipped: missing chembl_id or efo_id.")
        return result

    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; Open Targets evidence disabled.")
        return result

    query = """
    query DrugDiseaseAssoc($drugId: String!, $diseaseId: String!) {
      drug(chemblId: $drugId) {
        id
        name
        indications {
          rows {
            disease { id name }
            maxPhaseForIndication
          }
        }
      }
      associationDatasources: associationDatatypes(
        drugId: $drugId, diseaseId: $diseaseId
      ) {
        rows { datatypeId score }
      }
    }
    """
    variables = {"drugId": drug_chembl_id, "diseaseId": disease_efo_id}

    try:
        resp = requests.post(
            _OT_GRAPHQL,
            json={"query": query, "variables": variables},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})

        # Indication phase → quick proxy for overall evidence
        max_phase = 0
        drug_data = data.get("drug") or {}
        indications = (drug_data.get("indications") or {}).get("rows", [])
        for row in indications:
            disease = row.get("disease") or {}
            if disease.get("id") == disease_efo_id:
                phase = row.get("maxPhaseForIndication") or 0
                if isinstance(phase, (int, float)) and phase > max_phase:
                    max_phase = int(phase)

        # Phase 4 (approved) → 1.0; phase 3 → 0.75; phase 2 → 0.5; phase 1 → 0.25.
        overall = max_phase / 4.0 if max_phase else 0.0

        datatypes: Dict[str, float] = {}
        for row in (data.get("associationDatasources") or {}).get("rows", []):
            dt = row.get("datatypeId")
            score = row.get("score")
            if dt is not None and score is not None:
                datatypes[str(dt)] = float(score)
                overall = max(overall, float(score))

        result["overall_score"] = overall
        result["datatype_scores"] = datatypes
        result["source"] = "open_targets"
        return result
    except Exception as e:
        logger.warning(f"Open Targets query failed ({e}); returning unavailable.")
        return result


def annotate_with_opentargets(
    pairs: List[Dict],
    drug_id_col: str = "drug_chembl_id",
    disease_id_col: str = "disease_efo_id",
) -> List[Dict]:
    """
    Add `opentargets_score` (float in [0,1]) to each pair dict.

    Each query is one HTTP request; rate-limit at the caller for >100 pairs.
    """
    for pair in pairs:
        ev = query_opentargets_evidence(
            pair.get(drug_id_col),
            pair.get(disease_id_col),
        )
        pair["opentargets_score"] = ev["overall_score"]
        pair["opentargets_source"] = ev["source"]
    return pairs
