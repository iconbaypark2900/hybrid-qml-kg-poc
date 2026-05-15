from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_CT_API = "https://clinicaltrials.gov/api/v2/studies"


def query_clinical_trials(
    compound_name: str,
    disease_name: str,
    max_results: int = 5,
) -> List[Dict]:
    """
    Query ClinicalTrials.gov v2 API for studies involving compound and disease.

    Returns a list of study dicts with keys: nct_id, title, phase, status.
    Returns [] if the request fails or no studies are found.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; clinical trial validation disabled.")
        return []

    params = {
        "query.term": f"{compound_name} {disease_name}",
        "fields": "NCTId,BriefTitle,Phase,OverallStatus",
        "pageSize": max_results,
        "format": "json",
    }
    try:
        resp = requests.get(_CT_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        studies = data.get("studies", [])
        results = []
        for s in studies:
            proto = s.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design_mod = proto.get("designModule", {})
            results.append({
                "nct_id": id_mod.get("nctId", ""),
                "title": id_mod.get("briefTitle", ""),
                "phase": design_mod.get("phases", ["Unknown"])[0] if design_mod.get("phases") else "Unknown",
                "status": status_mod.get("overallStatus", ""),
            })
        return results
    except Exception as e:
        logger.warning(f"ClinicalTrials.gov query failed ({e}); returning empty.")
        return []


def annotate_with_trial_evidence(
    pairs: List[Dict],
    compound_name_col: str = "compound",
    disease_name_col: str = "disease",
) -> List[Dict]:
    """
    Add 'clinical_trial_found', 'trial_phase' to each pair dict.
    Makes one API call per pair (rate-limited by ClinicalTrials.gov).
    """
    for pair in pairs:
        studies = query_clinical_trials(
            pair.get(compound_name_col, ""),
            pair.get(disease_name_col, ""),
            max_results=3,
        )
        pair["clinical_trial_found"] = len(studies) > 0
        if studies:
            pair["trial_phase"] = studies[0].get("phase", "Unknown")
            pair["trial_nct_id"] = studies[0].get("nct_id", "")
        else:
            pair["trial_phase"] = None
            pair["trial_nct_id"] = None
    return pairs
