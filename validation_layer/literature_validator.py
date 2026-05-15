from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_ENTREZ_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"


def pubmed_cooccurrence_count(
    compound_name: str,
    disease_name: str,
    email: str = "research@example.com",
) -> int:
    """
    Count PubMed abstracts mentioning both compound and disease (co-occurrence proxy).

    Uses NCBI Entrez esearch with AND query. Returns 0 on failure.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; literature validation disabled.")
        return 0

    query = f'"{compound_name}"[tiab] AND "{disease_name}"[tiab]'
    params = {
        "db": "pubmed",
        "term": query,
        "rettype": "count",
        "retmode": "json",
        "tool": "hybrid-qml-kg-poc",
        "email": email,
    }
    try:
        resp = requests.get(_ENTREZ_SEARCH, params=params, timeout=10)
        resp.raise_for_status()
        count = int(resp.json()["esearchresult"]["count"])
        return count
    except Exception as e:
        logger.warning(f"PubMed query failed ({e}); returning 0.")
        return 0


def annotate_with_literature_support(
    pairs: List[Dict],
    compound_name_col: str = "compound",
    disease_name_col: str = "disease",
    email: str = "research@example.com",
) -> List[Dict]:
    """Add 'literature_support_count' to each pair dict."""
    for pair in pairs:
        pair["literature_support_count"] = pubmed_cooccurrence_count(
            pair.get(compound_name_col, ""),
            pair.get(disease_name_col, ""),
            email=email,
        )
    return pairs
