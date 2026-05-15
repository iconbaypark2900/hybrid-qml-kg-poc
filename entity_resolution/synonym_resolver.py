from __future__ import annotations

import logging
from typing import Dict, List, Optional

from entity_resolution.hetionet_resolver import HetionetResolver

logger = logging.getLogger(__name__)

# Seed synonyms imported from middleware/orchestrator.py's COMMON_NAME_ALIASES.
# These bridge brand names and informal spellings to Hetionet IDs.
_BUILTIN_ALIASES: Dict[str, str] = {
    # Compounds
    "aspirin":              "Compound::DB00945",
    "tylenol":              "Compound::DB00316",
    "acetaminophen":        "Compound::DB00316",
    "paracetamol":          "Compound::DB00316",
    "advil":                "Compound::DB01050",
    "motrin":               "Compound::DB01050",
    "ibuprofen":            "Compound::DB01050",
    "metformin":            "Compound::DB00331",
    "glucophage":           "Compound::DB00331",
    "atorvastatin":         "Compound::DB01076",
    "lipitor":              "Compound::DB01076",
    "dexamethasone":        "Compound::DB01234",
    "prednisone":           "Compound::DB00635",
    "warfarin":             "Compound::DB00682",
    "coumadin":             "Compound::DB00682",
    "lisinopril":           "Compound::DB00722",
    "metoprolol":           "Compound::DB00264",
    "amlodipine":           "Compound::DB00381",
    "omeprazole":           "Compound::DB00338",
    "prilosec":             "Compound::DB00338",
    "simvastatin":          "Compound::DB00641",
    "zocor":                "Compound::DB00641",
    "losartan":             "Compound::DB00678",
    "cozaar":               "Compound::DB00678",
    "levothyroxine":        "Compound::DB00451",
    "synthroid":            "Compound::DB00451",
    "albuterol":            "Compound::DB01001",
    "salbutamol":           "Compound::DB01001",
    "amoxicillin":          "Compound::DB01060",
    "ciprofloxacin":        "Compound::DB00537",
    "gabapentin":           "Compound::DB00996",
    "neurontin":            "Compound::DB00996",
    "sertraline":           "Compound::DB01104",
    "zoloft":               "Compound::DB01104",
    "fluoxetine":           "Compound::DB00472",
    "prozac":               "Compound::DB00472",
    "celecoxib":            "Compound::DB00482",
    "celebrex":             "Compound::DB00482",
    "montelukast":          "Compound::DB00471",
    "singulair":            "Compound::DB00471",
    # Diseases
    "diabetes":             "Disease::DOID:9351",
    "type 2 diabetes":      "Disease::DOID:9352",
    "hypertension":         "Disease::DOID:10763",
    "high blood pressure":  "Disease::DOID:10763",
    "cancer":               "Disease::DOID:162",
    "breast cancer":        "Disease::DOID:1612",
    "lung cancer":          "Disease::DOID:1324",
    "asthma":               "Disease::DOID:2841",
    "alzheimer":            "Disease::DOID:10652",
    "alzheimer's disease":  "Disease::DOID:10652",
    "alzheimers disease":   "Disease::DOID:10652",
    "parkinson":            "Disease::DOID:14330",
    "parkinson's disease":  "Disease::DOID:14330",
    "parkinsons disease":   "Disease::DOID:14330",
    "depression":           "Disease::DOID:1596",
    "rheumatoid arthritis": "Disease::DOID:7148",
    "multiple sclerosis":   "Disease::DOID:2377",
    "lupus":                "Disease::DOID:9074",
    "crohn":                "Disease::DOID:8778",
    "crohn's disease":      "Disease::DOID:8778",
    "heart failure":        "Disease::DOID:6000",
    "schizophrenia":        "Disease::DOID:5419",
}


class SynonymResolver:
    """
    Fuzzy + synonym-aware entity lookup.

    Resolution order:
      1. Built-in aliases (brand names, common misspellings)
      2. HetionetResolver name table (lowercased exact match)
      3. Simple normalisation: strip punctuation, collapse whitespace
      4. (Optional) difflib close-match if fuzzy=True
    """

    def __init__(
        self,
        resolver: Optional[HetionetResolver] = None,
        extra_aliases: Optional[Dict[str, str]] = None,
        fuzzy: bool = False,
        fuzzy_cutoff: float = 0.85,
    ) -> None:
        self._resolver = resolver or HetionetResolver()
        self._fuzzy = fuzzy
        self._fuzzy_cutoff = fuzzy_cutoff

        # Merge builtin + user aliases (user wins)
        self._aliases: Dict[str, str] = {**_BUILTIN_ALIASES}
        if extra_aliases:
            self._aliases.update({k.lower(): v for k, v in extra_aliases.items()})

    def _ensure_loaded(self) -> None:
        if not self._resolver._loaded:
            self._resolver.load()

    @staticmethod
    def _normalise(text: str) -> str:
        import re
        return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()
        # returns list of tokens; join for lookup

    def resolve(self, query: str) -> Optional[str]:
        """Return Hetionet node ID for query, or None."""
        self._ensure_loaded()
        q = query.strip()

        # 1. Alias table (lowercased)
        hit = self._aliases.get(q.lower())
        if hit:
            return hit

        # 2. HetionetResolver exact name
        hit = self._resolver.resolve_name(q)
        if hit:
            return hit

        # 3. Normalise punctuation and retry
        normalised = " ".join(self._normalise(q))
        if normalised != q.lower():
            hit = self._aliases.get(normalised)
            if hit:
                return hit
            hit = self._resolver.resolve_name(normalised)
            if hit:
                return hit

        # 4. Fuzzy match (optional, slow — only use on small candidate sets)
        if self._fuzzy:
            import difflib
            all_names = list(self._resolver._name_to_id.keys())
            matches = difflib.get_close_matches(
                q.lower(), all_names, n=1, cutoff=self._fuzzy_cutoff
            )
            if matches:
                return self._resolver._name_to_id[matches[0]]

        return None

    def resolve_many(self, queries: List[str]) -> Dict[str, Optional[str]]:
        return {q: self.resolve(q) for q in queries}
