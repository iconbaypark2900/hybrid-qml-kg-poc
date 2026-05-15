from __future__ import annotations

"""
Shared artifact loaders for the Streamlit dashboard pages.

All loaders return safe defaults (empty DataFrame, empty list, None) when
an artifact is missing — pages should render an info message rather than
raising.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_PRED_DIR = Path("artifacts/predictions")
_DEFAULT_SIG_DIR = Path("artifacts/signatures")
_DEFAULT_PERT_DIR = Path("artifacts/perturbations")


def load_top_candidates(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load top_candidates.csv emitted by run_full_repurposing_pipeline.py."""
    csv_path = Path(path) if path else _DEFAULT_PRED_DIR / "top_candidates.csv"
    if not csv_path.exists():
        logger.info(f"No candidates file at {csv_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} candidates from {csv_path}")
        return df
    except Exception as e:
        logger.warning(f"Failed to read {csv_path}: {e}")
        return pd.DataFrame()


def load_top_candidates_json(
    path: Optional[Path] = None,
) -> List[Dict]:
    """Load top_candidates.json (richer than CSV — keeps explanation field)."""
    json_path = Path(path) if path else _DEFAULT_PRED_DIR / "top_candidates.json"
    if not json_path.exists():
        return []
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read {json_path}: {e}")
        return []


def load_run_summary(path: Optional[Path] = None) -> Dict:
    """Load run_summary.json — mode, top_compound, tier distribution."""
    p = Path(path) if path else _DEFAULT_PRED_DIR / "run_summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read {p}: {e}")
        return {}


def list_diseases(candidates: pd.DataFrame) -> List[str]:
    """Distinct disease labels from the candidates table."""
    if candidates.empty or "disease" not in candidates.columns:
        return []
    return sorted(candidates["disease"].dropna().unique().tolist())


def filter_by_disease(candidates: pd.DataFrame, disease: str) -> pd.DataFrame:
    """Subset candidates to a single disease (case-insensitive substring match)."""
    if candidates.empty or not disease:
        return candidates
    mask = candidates["disease"].str.contains(disease, case=False, na=False)
    return candidates[mask].copy()


def load_disease_signature(
    path: Optional[Path] = None,
) -> Dict:
    """Load artifacts/signatures/disease_signature.json (single cohort)."""
    p = Path(path) if path else _DEFAULT_SIG_DIR / "disease_signature.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read {p}: {e}")
        return {}


def load_cell_type_signatures(
    path: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Load artifacts/signatures/cell_type_signatures.json (per-cell-type)."""
    p = Path(path) if path else _DEFAULT_SIG_DIR / "cell_type_signatures.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"Failed to read {p}: {e}")
        return {}


def load_reversal_scores(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load artifacts/perturbations/reversal_scores.csv."""
    p = Path(path) if path else _DEFAULT_PERT_DIR / "reversal_scores.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception as e:
        logger.warning(f"Failed to read {p}: {e}")
        return pd.DataFrame()


def candidate_to_dict(row) -> Dict:
    """Convert one pandas Series candidate row into an EvidenceFeatures-shaped dict."""
    return {
        "compound": row.get("compound", ""),
        "compound_hetionet_id": row.get("compound_hetionet_id", ""),
        "disease": row.get("disease", ""),
        "disease_hetionet_id": row.get("disease_hetionet_id", ""),
        "final_score": float(row.get("final_score", 0.0)),
        "confidence_tier": int(row.get("confidence_tier", 4)),
        "kg_rotate_score": float(row.get("kg_rotate_score", 0.0)),
        "kg_complex_score": float(row.get("kg_complex_score", 0.0)),
        "qsvc_score": float(row.get("qsvc_score", 0.0)),
        "classical_ensemble_score": float(row.get("classical_ensemble_score", 0.0)),
        "signature_reversal_score": float(row.get("signature_reversal_score", 0.0)),
        "cell_type_reversal_score": float(row.get("cell_type_reversal_score", 0.0)),
        "pathway_reversal_score": float(row.get("pathway_reversal_score", 0.0)),
        "clinical_evidence_score": float(row.get("clinical_evidence_score", 0.0)),
        "explanation": row.get("explanation", "") if "explanation" in row else "",
    }
