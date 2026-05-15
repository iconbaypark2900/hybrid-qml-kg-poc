from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvidenceFeatures:
    """
    Full feature vector for a single compound-disease candidate.

    All scores are floats in [0, 1] or [-1, 1].
    Fields are zero-filled (not None) so the vector is always dense.
    """
    compound: str
    compound_hetionet_id: Optional[str]
    disease: str
    disease_hetionet_id: Optional[str]

    # KG-based scores (from existing pipeline)
    kg_rotate_score: float = 0.0
    kg_complex_score: float = 0.0
    graph_topology_score: float = 0.0

    # QML scores
    qsvc_score: float = 0.0
    classical_ensemble_score: float = 0.0

    # Omics evidence (zero in kg-only mode)
    signature_reversal_score: float = 0.0
    cell_type_reversal_score: float = 0.0
    pathway_reversal_score: float = 0.0

    # Mechanism and clinical
    moa_alignment_score: float = 0.0
    clinical_evidence_score: float = 0.0

    # Computed fields
    final_score: float = 0.0
    confidence_tier: int = 4   # 1 = highest
    explanation: str = ""

    def to_dict(self) -> Dict:
        return {
            "compound": self.compound,
            "compound_hetionet_id": self.compound_hetionet_id,
            "disease": self.disease,
            "disease_hetionet_id": self.disease_hetionet_id,
            "kg_rotate_score": self.kg_rotate_score,
            "kg_complex_score": self.kg_complex_score,
            "graph_topology_score": self.graph_topology_score,
            "qsvc_score": self.qsvc_score,
            "classical_ensemble_score": self.classical_ensemble_score,
            "signature_reversal_score": self.signature_reversal_score,
            "cell_type_reversal_score": self.cell_type_reversal_score,
            "pathway_reversal_score": self.pathway_reversal_score,
            "moa_alignment_score": self.moa_alignment_score,
            "clinical_evidence_score": self.clinical_evidence_score,
            "final_score": self.final_score,
            "confidence_tier": self.confidence_tier,
            "explanation": self.explanation,
        }

    def feature_vector(self) -> list:
        """Return model-ready list of numeric features (same order as config)."""
        return [
            self.kg_rotate_score,
            self.kg_complex_score,
            self.graph_topology_score,
            self.qsvc_score,
            self.classical_ensemble_score,
            self.signature_reversal_score,
            self.cell_type_reversal_score,
            self.pathway_reversal_score,
            self.moa_alignment_score,
            self.clinical_evidence_score,
        ]
