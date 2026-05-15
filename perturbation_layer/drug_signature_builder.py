from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DrugSignature:
    compound: str
    compound_hetionet_id: Optional[str]
    source: str
    cell_line: Optional[str]
    dose: Optional[str]
    timepoint: Optional[str]
    up_genes: List[str] = field(default_factory=list)
    down_genes: List[str] = field(default_factory=list)
    ranked_genes: List[Dict] = field(default_factory=list)   # [{gene, score}]

    def to_dict(self) -> Dict:
        return {
            "compound": self.compound,
            "compound_hetionet_id": self.compound_hetionet_id,
            "source": self.source,
            "cell_line": self.cell_line,
            "dose": self.dose,
            "timepoint": self.timepoint,
            "up_genes": self.up_genes,
            "down_genes": self.down_genes,
            "ranked_genes": self.ranked_genes,
        }


def build_signatures(
    df: pd.DataFrame,
    top_n: int = 250,
    compound_col: str = "compound",
    gene_col: str = "gene",
    score_col: str = "score",
    source: str = "LINCS",
    cell_line_col: Optional[str] = "cell_line",
    dose_col: Optional[str] = "dose",
    time_col: Optional[str] = "timepoint",
    compound_mapper=None,
) -> Dict[str, DrugSignature]:
    """
    Build per-compound DrugSignature objects from a tidy LINCS/CMap DataFrame.

    For each compound:
      - Top N genes by |score| are used.
      - Positive score → up_genes; negative → down_genes.
      - compound_mapper (CompoundMapper) normalises IDs to Hetionet nodes.

    Returns {compound_name: DrugSignature}.
    """
    signatures: Dict[str, DrugSignature] = {}
    grouped = df.groupby(compound_col)

    for compound, group in grouped:
        group = group.copy()
        group["abs_score"] = group[score_col].abs()
        top = group.nlargest(top_n, "abs_score")

        ranked = [
            {"gene": row[gene_col], "score": round(float(row[score_col]), 4)}
            for _, row in top.iterrows()
        ]
        up_genes = [r["gene"] for r in ranked if r["score"] > 0]
        down_genes = [r["gene"] for r in ranked if r["score"] < 0]

        # ID normalisation
        hetionet_id: Optional[str] = None
        if compound_mapper is not None:
            hetionet_id = compound_mapper.map(str(compound))

        # Metadata from first row
        first = group.iloc[0]
        cell_line = str(first[cell_line_col]) if cell_line_col and cell_line_col in group.columns else None
        dose = str(first[dose_col]) if dose_col and dose_col in group.columns else None
        timepoint = str(first[time_col]) if time_col and time_col in group.columns else None

        signatures[str(compound)] = DrugSignature(
            compound=str(compound),
            compound_hetionet_id=hetionet_id,
            source=source,
            cell_line=cell_line,
            dose=dose,
            timepoint=timepoint,
            up_genes=up_genes,
            down_genes=down_genes,
            ranked_genes=ranked,
        )

    logger.info(f"Built {len(signatures)} drug signatures (top_n={top_n})")
    return signatures
