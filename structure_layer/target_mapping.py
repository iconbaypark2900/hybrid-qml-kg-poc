"""Utilities for aggregating target-level structure features into KG pairs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_AGGREGATED_FIELDS = [
    "has_structure",
    "parse_success",
    "missing_structure",
    "residue_count",
    "chain_count",
    "ca_atom_count",
    "residue_coverage",
    "plddt_mean",
    "plddt_low_conf_fraction",
    "contact_density",
    "radius_gyration",
    "compactness",
]

DEFAULT_COMPOUND_GENE_EDGES = ("CbG", "CdG", "CuG")
DEFAULT_DISEASE_GENE_EDGES = ("DaG", "DdG", "DuG")


@dataclass(frozen=True)
class StructureTargetIndex:
    """Pre-built lookup tables for pair-specific structure target resolution."""

    compound_to_genes: dict[str, set[str]] = field(default_factory=dict)
    disease_to_genes: dict[str, set[str]] = field(default_factory=dict)
    target_features: dict[str, dict[str, Any]] = field(default_factory=dict)
    feature_fields: tuple[str, ...] = tuple(DEFAULT_AGGREGATED_FIELDS)

    @classmethod
    def from_edges(
        cls,
        edges_df: pd.DataFrame,
        target_features: dict[str, dict[str, Any]],
        *,
        compound_gene_edges: Iterable[str] = DEFAULT_COMPOUND_GENE_EDGES,
        disease_gene_edges: Iterable[str] = DEFAULT_DISEASE_GENE_EDGES,
        feature_fields: Iterable[str] = DEFAULT_AGGREGATED_FIELDS,
    ) -> "StructureTargetIndex":
        df = edges_df[["source", "metaedge", "target"]].astype(str)
        compound_gene_edge_set = set(compound_gene_edges)
        disease_gene_edge_set = set(disease_gene_edges)
        compound_to_genes: dict[str, set[str]] = defaultdict(set)
        disease_to_genes: dict[str, set[str]] = defaultdict(set)

        for source, metaedge, target in df.itertuples(index=False):
            if metaedge in compound_gene_edge_set and source.startswith("Compound::") and target.startswith("Gene::"):
                compound_to_genes[source].add(target)
            elif metaedge in disease_gene_edge_set and source.startswith("Disease::") and target.startswith("Gene::"):
                disease_to_genes[source].add(target)

        return cls(
            compound_to_genes=dict(compound_to_genes),
            disease_to_genes=dict(disease_to_genes),
            target_features=target_features,
            feature_fields=tuple(feature_fields),
        )

    def shared_targets_for_pair(self, source_id: str, target_id: str) -> list[str]:
        compound_id, disease_id = _compound_disease_orientation(source_id, target_id)
        if not compound_id or not disease_id:
            return []
        shared = self.compound_to_genes.get(compound_id, set()) & self.disease_to_genes.get(disease_id, set())
        return sorted(shared)

    def aggregate_pair(self, source_id: str, target_id: str) -> dict[str, Any]:
        pair = (source_id, target_id)
        targets = self.shared_targets_for_pair(source_id, target_id)
        return aggregate_pair_structure_features(
            {pair: targets},
            self.target_features,
            feature_fields=list(self.feature_fields),
        )[pair]


def structure_pair_feature_names(
    *,
    feature_fields: Iterable[str] = DEFAULT_AGGREGATED_FIELDS,
) -> list[str]:
    names = [
        "structure_target_count",
        "structure_feature_target_count",
        "structure_feature_missing_rate",
    ]
    for field in feature_fields:
        names.extend([f"structure_{field}_mean", f"structure_{field}_max"])
    return names


def build_pair_to_targets_from_edges(
    pairs_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    *,
    source_col: str = "source",
    target_col: str = "target",
    target_features: dict[str, dict[str, Any]] | None = None,
    compound_gene_edges: Iterable[str] = DEFAULT_COMPOUND_GENE_EDGES,
    disease_gene_edges: Iterable[str] = DEFAULT_DISEASE_GENE_EDGES,
) -> dict[tuple[str, str], list[str]]:
    """Resolve pair-specific structure targets from shared Hetionet genes."""

    index = StructureTargetIndex.from_edges(
        edges_df,
        target_features or {},
        compound_gene_edges=compound_gene_edges,
        disease_gene_edges=disease_gene_edges,
    )
    result: dict[tuple[str, str], list[str]] = {}
    for _, row in pairs_df.iterrows():
        source_id = str(row[source_col])
        target_id = str(row[target_col])
        result[(source_id, target_id)] = index.shared_targets_for_pair(source_id, target_id)
    return result


def build_pair_structure_feature_matrix(
    pairs_df: pd.DataFrame,
    index: StructureTargetIndex,
    *,
    source_col: str = "source",
    target_col: str = "target",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Build a dense numeric structure-feature block for pair rows."""

    names = structure_pair_feature_names(feature_fields=index.feature_fields)
    rows: list[list[float]] = []
    target_provenance: list[str] = []
    for _, row in pairs_df.iterrows():
        source_id = str(row[source_col])
        target_id = str(row[target_col])
        aggregated = index.aggregate_pair(source_id, target_id)
        target_provenance.append(str(aggregated.get("structure_feature_target_ids", "")))
        rows.append([_numeric_or_zero(aggregated.get(name)) for name in names])
    return np.array(rows, dtype=np.float32), names, target_provenance


def aggregate_pair_structure_features(
    pair_to_targets: dict[tuple[str, str], Iterable[str]],
    target_features: dict[str, dict[str, Any]],
    *,
    feature_fields: list[str] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Aggregate target-level features for compound-disease pairs.

    The caller controls how pair targets are resolved from KG evidence. This
    function only performs deterministic mean/max/count/missingness aggregation.
    """

    fields = feature_fields or DEFAULT_AGGREGATED_FIELDS
    aggregated: dict[tuple[str, str], dict[str, Any]] = {}
    for pair, target_ids_iter in pair_to_targets.items():
        target_ids = list(dict.fromkeys(target_ids_iter))
        available = [target_features[target_id] for target_id in target_ids if target_id in target_features]
        row: dict[str, Any] = {
            "structure_target_count": len(target_ids),
            "structure_feature_target_count": len(available),
            "structure_feature_missing_rate": (
                1.0 - (len(available) / len(target_ids)) if target_ids else 1.0
            ),
            "structure_feature_target_ids": "|".join(target_ids),
        }
        numeric_values: dict[str, list[float]] = defaultdict(list)
        for feature in available:
            for field in fields:
                value = feature.get(field)
                if isinstance(value, bool):
                    numeric_values[field].append(float(value))
                elif isinstance(value, (int, float)):
                    numeric_values[field].append(float(value))
        for field in fields:
            values = numeric_values.get(field, [])
            if values:
                row[f"structure_{field}_mean"] = sum(values) / len(values)
                row[f"structure_{field}_max"] = max(values)
            else:
                row[f"structure_{field}_mean"] = None
                row[f"structure_{field}_max"] = None
        aggregated[pair] = row
    return aggregated


def _compound_disease_orientation(source_id: str, target_id: str) -> tuple[str | None, str | None]:
    if source_id.startswith("Compound::") and target_id.startswith("Disease::"):
        return source_id, target_id
    if target_id.startswith("Compound::") and source_id.startswith("Disease::"):
        return target_id, source_id
    return None, None


def _numeric_or_zero(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
