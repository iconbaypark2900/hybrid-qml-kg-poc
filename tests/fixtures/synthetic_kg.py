"""Deterministic synthetic biomedical-flavored knowledge graph generator.

Mimics the structure of Hetionet's drug-repurposing subgraph at smaller
scale: Compounds, Diseases, Genes connected by the four edge types in
preregistration §3.2 (CtD, CrC, CbG, DaG). Suitable as a CI fixture so
the test suite does not require Hetionet downloads or PyKEEN-trained
embeddings.

Node-type encoding:
  0 = Compound, 1 = Disease, 2 = Gene
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class LinkPredictionInstance:
    """A candidate edge for link prediction."""
    head: int
    tail: int
    edge_type: str
    features: np.ndarray
    label: int
    instance_id: str

    def __post_init__(self) -> None:
        if self.features.ndim != 1:
            raise ValueError(f"features must be 1-D, got shape {self.features.shape}")
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label}")


@dataclass
class SyntheticKG:
    """Minimal multi-relational knowledge graph."""
    n_nodes: int
    node_types: np.ndarray
    edges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_types.shape != (self.n_nodes,):
            raise ValueError(
                f"node_types shape {self.node_types.shape} != ({self.n_nodes},)"
            )

    def neighbors(self, node: int, edge_type: str) -> list[int]:
        return [t for h, t in self.edges.get(edge_type, []) if h == node]

    def has_edge(self, head: int, tail: int, edge_type: str) -> bool:
        return (head, tail) in self.edges.get(edge_type, [])


@dataclass(frozen=True)
class SyntheticKGConfig:
    """Configuration for synthetic KG generation."""
    n_compounds: int = 50
    n_diseases: int = 30
    n_genes: int = 60
    edge_density_ctd: float = 0.05
    edge_density_crc: float = 0.10
    edge_density_cbg: float = 0.08
    edge_density_dag: float = 0.06

    def __post_init__(self) -> None:
        for name, val in (
            ("n_compounds", self.n_compounds),
            ("n_diseases", self.n_diseases),
            ("n_genes", self.n_genes),
        ):
            if val < 2:
                raise ValueError(f"{name} must be >= 2, got {val}")
        for name, val in (
            ("edge_density_ctd", self.edge_density_ctd),
            ("edge_density_crc", self.edge_density_crc),
            ("edge_density_cbg", self.edge_density_cbg),
            ("edge_density_dag", self.edge_density_dag),
        ):
            if not 0 < val < 1:
                raise ValueError(f"{name} must be in (0, 1), got {val}")


def generate_synthetic_kg(config: SyntheticKGConfig, *, seed: int) -> SyntheticKG:
    """Generate a synthetic knowledge graph matching the given config.

    The graph is **not** uniformly random across edge types: CrC edges are
    biased to connect compound pairs that share CtD targets, mimicking the
    real biomedical regularity that compounds treating overlapping diseases
    tend to be structurally similar. This structural prior is what lets the
    2-hop (CrC, CtD) and common-neighbor features carry class signal in the
    smoke test, after the target-edge direct feature is removed in
    ``extract_simple_features``.
    """
    rng = np.random.default_rng(seed)
    n_total = config.n_compounds + config.n_diseases + config.n_genes
    node_types = np.zeros(n_total, dtype=np.int64)
    node_types[config.n_compounds : config.n_compounds + config.n_diseases] = 1
    node_types[config.n_compounds + config.n_diseases :] = 2

    compound_ids = list(range(config.n_compounds))
    disease_ids = list(range(config.n_compounds, config.n_compounds + config.n_diseases))
    gene_ids = list(range(config.n_compounds + config.n_diseases, n_total))

    def random_edges(
        heads: list[int],
        tails: list[int],
        density: float,
        allow_self: bool = True,
    ) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for h in heads:
            for t in tails:
                if not allow_self and h == t:
                    continue
                if rng.random() < density:
                    out.append((h, t))
        return out

    # CtD first — uniformly random.
    ctd_edges = random_edges(compound_ids, disease_ids, config.edge_density_ctd)

    # Build compound -> {treated diseases} map.
    compound_treats: dict[int, set[int]] = {c: set() for c in compound_ids}
    for c, d in ctd_edges:
        compound_treats[c].add(d)

    # CrC — structural prior: each shared CtD target boosts the pair's edge
    # probability. With BOOST_PER_SHARED = 0.15, two compounds sharing 2
    # treatments have a baseline + 0.30 probability cap (capped at 0.95).
    BOOST_PER_SHARED = 0.15
    crc_edges: list[tuple[int, int]] = []
    for c1 in compound_ids:
        for c2 in compound_ids:
            if c1 == c2:
                continue
            shared = len(compound_treats[c1] & compound_treats[c2])
            p = min(config.edge_density_crc + BOOST_PER_SHARED * shared, 0.95)
            if rng.random() < p:
                crc_edges.append((c1, c2))

    edges = {
        "CtD": ctd_edges,
        "CrC": crc_edges,
        "CbG": random_edges(compound_ids, gene_ids, config.edge_density_cbg),
        "DaG": random_edges(disease_ids, gene_ids, config.edge_density_dag),
    }
    return SyntheticKG(n_nodes=n_total, node_types=node_types, edges=edges)


def extract_simple_features(
    kg: SyntheticKG, head: int, tail: int, edge_type: str
) -> np.ndarray:
    """Lightweight path-count features for a candidate (head, tail, edge_type).

    The 1-hop direct-edge feature is intentionally **omitted for the target
    edge type** to avoid a label leak (a positive instance of `edge_type`
    has `kg.has_edge(head, tail, edge_type) == 1` by construction). The
    direct-edge feature is retained for all *other* edge types because
    those carry useful signal without leaking the label.
    """
    edge_types = sorted(kg.edges.keys())
    feats: list[float] = []

    for et in edge_types:
        if et == edge_type:
            # Skip the target edge type's direct-edge feature — it would leak the label.
            continue
        feats.append(1.0 if kg.has_edge(head, tail, et) else 0.0)

    for et1 in edge_types:
        for et2 in edge_types:
            mid_nodes = kg.neighbors(head, et1)
            count = sum(1 for m in mid_nodes if kg.has_edge(m, tail, et2))
            feats.append(float(np.log1p(count)))

    head_neighbors: set[int] = set()
    tail_neighbors: set[int] = set()
    for et in edge_types:
        head_neighbors.update(kg.neighbors(head, et))
        tail_neighbors.update(h for h, t in kg.edges.get(et, []) if t == tail)
    common = len(head_neighbors & tail_neighbors)
    feats.append(float(np.log1p(common)))

    n_node_types = int(kg.node_types.max() + 1) if kg.n_nodes > 0 else 1
    head_type_onehot = np.zeros(n_node_types)
    head_type_onehot[kg.node_types[head]] = 1.0
    tail_type_onehot = np.zeros(n_node_types)
    tail_type_onehot[kg.node_types[tail]] = 1.0

    return np.concatenate([np.array(feats), head_type_onehot, tail_type_onehot])


def generate_link_prediction_instances(
    kg: SyntheticKG,
    *,
    target_edge_type: str,
    n_positive: int,
    n_negative: int,
    seed: int,
) -> list[LinkPredictionInstance]:
    """Sample link prediction instances for a target edge type.

    Positives: sampled from existing edges of `target_edge_type`.
    Negatives: random non-edges between nodes of correct types (1:1 stratified
    in the default config — note this is the random-negatives variant
    described in preregistration §7.1; the production pipeline uses hard
    negatives instead).
    """
    rng = np.random.default_rng(seed)
    if target_edge_type not in kg.edges:
        raise ValueError(f"target_edge_type {target_edge_type} not in KG")
    positive_edges = kg.edges[target_edge_type]
    if len(positive_edges) == 0:
        raise ValueError(
            f"No positive edges of type {target_edge_type} in KG; "
            "cannot generate instances"
        )

    if len(positive_edges) >= n_positive:
        pos_idx = rng.choice(len(positive_edges), size=n_positive, replace=False)
        sampled_positives = [positive_edges[i] for i in pos_idx]
    else:
        sampled_positives = list(positive_edges)
        n_positive = len(sampled_positives)

    if target_edge_type == "CtD":
        head_pool = [n for n in range(kg.n_nodes) if kg.node_types[n] == 0]
        tail_pool = [n for n in range(kg.n_nodes) if kg.node_types[n] == 1]
    elif target_edge_type == "CrC":
        head_pool = [n for n in range(kg.n_nodes) if kg.node_types[n] == 0]
        tail_pool = head_pool
    elif target_edge_type == "CbG":
        head_pool = [n for n in range(kg.n_nodes) if kg.node_types[n] == 0]
        tail_pool = [n for n in range(kg.n_nodes) if kg.node_types[n] == 2]
    elif target_edge_type == "DaG":
        head_pool = [n for n in range(kg.n_nodes) if kg.node_types[n] == 1]
        tail_pool = [n for n in range(kg.n_nodes) if kg.node_types[n] == 2]
    else:
        raise ValueError(f"Unknown edge type for negative sampling: {target_edge_type}")

    positive_set = set(positive_edges)
    sampled_negatives: list[tuple[int, int]] = []
    seen_negatives: set[tuple[int, int]] = set()
    attempts = 0
    while len(sampled_negatives) < n_negative and attempts < n_negative * 100:
        attempts += 1
        h = int(rng.choice(head_pool))
        t = int(rng.choice(tail_pool))
        if (h, t) not in positive_set and (h, t) not in seen_negatives:
            sampled_negatives.append((h, t))
            seen_negatives.add((h, t))

    instances: list[LinkPredictionInstance] = []
    for i, (h, t) in enumerate(sampled_positives):
        feats = extract_simple_features(kg, h, t, target_edge_type)
        instances.append(
            LinkPredictionInstance(
                head=h,
                tail=t,
                edge_type=target_edge_type,
                features=feats,
                label=1,
                instance_id=f"{target_edge_type}_pos_{i}",
            )
        )
    for i, (h, t) in enumerate(sampled_negatives):
        feats = extract_simple_features(kg, h, t, target_edge_type)
        instances.append(
            LinkPredictionInstance(
                head=h,
                tail=t,
                edge_type=target_edge_type,
                features=feats,
                label=0,
                instance_id=f"{target_edge_type}_neg_{i}",
            )
        )
    return instances


def split_instances(
    instances: list[LinkPredictionInstance],
    *,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[
    list[LinkPredictionInstance],
    list[LinkPredictionInstance],
    list[LinkPredictionInstance],
]:
    """Split instances into train/val/test partitions, stratified by label."""
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train + val fractions leave no test partition")
    rng = np.random.default_rng(seed)

    pos = [inst for inst in instances if inst.label == 1]
    neg = [inst for inst in instances if inst.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_list(
        lst: list[LinkPredictionInstance], train_f: float, val_f: float
    ) -> tuple[
        list[LinkPredictionInstance],
        list[LinkPredictionInstance],
        list[LinkPredictionInstance],
    ]:
        n = len(lst)
        n_train = int(round(n * train_f))
        n_val = int(round(n * val_f))
        return lst[:n_train], lst[n_train : n_train + n_val], lst[n_train + n_val :]

    pt, pv, pte = split_list(pos, train_frac, val_frac)
    nt, nv, nte = split_list(neg, train_frac, val_frac)
    return pt + nt, pv + nv, pte + nte


def instances_to_arrays(
    instances: list[LinkPredictionInstance],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack instance features and labels into ``(X, y)`` numpy arrays."""
    if not instances:
        raise ValueError("cannot stack empty instance list")
    X = np.stack([inst.features for inst in instances], axis=0)
    y = np.array([inst.label for inst in instances], dtype=int)
    return X, y


__all__ = [
    "LinkPredictionInstance",
    "SyntheticKG",
    "SyntheticKGConfig",
    "generate_synthetic_kg",
    "generate_link_prediction_instances",
    "extract_simple_features",
    "split_instances",
    "instances_to_arrays",
]
