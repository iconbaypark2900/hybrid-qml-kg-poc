from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNRunResult:
    model_name: str
    pr_auc: float
    roc_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_seconds: float
    n_train: int
    n_test: int
    n_nodes: int
    n_edges: int


def _score_binary(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    thr = 0.0  # logits threshold
    y_pred = (y_score >= thr).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _build_sparse_adj(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """
    Build an undirected normalized adjacency (mean aggregator): D^{-1} A
    """
    # edge_index: (2, E)
    row, col = edge_index[0], edge_index[1]
    values = torch.ones(row.shape[0], device=device)
    A = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=values,
        size=(num_nodes, num_nodes),
        device=device,
    ).coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
    inv_deg = 1.0 / deg
    # normalize values by inv_deg[row]
    norm_values = inv_deg[row] * values
    A_norm = torch.sparse_coo_tensor(
        indices=A.indices(),
        values=norm_values,
        size=A.size(),
        device=device,
    ).coalesce()
    return A_norm


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(A_norm, h)
        out = self.lin_self(h) + self.lin_neigh(neigh)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class GINLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, eps: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(float(eps)))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # Sum aggregation (unnormalized adjacency)
        neigh = torch.sparse.mm(A, h)
        out = self.mlp((1.0 + self.eps) * h + neigh)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class EdgeDecoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        hs = h[src]
        ht = h[dst]
        feat = torch.cat([hs, ht, torch.abs(hs - ht), hs * ht], dim=-1)
        return self.mlp(feat).squeeze(-1)  # logits


def train_gnn_link_predictor(
    *,
    df_edges: pd.DataFrame,
    entity_to_id: Dict[str, int],
    node_features: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    arch: str = "graphsage",
    hidden_dim: int = 128,
    layers: int = 2,
    epochs: int = 60,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    random_state: int = 42,
    device: Optional[str] = None,
) -> GNNRunResult:
    """
    Lightweight GNN baseline using torch only (no torch-geometric).
    Graph is built from `df_edges` restricted to `entity_to_id` nodes.
    """
    import time
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_nodes = int(len(entity_to_id))
    if node_features.shape[0] != n_nodes:
        raise ValueError(f"node_features rows ({node_features.shape[0]}) != n_nodes ({n_nodes})")

    # Build edge index (undirected) from df_edges
    df = df_edges[["source", "target"]].astype(str)
    src = df["source"].map(entity_to_id).dropna().astype(int).to_numpy()
    dst = df["target"].map(entity_to_id).dropna().astype(int).to_numpy()
    src_t = torch.tensor(src, dtype=torch.long, device=dev)
    dst_t = torch.tensor(dst, dtype=torch.long, device=dev)
    # undirected
    edge_index = torch.stack([torch.cat([src_t, dst_t]), torch.cat([dst_t, src_t])], dim=0)
    n_edges = int(edge_index.shape[1])

    # Sparse adjacency
    A = torch.sparse_coo_tensor(edge_index, torch.ones(n_edges, device=dev), size=(n_nodes, n_nodes), device=dev).coalesce()
    A_norm = _build_sparse_adj(edge_index, n_nodes, dev)

    x0 = torch.tensor(node_features, dtype=torch.float32, device=dev)

    # Build GNN
    arch_l = str(arch).lower().strip()
    gnn_layers = nn.ModuleList()
    in_dim = int(x0.shape[1])
    for i in range(int(layers)):
        out_dim = int(hidden_dim) if i < int(layers) - 1 else int(hidden_dim)
        if arch_l == "gin":
            gnn_layers.append(GINLayer(in_dim, out_dim, dropout=dropout))
        else:
            gnn_layers.append(GraphSAGELayer(in_dim, out_dim, dropout=dropout))
        in_dim = out_dim
    decoder = EdgeDecoder(node_dim=int(hidden_dim), hidden_dim=int(hidden_dim), dropout=dropout)

    model = nn.ModuleDict({"gnn": gnn_layers, "dec": decoder}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    # Prepare train/test edges (by integer IDs)
    def _to_edge_tensors(df_: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.tensor(df_["source_id"].astype(int).to_numpy(), dtype=torch.long, device=dev)
        t = torch.tensor(df_["target_id"].astype(int).to_numpy(), dtype=torch.long, device=dev)
        y = torch.tensor(df_["label"].astype(int).to_numpy(), dtype=torch.float32, device=dev)
        return s, t, y

    tr_s, tr_t, tr_y = _to_edge_tensors(train_df)
    te_s, te_t, te_y = _to_edge_tensors(test_df)

    def _encode() -> torch.Tensor:
        h = x0
        for layer in model["gnn"]:
            if isinstance(layer, GINLayer):
                h = layer(h, A)
            else:
                h = layer(h, A_norm)
        return h

    t0 = time.time()
    model.train()
    for _epoch in range(int(epochs)):
        opt.zero_grad(set_to_none=True)
        h = _encode()
        logits = model["dec"](h, tr_s, tr_t)
        loss = F.binary_cross_entropy_with_logits(logits, tr_y)
        loss.backward()
        opt.step()
    train_seconds = float(time.time() - t0)

    # Evaluate
    model.eval()
    with torch.no_grad():
        h = _encode()
        te_logits = model["dec"](h, te_s, te_t).detach().cpu().numpy()
        y_true = te_y.detach().cpu().numpy()

    m = _score_binary(y_true, te_logits)
    return GNNRunResult(
        model_name=f"GNN-{arch_l.upper()}",
        pr_auc=float(m["pr_auc"]),
        roc_auc=float(m["roc_auc"]),
        accuracy=float(m["accuracy"]),
        precision=float(m["precision"]),
        recall=float(m["recall"]),
        f1=float(m["f1"]),
        train_seconds=train_seconds,
        n_train=int(len(train_df)),
        n_test=int(len(test_df)),
        n_nodes=int(n_nodes),
        n_edges=int(n_edges),
    )

