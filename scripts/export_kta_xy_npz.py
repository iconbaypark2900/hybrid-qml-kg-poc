#!/usr/bin/env python3
"""
Export binary-labeled quantum-input features ``X`` and ``y`` as ``.npz`` for KTA tooling.

Feeds ``scripts/compute_kta_zz_vs_pauli_subset.py --npz ...`` with arrays ``X`` (n, qml_dim)
and ``y`` in ``{0, 1}``.

Feature path mirrors lightweight stacks (``scripts/e2e_smoke.py``, ``scripts/train_on_heron.py``):
``HetionetEmbedder`` + ``prepare_link_features_qml``. Full ``run_optimized_pipeline.py`` may
instead build QML tensors via ``AdvancedQMLFeatureEngineer`` and ``--qml_pre_pca_dim``; use those
runs for strict paper parity — this exporter is intended for seconds-scale KTA pre-screens.

Example:

    python scripts/export_kta_xy_npz.py --out results/kta_train_subset.npz \\
      --subset 100 --qml_dim 16
    python scripts/compute_kta_zz_vs_pauli_subset.py --npz results/kta_train_subset.npz --qml_dim 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from kg_layer.kg_embedder import HetionetEmbedder  # noqa: E402
from kg_layer.kg_loader import (  # noqa: E402
    extract_task_edges,
    load_hetionet_edges,
    prepare_link_prediction_dataset,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Write train-split (X,y) npz for KTA subset scripts.")
    p.add_argument("--relation", type=str, default="CtD")
    p.add_argument("--max_entities", type=int, default=200)
    p.add_argument("--embedding_dim", type=int, default=64)
    p.add_argument("--qml_dim", type=int, default=16)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument(
        "--qml_features_mode",
        type=str,
        default="diff",
        choices=("diff", "hadamard", "both"),
        help="Passes through to HetionetEmbedder.prepare_link_features_qml",
    )
    p.add_argument(
        "--subset",
        type=int,
        default=None,
        help="If set, shuffle and keep this many train rows (stratified if possible).",
    )
    p.add_argument("--out", type=str, required=True, help="Output path, e.g. results/kta_xy.npz")
    args = p.parse_args()

    df = load_hetionet_edges()
    task_edges, _, _ = extract_task_edges(
        df,
        relation_type=args.relation,
        max_entities=args.max_entities,
    )
    train_df, _test_df = prepare_link_prediction_dataset(task_edges, random_state=args.random_state)

    embedder = HetionetEmbedder(
        embedding_dim=args.embedding_dim,
        qml_dim=args.qml_dim,
    )
    if not embedder.load_saved_embeddings():
        embedder.train_embeddings(task_edges)

    embedder.reduce_to_qml_dim()
    X = embedder.prepare_link_features_qml(train_df, mode=args.qml_features_mode)
    y = train_df["label"].astype(int).values

    if X.shape[0] != len(y):
        raise SystemExit("internal error: X rows != y length")

    uniq = np.unique(y)
    if not np.all(np.isin(uniq, [0, 1])):
        raise SystemExit(f"labels must be binary 0/1; got {uniq}")

    if args.subset is not None:
        rng = np.random.default_rng(args.random_state)
        n = min(int(args.subset), len(X))
        sel = rng.permutation(len(X))[:n]
        X, y = X[sel], y[sel]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(args.out, X=X.astype(np.float64), y=y.astype(np.int64))
    print(
        f"wrote {args.out}  X{X.shape}  y{y.shape}  "
        f"pos={int(np.sum(y == 1))} neg={int(np.sum(y == 0))}  "
        f"relation={args.relation}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
