#!/usr/bin/env python3
"""Compute the degree-heuristic baseline PR-AUC on the existing CtD test split.

Ranks test pairs by compound_degree * disease_degree (training graph only)
and reports average_precision_score. Also reports the random baseline (0.50
for a balanced test set). Outputs JSON for reproducibility.

Run:  .venv/bin/python scripts/degree_heuristic_baseline.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kg_layer.kg_loader import load_hetionet_edges, extract_task_edges
from scripts.run_optimized_pipeline import _make_split_with_negatives


def compute_degree_heuristic_pr_auc(
    *,
    relation: str = "CtD",
    test_size: float = 0.2,
    random_state: int = 42,
    neg_ratio: float = 1.0,
    negative_sampling: str = "random",
    max_entities: int = 0,
) -> dict:
    df = load_hetionet_edges()
    task_edges, entity_to_id, id_to_entity = extract_task_edges(
        df, relation_type=relation, max_entities=max_entities
    )

    train_df, test_df = _make_split_with_negatives(
        task_edges,
        test_size=test_size,
        random_state=random_state,
        neg_ratio=neg_ratio,
        negative_sampling=negative_sampling,
        id_to_entity=id_to_entity,
    )

    train_pos = train_df[train_df["label"] == 1]
    compound_degree = train_pos["source"].value_counts().to_dict()
    disease_degree = train_pos["target"].value_counts().to_dict()

    scores = np.array(
        [
            compound_degree.get(c, 0) * disease_degree.get(d, 0)
            for c, d in zip(
                test_df["source"].astype(str), test_df["target"].astype(str)
            )
        ],
        dtype=float,
    )
    labels = test_df["label"].values.astype(int)

    degree_pr_auc = float(average_precision_score(labels, scores))
    random_pr_auc = 0.50
    test_pos_rate = float(labels.mean())

    result = {
        "relation": relation,
        "test_size": test_size,
        "random_state": random_state,
        "neg_ratio": neg_ratio,
        "negative_sampling": negative_sampling,
        "max_entities": max_entities,
        "n_test_pairs": int(len(labels)),
        "n_test_pos": int(labels.sum()),
        "n_test_neg": int((labels == 0).sum()),
        "test_pos_rate": test_pos_rate,
        "n_train_pos": int((train_df["label"] == 1).sum()),
        "n_compounds_with_degree": len(compound_degree),
        "n_diseases_with_degree": len(disease_degree),
        "degree_heuristic_pr_auc": degree_pr_auc,
        "random_baseline_pr_auc": random_pr_auc,
        "note": (
            "Degree = compound_train_degree * disease_train_degree on training "
            "positives only. Random baseline = positive rate for a balanced "
            "test set (0.50)."
        ),
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    return result


def main() -> int:
    result = compute_degree_heuristic_pr_auc()
    print(json.dumps(result, indent=2))

    out = REPO_ROOT / "results" / "degree_heuristic_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved: {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
