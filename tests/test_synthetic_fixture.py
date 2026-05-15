"""Smoke tests for the synthetic-KG fixture.

Verifies determinism (same seed -> bit-identical KG), basic shape
(instance counts and feature dimension match the configuration), and
non-trivial separability (the fixture's features are signal-bearing
but do not leak the label, so a simple model lands strictly between
random and perfect).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from tests.fixtures.synthetic_kg import (
    SyntheticKGConfig,
    generate_link_prediction_instances,
    generate_synthetic_kg,
    instances_to_arrays,
    split_instances,
)
from utils.bootstrap_ci import paired_bootstrap_pr_auc_difference
from utils.preregistered_constants import BOOTSTRAP_SEED, TRAIN_FRAC, VAL_FRAC


def test_synthetic_kg_is_deterministic():
    config = SyntheticKGConfig()
    kg_a = generate_synthetic_kg(config, seed=BOOTSTRAP_SEED)
    kg_b = generate_synthetic_kg(config, seed=BOOTSTRAP_SEED)

    assert kg_a.n_nodes == kg_b.n_nodes
    np.testing.assert_array_equal(kg_a.node_types, kg_b.node_types)
    assert kg_a.edges.keys() == kg_b.edges.keys()
    for et in kg_a.edges:
        assert kg_a.edges[et] == kg_b.edges[et]


def test_link_prediction_instance_shape():
    config = SyntheticKGConfig()
    kg = generate_synthetic_kg(config, seed=BOOTSTRAP_SEED)
    instances = generate_link_prediction_instances(
        kg,
        target_edge_type="CtD",
        n_positive=40,
        n_negative=40,
        seed=BOOTSTRAP_SEED + 1,
    )

    assert len(instances) == 80
    assert sum(1 for i in instances if i.label == 1) == 40
    assert sum(1 for i in instances if i.label == 0) == 40

    X, y = instances_to_arrays(instances)
    assert X.shape[0] == 80
    assert X.ndim == 2
    assert y.shape == (80,)
    assert set(np.unique(y).tolist()) == {0, 1}

    train, val, test = split_instances(
        instances, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=BOOTSTRAP_SEED
    )
    assert len(train) + len(val) + len(test) == len(instances)
    # 70/15/15 of 40 positives = 28/6/6 (rounding); same for negatives.
    n_train_pos = sum(1 for i in train if i.label == 1)
    n_train_neg = sum(1 for i in train if i.label == 0)
    assert n_train_pos == 28
    assert n_train_neg == 28


def test_fixture_is_separable_but_not_trivial():
    """Fixture should carry signal (PR-AUC > 0.55) but not leak the label (< 0.99).

    Verifies the de-trivialization fix: the 1-hop direct-edge feature for
    the target edge type is omitted, so a logistic regression cannot
    achieve perfect separation. The remaining 2-hop / common-neighbor /
    node-type features still provide above-random signal.
    """
    config = SyntheticKGConfig()
    kg = generate_synthetic_kg(config, seed=BOOTSTRAP_SEED)
    instances = generate_link_prediction_instances(
        kg,
        target_edge_type="CtD",
        n_positive=60,
        n_negative=60,
        seed=BOOTSTRAP_SEED + 1,
    )
    train, _val, test = split_instances(
        instances, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=BOOTSTRAP_SEED
    )
    X_train, y_train = instances_to_arrays(train)
    X_test, y_test = instances_to_arrays(test)

    lr = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
    scores = lr.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, scores)

    # Signal present but not leaking — the bounds are deliberately loose
    # because exact PR-AUC depends on the fixture's particular RNG state.
    assert pr_auc > 0.55, f"fixture too weak (PR-AUC = {pr_auc:.4f}); features carry no signal"
    assert pr_auc < 0.99, f"fixture leaks label (PR-AUC = {pr_auc:.4f}); direct-edge feature still present?"


def test_paired_bootstrap_returns_non_degenerate_ci():
    """The bootstrap helper produces a finite CI with strictly positive width
    when the two score vectors differ.

    Catches the regression where a perfectly-separable fixture (PR-AUC 1.0
    for every model) gave a degenerate `[0.0, 0.0]` CI.
    """
    config = SyntheticKGConfig()
    kg = generate_synthetic_kg(config, seed=BOOTSTRAP_SEED)
    instances = generate_link_prediction_instances(
        kg,
        target_edge_type="CtD",
        n_positive=60,
        n_negative=60,
        seed=BOOTSTRAP_SEED + 1,
    )
    train, _val, test = split_instances(
        instances, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=BOOTSTRAP_SEED
    )
    X_train, y_train = instances_to_arrays(train)
    X_test, y_test = instances_to_arrays(test)

    lr = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
    scores_a = lr.predict_proba(X_test)[:, 1]
    # Rank-changing perturbation: Gaussian noise reorders some scores so
    # PR-AUC(scores_a) != PR-AUC(scores_b) and the bootstrap CI has width.
    # A monotone transform (e.g. scores * 0.9 + 0.05) preserves rank and
    # therefore preserves PR-AUC, giving a degenerate `[0, 0]` CI.
    rng = np.random.default_rng(BOOTSTRAP_SEED + 99)
    scores_b = np.clip(scores_a + rng.normal(0.0, 0.2, size=scores_a.shape), 0.001, 0.999)

    point, lo, hi = paired_bootstrap_pr_auc_difference(
        scores_a, y_test, scores_b, y_test, n_resamples=500
    )
    assert np.isfinite([point, lo, hi]).all()
    assert hi > lo, f"CI is degenerate: lo={lo}, hi={hi}"
