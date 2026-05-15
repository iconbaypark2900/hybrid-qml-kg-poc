"""Pytest configuration and shared fixtures for hybrid-qml-kg-poc.

Provides deterministic synthetic-KG fixtures so the test suite can run
end-to-end without Hetionet downloads or PyKEEN-trained embeddings. See
``tests/fixtures/synthetic_kg.py`` for the underlying generator.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make the repository root importable so tests can `from utils...` and
# `from tests.fixtures...` without installing the project.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.fixtures.synthetic_kg import (  # noqa: E402
    SyntheticKGConfig,
    generate_link_prediction_instances,
    generate_synthetic_kg,
    split_instances,
)
from utils.preregistered_constants import (  # noqa: E402
    BOOTSTRAP_SEED,
    SPLIT_SEED,
    TRAIN_FRAC,
    VAL_FRAC,
)


@pytest.fixture(scope="session")
def synthetic_kg_config():
    """Default synthetic-KG configuration (Hetionet-flavored, small)."""
    return SyntheticKGConfig()


@pytest.fixture(scope="session")
def synthetic_kg_small(synthetic_kg_config):
    """Deterministic small synthetic biomedical KG.

    Seeded with ``BOOTSTRAP_SEED`` (20260504) so two runs of the suite
    produce bit-identical graphs.
    """
    return generate_synthetic_kg(synthetic_kg_config, seed=BOOTSTRAP_SEED)


@pytest.fixture(scope="session")
def synthetic_link_prediction_instances(synthetic_kg_small):
    """CtD link-prediction instances from the synthetic KG, 1:1 random negatives."""
    return generate_link_prediction_instances(
        synthetic_kg_small,
        target_edge_type="CtD",
        n_positive=40,
        n_negative=40,
        seed=BOOTSTRAP_SEED + 1,
    )


@pytest.fixture(scope="session")
def synthetic_split(synthetic_link_prediction_instances):
    """Train/val/test split (70/15/15) of the synthetic CtD instances."""
    train, val, test = split_instances(
        synthetic_link_prediction_instances,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        seed=SPLIT_SEED,
    )
    return {"train": train, "val": val, "test": test}
