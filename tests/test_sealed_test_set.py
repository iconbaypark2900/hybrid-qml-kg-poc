"""Tests for ``utils.sealed_test_set.SealedTestSet``.

Verifies:
  - Construction validates shapes and matched row counts.
  - ``sha256()`` is content- and shape-stable.
  - ``unseal()`` returns the same arrays and appends an audit log entry.
  - Empty reason or methodology_version is rejected.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from utils.sealed_test_set import SealedTestSet


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    y = rng.integers(0, 2, size=20)
    return X, y


def test_construction_validates_shapes(sample_data):
    X, y = sample_data
    SealedTestSet(X, y)
    with pytest.raises(ValueError, match="must agree"):
        SealedTestSet(X, y[:10])
    with pytest.raises(ValueError, match="1-D"):
        SealedTestSet(X, y.reshape(-1, 1))


def test_sha256_is_stable(sample_data):
    X, y = sample_data
    a = SealedTestSet(X, y).sha256()
    b = SealedTestSet(X, y).sha256()
    assert a == b
    assert isinstance(a, str)
    assert len(a) == 64


def test_sha256_changes_when_data_changes(sample_data):
    X, y = sample_data
    a = SealedTestSet(X, y).sha256()
    X2 = X.copy()
    X2[0, 0] += 1.0
    b = SealedTestSet(X2, y).sha256()
    assert a != b


def test_unseal_returns_same_arrays(sample_data, tmp_path):
    X, y = sample_data
    log = tmp_path / "unseal.jsonl"
    sealed = SealedTestSet(X, y, log_path=str(log))
    Xu, yu = sealed.unseal(reason="test", methodology_version="abc1234")
    assert Xu is X
    assert yu is y


def test_unseal_appends_audit_record(sample_data, tmp_path):
    X, y = sample_data
    log = tmp_path / "unseal.jsonl"
    sealed = SealedTestSet(X, y, log_path=str(log))
    sealed.unseal(reason="first event", methodology_version="commit1")
    sealed.unseal(reason="second event", methodology_version="commit2")

    assert log.exists()
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    rec2 = json.loads(lines[1])
    assert rec1["reason"] == "first event"
    assert rec1["methodology_version"] == "commit1"
    assert rec1["n_rows"] == 20
    assert rec2["reason"] == "second event"
    # Both events reference the same sealed set, so the hash matches.
    assert rec1["test_set_sha256"] == rec2["test_set_sha256"]


def test_unseal_rejects_empty_strings(sample_data, tmp_path):
    X, y = sample_data
    sealed = SealedTestSet(X, y, log_path=str(tmp_path / "unseal.jsonl"))
    with pytest.raises(ValueError, match="reason"):
        sealed.unseal(reason="", methodology_version="commit")
    with pytest.raises(ValueError, match="reason"):
        sealed.unseal(reason="   ", methodology_version="commit")
    with pytest.raises(ValueError, match="methodology_version"):
        sealed.unseal(reason="why", methodology_version="")
